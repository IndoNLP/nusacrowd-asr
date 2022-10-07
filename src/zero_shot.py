import os, sys
import logging
import numpy as np
import pandas as pd
import argparse

import torchaudio
import torch
import re
import json 
import librosa
from datasets import load_from_disk, load_dataset, load_metric

from transformers import (
    set_seed,
    Wav2Vec2Processor, 
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Config,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    EarlyStoppingCallback
)

from datasets import DatasetDict, load_metric, load_from_disk
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import datasets
import pickle

import editdistance
import jieba
from itertools import chain

import transformers
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from args_helper import ModelArguments, DataArguments, AdditionalTrainingArguments, TrainingArguments
from utils import CHARS_TO_IGNORE, remove_special_characters, tokenize_for_mer, tokenize_for_cer
from data_utils import speech_file_to_array_fn, load_dataset, DataCollatorCTCWithPadding
from datasets import set_caching_enabled

set_caching_enabled(True)    
logger = logging.getLogger(__name__)

#####
# Main Functions
#####
def run(model_args, data_args, training_args, additional_training_args):
    ###
    # Prepare Processor & Model    
    ###
    print('Load Wav2Vec2 model and processor...')
    config = Wav2Vec2Config.from_pretrained(model_args.model_name_or_path)
    config.update({
        "mask_time_prob": 0,
        "mask_time_length": 0,
        "mask_feature_prob": 0,
        "mask_feature_length": 0,
        "gradient_checkpointing": True,
    })
    # if len(model_args.model_name_or_path.split('/')[:-1]) > 1:
    #     parent_model_path = '/'.join(model_args.model_name_or_path.split('/')[:-1])
    #     processor = Wav2Vec2Processor.from_pretrained(parent_model_path)
    # else:
    processor = Wav2Vec2Processor.from_pretrained("baselines/save/scottykwok/wav2vec2-large-xlsr-cantonese")
    model = Wav2Vec2ForCTC.from_pretrained(model_args.model_name_or_path, config=config)
    model.cuda()

    def _resize_token_embeddings(model, new_num_tokens):
        old_lm_head = model.lm_head
        new_lm_head = model._get_resized_lm_head(old_lm_head, new_num_tokens)
        model.lm_head = new_lm_head
        model.config.update({"vocab_size": new_num_tokens})
        return model

    model = _resize_token_embeddings(model, processor.tokenizer.vocab_size)
    
    cache_dir_path = data_args.cache_dir_name
    print('cache_dir_path', cache_dir_path)

    lang = additional_training_args.lang.split(",")
    print('LANGUAGE TYPES USED: {}'.format(lang))
    
    ###
    # Prepare Dataset
    ###
    raw_datasets = DatasetDict()

    print('Loading valid dataset...')
    raw_datasets["valid"] = load_dataset(data_args.valid_manifest_path, data_args.preprocessing_num_workers, 
                                    data_args.audio_column_name, data_args.text_column_name, lang=lang)

    print('Loading test dataset...')
    raw_datasets["test"] = load_dataset(data_args.test_manifest_path, data_args.preprocessing_num_workers, 
                                    data_args.audio_column_name, data_args.text_column_name, lang=lang)

    print('Preprocess dataset...')

    # Remove ignorable characters
    print('Removing ignorable characters')
    chars_to_ignore_re = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"
    def remove_special_characters(batch):
        if chars_to_ignore_re is not None:
            batch[data_args.text_column_name] = re.sub(chars_to_ignore_re, "", batch[data_args.text_column_name]).upper() + " "
        else:
            batch[data_args.text_column_name] = batch[data_args.text_column_name].upper() + " "
        return batch

    with training_args.main_process_first(desc="dataset map special characters removal"):
        raw_datasets = raw_datasets.map(
            remove_special_characters,
            num_proc=data_args.preprocessing_num_workers,
            desc="remove special characters from datasets",
            load_from_cache_file=False
        )

    # Preprocess audio sample and label text
    print('Vectorize dataset...')

    def prepare_dataset(batch):
        # Preprocess audio
        batch["input_values"] = processor(batch["speech_sample"]).input_values[0]

        # Preprocess text
        with processor.as_target_processor():
            batch["labels"] = processor(batch[data_args.text_column_name]).input_ids
            # print(processor.batch_decode(batch["labels"]))

        return batch

    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=raw_datasets["test"].column_names,
            num_proc=data_args.preprocessing_num_workers,
            desc="preprocess datasets",
            load_from_cache_file=False
        )
    
    # vectorized_datasets.save_to_disk('./cache/preprocess_data.arrow')
    
    ###
    # Prepare Data Collator and Trainer
    ###
    print('Preparing Trainer...')
    
    # Instantiate custom data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Define compute metric function
    def compute_metrics(pred):
        logger.info("*** Compute metrics ***")
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_strs = processor.batch_decode(pred_ids)

        # we do not want to group tokens when computing the metrics
        label_strs = processor.batch_decode(pred.label_ids, group_tokens=False)

        f = open(f'{training_args.output_dir}/test.results', 'w')
        f.writelines([item+'\n' for item in pred_strs])
        f.close()
        f = open(f'{training_args.output_dir}/test.label', 'w')
        f.writelines([item+'\n' for item in label_strs])
        f.close()

        def _what_language(sentence):
            def __contains_character(sentence, language="eng"):
                _sentence = re.sub(r"[UNK]", "", sentence)
                if language == "eng":
                    pattern = "([a-z]|[A-Z])+"
                elif language == "yue":
                    pattern = "[\\u4e00-\\u9fff]+"
                else:
                    return False
                return re.search(pattern, _sentence)

            eng = __contains_character(sentence, language="eng")
            yue = __contains_character(sentence, language="yue")
            if eng is not None and yue is not None:
                return "cs"
            elif eng is not None:
                return "eng"
            elif yue is not None:
                return "yue"
            else:
                return "others"

        # split based on language
        eng_pred_strs, eng_label_strs = [], []
        yue_pred_strs, yue_label_strs = [], []
        cs_pred_strs, cs_label_strs = [], []
        for i, (pred_str, label_str) in enumerate(zip(pred_strs, label_strs)):
            if _what_language(label_str) == "cs":
                cs_pred_strs.append(pred_str)
                cs_label_strs.append(label_str)
            elif _what_language(label_str) == "eng":
                eng_pred_strs.append(pred_str)
                eng_label_strs.append(label_str)
            else:
                yue_pred_strs.append(pred_str)
                yue_label_strs.append(label_str)

        print('cs', len(cs_label_strs), 'eng', len(eng_label_strs), 'yue', len(yue_label_strs))

        def _calculate_mer_and_cer(pred_strs, label_strs):
            if len(label_strs) == 0:
                return 0, 0
            else:
                mixed_distance, mixed_tokens = 0, 0
                char_distance, char_tokens = 0, 0
                for i, (pred_str, label_str) in enumerate(zip(pred_strs, label_strs)):
                    # Calculate 
                    m_pred = tokenize_for_mer(pred_str)
                    m_ref = tokenize_for_mer(label_str)
                    mixed_distance += editdistance.distance(m_pred, m_ref)
                    mixed_tokens += len(m_ref)

                    c_pred = tokenize_for_cer(pred_str)
                    c_ref = tokenize_for_cer(label_str)
                    char_distance += editdistance.distance(c_pred, c_ref)
                    char_tokens += len(c_ref)
                mer = mixed_distance / mixed_tokens
                cer = char_distance / char_tokens
                return mer, cer

        mer, cer = _calculate_mer_and_cer(pred_strs, label_strs)
        cs_mer, cs_cer = _calculate_mer_and_cer(cs_pred_strs, cs_label_strs)
        eng_mer, eng_cer = _calculate_mer_and_cer(eng_pred_strs, eng_label_strs)
        yue_mer, yue_cer = _calculate_mer_and_cer(yue_pred_strs, yue_label_strs)

        metrics = {
            "mer": mer, "cer": cer,
            "mer_cs": cs_mer, "cer_cs": cs_cer,
            "mer_eng": eng_mer, "cer_eng": eng_cer,
            "mer_yue": yue_mer, "cer_yue": yue_cer,
        }

        logger.info(json.dumps(metrics))
        return metrics
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=processor.feature_extractor,
    )

    ###
    # Evaluation Phase (Validation)
    ###
    results = {}
    logger.info("*** Valid Phase ***")
    metrics = trainer.evaluate(eval_dataset=vectorized_datasets["valid"])
    metrics["eval_samples"] = len(vectorized_datasets["valid"])

    trainer.log_metrics("valid", metrics)
    trainer.save_metrics("valid", metrics)
    
    ###
    # Evaluation Phase (Test)
    ###
    results = {}
    logger.info("*** Test Phase ***")
    metrics = trainer.evaluate(eval_dataset=vectorized_datasets["test"])
    metrics["eval_samples"] = len(vectorized_datasets["test"])

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "speech-recognition",
        "tags": ["automatic-speech-recognition", "ASCEND"],
        "dataset_args": "Config: na",
        "dataset": "ASCEND",
        "language": "yue-eng"
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results
    
#####
# Entry Point
#####
def main():
    ###
    # Parsing & Initialization
    ###
    # Parse argument
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AdditionalTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, additional_training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, additional_training_args = parser.parse_args_into_dataclasses()

    # Set random seed
    set_seed(training_args.seed)
    
    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    ###
    # Prepare logger
    ###
    # Init logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to warn of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity(transformers.logging.WARNING)
    logger.info("Training/evaluation parameters %s", training_args)
    
    ###
    # RUN RUN RUN!!!
    ###
    run(model_args, data_args, training_args, additional_training_args)
    
if __name__ == '__main__':
    main()