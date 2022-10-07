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
from datasets import DatasetDict

from transformers import (
    set_seed,
    Wav2Vec2Processor, 
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Config,
    Trainer,
    HfArgumentParser,
    EarlyStoppingCallback
)

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

from args_helper import AdditionalTrainingArguments, ModelArguments, DataArguments, TrainingArguments
from utils import CHARS_TO_IGNORE, tokenize_for_mer, tokenize_for_cer
from data_utils import speech_file_to_array_fn, load_dataset, DataCollatorCTCWithPadding

import datasets
from datasets import load_from_disk, set_caching_enabled

set_caching_enabled(True)
logger = logging.getLogger(__name__)    


def load_processor(model_args, training_args):
    # Load processor
    print('Load Wav2Vec2 processor...')

    def _get_pretrained_special_tokens(tokenizer=None):
        special_tokens = {}
        if tokenizer is None:
            special_tokens["bos_token"] = ("<s>", None)
            special_tokens["eos_token"] = ("</s>", None)
            special_tokens["unk_token"] = ("[UNK]", None)
            special_tokens["pad_token"] = ("<pad>", None)
            special_tokens["word_delimiter_token"] = ("|", None)
            special_tokens["do_lower_case"] = False
        else:
            special_tokens["bos_token"] = (tokenizer.bos_token, tokenizer.bos_token_id)
            special_tokens["eos_token"] = (tokenizer.eos_token, tokenizer.eos_token_id)
            special_tokens["unk_token"] = (tokenizer.unk_token, tokenizer.unk_token_id)
            special_tokens["pad_token"] = (tokenizer.pad_token, tokenizer.pad_token_id)
            special_tokens["word_delimiter_token"] = (tokenizer.word_delimiter_token, tokenizer.word_delimiter_token_id)
            special_tokens["do_lower_case"] = tokenizer.do_lower_case
        return special_tokens

    try:
        pretrained_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_args.model_name_or_path)
        special_tokens = _get_pretrained_special_tokens(pretrained_tokenizer)
        pretrained_vocab = list(map(lambda x: x[0], sorted(pretrained_tokenizer.get_vocab().items(), key=lambda x: x[1])))
    except:
        special_tokens = _get_pretrained_special_tokens()
        pretrained_vocab = []
        

    logger.info("Vocab length (initial): {}".format(len(pretrained_vocab)))
    print("Vocab length (initial):", len(pretrained_vocab))

    with open("{}/new_vocab.json".format(training_args.output_dir), "r") as new_vocab_file:
        new_vocab_list = json.load(new_vocab_file)
        logger.info("New vocabulary length: {}".format(len(new_vocab_list)))

    all_vocab = list(dict.fromkeys(pretrained_vocab + new_vocab_list))

    vocab_dict = {v: k for k, v in enumerate(all_vocab)}

    def _assign_id_to_special_tokens(special_tokens, vocab_dict):

        def __get_key_by_value(dict, value):
            return (list(dict.keys())[list(dict.values()).index(value)])

        bos_token = special_tokens["bos_token"][0]
        if bos_token not in vocab_dict:
            bos_token_id = special_tokens["bos_token"][1]
            if bos_token_id is None:
                bos_token_id = 1 # common token id for bos in config.json
            vocab_dict[__get_key_by_value(vocab_dict, bos_token_id)] = len(vocab_dict)
            vocab_dict[bos_token] = bos_token_id

        eos_token = special_tokens["eos_token"][0]
        if eos_token not in vocab_dict:
            eos_token_id = special_tokens["eos_token"][1]
            if eos_token_id is None:
                eos_token_id = 2 # common token id for eos in config.json
            vocab_dict[__get_key_by_value(vocab_dict, eos_token_id)] = len(vocab_dict)
            vocab_dict[eos_token] = eos_token_id

        pad_token = special_tokens["pad_token"][0]
        if pad_token not in vocab_dict:
            pad_token_id = special_tokens["pad_token"][1]
            if pad_token_id is None:
                pad_token_id = 0 # common token id for pad in config.json
            vocab_dict[__get_key_by_value(vocab_dict, pad_token_id)] = len(vocab_dict)
            vocab_dict[pad_token] = pad_token_id

        unk_token = special_tokens["unk_token"][0]
        if unk_token not in vocab_dict:
            unk_token_id = special_tokens["unk_token"][1]
            if unk_token_id is None:
                unk_token_id = 3 # common token id for unk, following jonatangrosman's setting
            vocab_dict[__get_key_by_value(vocab_dict, unk_token_id)] = len(vocab_dict)
            vocab_dict[unk_token] = unk_token_id

        word_delimiter_token = special_tokens["word_delimiter_token"][0]
        if word_delimiter_token not in vocab_dict:
            word_delimiter_token_id = special_tokens["word_delimiter_token"][1]
            if word_delimiter_token_id is None:
                word_delimiter_token_id = 4 # common token id for word delimiter, following jonatangrosman's setting    
            vocab_dict[__get_key_by_value(vocab_dict, word_delimiter_token_id)] = len(vocab_dict)
            vocab_dict[word_delimiter_token] = word_delimiter_token_id

        return vocab_dict

    vocab_dict = _assign_id_to_special_tokens(special_tokens, vocab_dict)
    print("len vocab dict", len(vocab_dict))

    with open("{}/all_vocab.json".format(training_args.output_dir), "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)
        
    tokenizer = Wav2Vec2CTCTokenizer(
        "{}/all_vocab.json".format(training_args.output_dir),
        bos_token=special_tokens["bos_token"][0],
        eos_token=special_tokens["eos_token"][0],
        pad_token=special_tokens["pad_token"][0],
        unk_token=special_tokens["unk_token"][0],
        word_delimiter_token=special_tokens["word_delimiter_token"][0],
        do_lower_case=special_tokens["do_lower_case"],
    )

    logger.info("Vocab size (final): {}".format(tokenizer.vocab_size))
    print("Vocab size (final):", tokenizer.vocab_size)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_args.model_name_or_path)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    return processor
    
#####
# Main Functions
#####
def run(model_args, data_args, training_args, additional_training_args):
    ###
    # Prepare Processor & Model    
    ###
    if training_args.output_path is None:
        training_args.output_dir="{}/{}".format(training_args.output_dir, model_args.model_name_or_path)
    else:
        training_args.output_dir = training_args.output_path
    os.makedirs(training_args.output_dir, exist_ok=True)

    if data_args.cache_path is None:
        cache_dir_path = "./{}/{}".format(data_args.cache_dir_name, model_args.model_name_or_path)
    else:
        cache_dir_path = data_args.cache_path
    os.makedirs(cache_dir_path, exist_ok=True)
    print('cache_dir_path', cache_dir_path)

    if not os.path.exists("{}/preprocess_data.arrow".format(cache_dir_path)):
        ###
        # Prepare Dataset
        ###
        lang = additional_training_args.lang.split(",")
        print('LANGUAGE TYPES USED: {}'.format(lang))

        raw_datasets = DatasetDict()
        print('Loading train dataset...')
        raw_datasets["train"] = load_dataset(data_args.train_manifest_path, data_args.preprocessing_num_workers, 
                                        data_args.audio_column_name, data_args.text_column_name, lang=lang, print_duration=False)
        print('Loading validation dataset...')
        raw_datasets["valid"] = load_dataset(data_args.valid_manifest_path, data_args.preprocessing_num_workers, 
                                        data_args.audio_column_name, data_args.text_column_name, lang=lang, print_duration=False)
        print('Loading test dataset...')
        raw_datasets["test"] = load_dataset(data_args.test_manifest_path, data_args.preprocessing_num_workers, 
                                        data_args.audio_column_name, data_args.text_column_name, lang=lang, print_duration=False)

        print('Preprocess dataset...')

        # Remove ignorable characters
        print('Removing ignorable characters')
        chars_to_ignore_re = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"
        def remove_special_characters(batch):
            if chars_to_ignore_re is not None:
                batch[data_args.text_column_name] = re.sub(chars_to_ignore_re, "", batch[data_args.text_column_name]) + " "
            else:
                batch[data_args.text_column_name] = batch[data_args.text_column_name] + " "
            return batch

        with training_args.main_process_first(desc="dataset map special characters removal"):
            raw_datasets = raw_datasets.map(
                remove_special_characters,
                num_proc=data_args.preprocessing_num_workers,
                desc="remove special characters from datasets",
                load_from_cache_file=True,
                cache_file_names={
                    "train": "{}/train_clean.arrow".format(cache_dir_path),
                    "valid": "{}/valid_clean.arrow".format(cache_dir_path),
                    "test": "{}/test_clean.arrow".format(cache_dir_path),
                }
            )

        # Build vocabulary
        print('Build vocabulary...')
        def extract_all_chars(batch):
            all_text = " ".join(batch[data_args.text_column_name])
            vocab = list(set(all_text))
            return {"vocab": [vocab], "all_text": [all_text]}

        with training_args.main_process_first(desc="vocab building"):
            _vocab = raw_datasets.map(
                extract_all_chars,
                num_proc=data_args.preprocessing_num_workers,
                desc="build vocabulary",
                load_from_cache_file=True,
                cache_file_names={
                    "train": "{}/train_vocab.arrow".format(cache_dir_path),
                    "valid": "{}/valid_vocab.arrow".format(cache_dir_path),
                    "test": "{}/test_vocab.arrow".format(cache_dir_path),
                }
            )

            def flatten(vocab_split):
                return list(chain.from_iterable(list(chain.from_iterable(vocab_split))))

            vocab_list = list(set(flatten(_vocab["train"]["vocab"]) + flatten(_vocab["valid"]["vocab"]) + flatten(_vocab["test"]["vocab"])))
            # vocab_dict = {v: k for k, v in enumerate(vocab_list)}
            # vocab_dict["|"] = vocab_dict[" "]
            # vocab_dict["[UNK]"] = len(vocab_dict)
            # vocab_dict["[PAD]"] = len(vocab_dict)

            # Dump vocabulary
            with open("{}/new_vocab.json".format(training_args.output_dir), "w") as vocab_file:
                json.dump(vocab_list, vocab_file)

        # Load processor
        processor = load_processor(model_args, training_args)

        # Preprocess audio sample and label text
        print('Vectorize dataset...')

        def prepare_dataset(batch):
            # Preprocess audio
            batch["input_values"] = processor(batch["speech_sample"], sampling_rate=16000).input_values[0]

            # Preprocess text
            with processor.as_target_processor():
                batch["labels"] = processor(batch[data_args.text_column_name]).input_ids

            return batch

        with training_args.main_process_first(desc="dataset map preprocessing"):
            vectorized_datasets = raw_datasets.map(
                prepare_dataset,
                remove_columns=raw_datasets["train"].column_names,
                num_proc=data_args.preprocessing_num_workers,
                desc="preprocess datasets",
                load_from_cache_file=True,
                cache_file_names={
                    "train": "{}/train_vec.arrow".format(cache_dir_path),
                    "valid": "{}/valid_vec.arrow".format(cache_dir_path),
                    "test": "{}/test_vec.arrow".format(cache_dir_path),
                }
            )
        
        vectorized_datasets.save_to_disk("{}/preprocess_data.arrow".format(cache_dir_path))
    else:
        print('Loading cached dataset...')
        vectorized_datasets = datasets.load_from_disk('{}/preprocess_data.arrow'.format(cache_dir_path))

        # Load processor
        processor = load_processor(model_args, training_args)

    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}")
        return
    
    ###
    # Prepare Data Collator and Trainer
    ###
    print('Preparing Trainer...')

    print('Load Wav2Vec2 model...')
    print('Model ID', model_args.model_name_or_path)
    config = Wav2Vec2Config.from_pretrained(model_args.model_name_or_path)
    config.update({
        "mask_time_prob": model_args.mask_time_prob,
        "mask_time_length": model_args.mask_time_length,
        "mask_feature_prob": model_args.mask_feature_prob,
        "mask_feature_length": model_args.mask_feature_length,
        "gradient_checkpointing": training_args.gradient_checkpointing,
    })
    model = Wav2Vec2ForCTC.from_pretrained(model_args.model_name_or_path, config=config)
    model.cuda()

    def _resize_token_embeddings(model, new_num_tokens):
        old_lm_head = model.lm_head
        new_lm_head = model._get_resized_lm_head(old_lm_head, new_num_tokens)
        model.lm_head = new_lm_head
        model.config.update({"vocab_size": new_num_tokens})
        return model

    model = _resize_token_embeddings(model, processor.tokenizer.vocab_size)


    # Instantiate custom data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Define compute metric function
    def compute_metrics(pred):
        logger.info("*** Compute metrics ***")
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_strs = processor.batch_decode(pred_ids)
        # pred_strs = [s.replace("[PAD]", "") for s in pred_strs]
        print(pred_strs)

        # we do not want to group tokens when computing the metrics
        label_strs = processor.batch_decode(pred.label_ids, group_tokens=False)

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
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["valid"],
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=additional_training_args.early_stopping_patience)]
    )

    ###
    # Training Phase
    ###
    print('*** Training Phase ***')
    
    # use last checkpoint if exist
    if os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    # Save the feature_extractor and the tokenizer
    if is_main_process(training_args.local_rank):
        processor.save_pretrained(training_args.output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(vectorized_datasets["train"])

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    ###
    # Evaluation Phase
    ###
    results = {}
    logger.info("*** Evaluation Phase ***")
    metrics = trainer.evaluate(eval_dataset=vectorized_datasets["valid"])
    metrics["eval_samples"] = len(vectorized_datasets["valid"])

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
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
    os.makedirs("./log", exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(
            "./log/log__{}".format(model_args.model_name_or_path.replace("/", "_")), mode="w")],
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