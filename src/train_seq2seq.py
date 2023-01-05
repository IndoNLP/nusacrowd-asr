import logging
import os
import re
import sys

import datasets
import evaluate
import numpy as np
import torch
import transformers
from datasets import DatasetDict, load_from_disk, set_caching_enabled
from transformers import (AutoConfig, AutoFeatureExtractor,
                          AutoModelForSpeechSeq2Seq, WhisperProcessor,
                          AutoTokenizer, EarlyStoppingCallback,
                          HfArgumentParser, Seq2SeqTrainer, set_seed)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from utils.args_helper import (AdditionalTrainingArguments, DataArguments,
                               ModelArguments, Seq2SeqTrainingArguments)
from utils.asr_utils import CHARS_TO_IGNORE
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.dataloader import load_speech_datasets, load_speech_task

set_caching_enabled(True)
logger = logging.getLogger(__name__)    


def load_processor(model_args, training_args, additional_training_args):
    # Load processor
    print('Load Whisper processor...')
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    logger.info("Vocab size: {}".format(tokenizer.vocab_size))
    print("Vocab size:", tokenizer.vocab_size)

    if additional_training_args.lang is not None:
        if additional_training_args.lang == "sun":
            language = "sundanese"
        elif additional_training_args.lang == "jav":
            language = "javanese"
        else:
            language = "indonesian"
    else:
        if "_sun" in additional_training_args.task_config_name or "_su" in additional_training_args.task_config_name:
            language = "sundanese"
        elif "_jav" in additional_training_args.task_config_name or "_jv" in additional_training_args.task_config_name:
            language = "javanese"
        else:
            language = "indonesian"

    # if data_args.language is not None:
    # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
    tokenizer.set_prefix_tokens(
        language=language,
        task="transcribe"
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.model_name_or_path)

    # Create a single speech processor
    feature_extractor.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor
    
#####
# Main Functions
#####
def run(model_args, data_args, training_args, additional_training_args):
    ###
    # Prepare Processor & Model    
    ###
    training_args.output_dir="{}/{}".format(training_args.output_dir, model_args.model_name_or_path)

    os.makedirs(training_args.output_dir, exist_ok=True)

    cache_dir_path = "./{}/{}".format(data_args.cache_dir_name, model_args.model_name_or_path)
    os.makedirs(cache_dir_path, exist_ok=True)

    print('cache_dir_path', cache_dir_path)
    if not os.path.exists("{}/preprocess_data.arrow".format(cache_dir_path)):
        ###
        # Prepare Dataset
        ###
        task_config_name = additional_training_args.task_config_name
        if task_config_name is not None:
            # Use task config name
            print('Loading dataset...')
            train_dataset, valid_dataset, test_dataset_dict = load_speech_task(task_config_name)
        else:
            # Use language
            lang = additional_training_args.lang.split(",") if ',' in additional_training_args.lang else additional_training_args.lang
            print('LANGUAGE TYPES USED: {}'.format(lang))

            print('Loading dataset...')
            train_dataset, valid_dataset, test_dataset_dict = load_speech_datasets(asr=True, tts=True, train_lang=lang)

        raw_datasets = DatasetDict()
        raw_datasets["train"] = train_dataset
        raw_datasets["valid"] = valid_dataset
        for config_name, test_dset in test_dataset_dict.items():
            raw_datasets[f"test_{config_name}"] = test_dset

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

        datasets.set_caching_enabled(True)

        with training_args.main_process_first(desc="dataset map special characters removal"):
            raw_datasets = raw_datasets.map(
                remove_special_characters,
                num_proc=data_args.preprocessing_num_workers,
                desc="remove special characters from datasets",
                load_from_cache_file=True,
                cache_file_names={
                    subset: f"{cache_dir_path}/{subset}_clean.arrow" for subset in raw_datasets.keys()
                }
            )

        # Load processor
        processor = load_processor(model_args, training_args, additional_training_args)

        # 6. Resample speech dataset if necessary
        dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
        if dataset_sampling_rate != processor.feature_extractor.sampling_rate:
            raw_datasets = raw_datasets.cast_column(
                data_args.audio_column_name,
                datasets.features.Audio(sampling_rate=processor.feature_extractor.sampling_rate)
            )

        # Preprocess audio sample and label text
        print('Vectorize dataset...')

        model_input_name = processor.feature_extractor.model_input_names[0]

        def prepare_dataset(batch):
            # Preprocess audio
            inputs = processor.feature_extractor(batch[data_args.audio_column_name]['array'], sampling_rate=16000)
            # Process audio length
            batch[model_input_name] = inputs.get(model_input_name)[0]
            batch["input_length"] = len(batch[data_args.audio_column_name]["array"])
            # Preprocess text
            batch["labels"] = processor.tokenizer(text=batch[data_args.text_column_name].lower()).input_ids
            return batch

        with training_args.main_process_first(desc="dataset map preprocessing"):
            vectorized_datasets = raw_datasets.map(
                prepare_dataset,
                remove_columns=raw_datasets["train"].column_names,
                num_proc=data_args.preprocessing_num_workers,
                desc="preprocess datasets",
                load_from_cache_file=True,
                cache_file_names={
                    subset: f"{cache_dir_path}/{subset}_vec.arrow" for subset in raw_datasets.keys()
                }
            )

        vectorized_datasets.save_to_disk("{}/preprocess_data.arrow".format(cache_dir_path))
    else:
        with training_args.main_process_first(desc="dataset map special characters removal"):
            print('Loading cached dataset...')
            vectorized_datasets = datasets.load_from_disk('{}/preprocess_data.arrow'.format(cache_dir_path))

        # Load processor
        processor = load_processor(model_args, training_args, additional_training_args)

    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}")
        return
    
    ###
    # Prepare Data Collator and Trainer
    ###
    print('Preparing Trainer...')

    print('Load Whisper model...')
    print('Model ID', model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.update({
        "use_cache": False,
    })
    config.save_pretrained(training_args.output_dir)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_args.model_name_or_path, config=config)

    # Instantiate custom data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Load metrics
    metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        print("pred_str |", pred_str[0], "--- label_str |", label_str[0])
        wer = metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["valid"],
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    ###
    # Training Phase
    ###
    print('*** Training Phase ***')
    
    # use last checkpoint if exist
    if model_args.checkpoint_path is not None:
        checkpoint = model_args.checkpoint_path
    else:
        checkpoint = None

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    metrics = train_result.metrics
    metrics["train_samples"] = len(vectorized_datasets["train"])

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    ###
    # Evaluation Phase
    ###
    logger.info("*** Evaluation Phase ***")
    for subset in vectorized_datasets.keys():
        if 'test_' not in subset:
            continue            
        
        subset_id = subset.replace('test_', '')
        print(f'Performing evaluation on `{subset_id}`')
        metrics = trainer.evaluate(eval_dataset=vectorized_datasets[subset])
        
        keys = list(metrics.keys())
        for key in keys:
            metrics[key.replace('eval_',f'eval_{subset_id}_')] = metrics[key]
            del metrics[key]
        metrics[f"eval_{subset}_samples"] = len(vectorized_datasets[subset])

        trainer.log_metrics(f"eval_{subset_id}", metrics)
        trainer.save_metrics(f"eval_{subset_id}", metrics)
    
#####
# Entry Point
#####
def main():
    ###
    # Parsing & Initialization
    ###
    # Parse argument
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, AdditionalTrainingArguments))
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
