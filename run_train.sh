################# ALL DATA

#######
## Task Specific
#######
CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=facebook/wav2vec2-large-xlsr-53 \
   --task_config_name indspeech_digit_cdsr_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=4 --per_device_eval_batch_size=8 --gradient_accumulation_steps 2 \
   --dataloader_num_workers=16 --dataloader_pin_memory --group_by_length \
   --seed=14045 --num_train_epochs=3 --learning_rate=5e-5 \
   --fp16 --fp16_backend=cuda_amp \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/direct_ft/ \
   --output_dir=./save/direct_ft/
   
#######
## Monolingual Sundanese
#######
CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=facebook/wav2vec2-large-xlsr-53 \
   --lang sun \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=4 --per_device_eval_batch_size=8 --gradient_accumulation_steps 2 \
   --dataloader_num_workers=16 --dataloader_pin_memory --group_by_length \
   --seed=14045 --num_train_epochs=3 --learning_rate=5e-5 \
   --fp16 --fp16_backend=cuda_amp \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/sun_lang/ \
   --output_dir=./save/sun_lang/
   
#######
## Monolingual Javanese
#######
CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=facebook/wav2vec2-large-xlsr-53 \
   --lang jav \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=4 --per_device_eval_batch_size=8 --gradient_accumulation_steps 2 \
   --dataloader_num_workers=16 --dataloader_pin_memory --group_by_length \
   --seed=14045 --num_train_epochs=3 --learning_rate=5e-5 \
   --fp16 --fp16_backend=cuda_amp \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/jav_lang/ \
   --output_dir=./save/jav_lang/

#######
## Multilingual All
#######
CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=facebook/wav2vec2-large-xlsr-53 \
   --lang all \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=2 --per_device_eval_batch_size=4 --gradient_accumulation_steps 4 \
   --dataloader_num_workers=16 --dataloader_pin_memory --group_by_length \
   --seed=14045 --num_train_epochs=3 --learning_rate=5e-5 \
   --fp16 --fp16_backend=cuda_amp \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/all_lang/ \
   --output_dir=./save/all_lang/
   
# # Wav2Vec 2.0 Cantonese
# CUDA_VISIBLE_DEVICES=4 python src/train.py --model_name_or_path=CAiRE/wav2vec2-large-xlsr-53-cantonese \
#    --train_manifest_path=./data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#    --valid_manifest_path=./data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#    --test_manifest_path=./data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#    --preprocessing_num_workers=16 --audio_column_name=audio_path --text_column_name=text_path \
#    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
#    --dataloader_num_workers=16 --dataloader_pin_memory --group_by_length \
#    --seed=14045 --num_train_epochs=100 --learning_rate=5e-5 \
#    --fp16 --fp16_backend=cuda_amp \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
#    --metric_for_best_model=mer --greater_is_better=False \
#    --gradient_checkpointing=True \
#    --cache_dir=./cache/all_lang/ \
#    --output_dir=./save/all_lang/

# # Wav2Vec 2.0 Multilingual
# CUDA_VISIBLE_DEVICES=4 python train.py --model_name_or_path=facebook/wav2vec2-large-xlsr-53 \
#    --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#    --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#    --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#    --preprocessing_num_workers=16 --audio_column_name=audio_path --text_column_name=text_path \
#    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
#    --dataloader_num_workers=16 --dataloader_pin_memory --group_by_length \
#    --seed=14045 --num_train_epochs=100 --learning_rate=5e-5 \
#    --fp16 --fp16_backend=cuda_amp \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
#    --metric_for_best_model=mer --greater_is_better=False \
#    --gradient_checkpointing=True \
#    --cache_dir=./cache/all_lang/ \
#    --output_dir=./save/all_lang/

################# CANTONESE AND CANTONESE CODE-SWITCHING ONLY

# # Wav2Vec 2.0 Multilingual
# CUDA_VISIBLE_DEVICES=0 python train.py --model_name_or_path=facebook/wav2vec2-large-xlsr-53 \
#    --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#    --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#    --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#    --preprocessing_num_workers=16 --audio_column_name=audio_path --text_column_name=text_path \
#    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
#    --dataloader_num_workers=16 --dataloader_pin_memory --group_by_length \
#    --seed=14045 --num_train_epochs=100 --learning_rate=5e-5 \
#    --fp16 --fp16_backend=amp \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=10 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
#    --metric_for_best_model=mer --greater_is_better=False \
#    --gradient_checkpointing=True \
#    --lang="yue,cs-yue" \
#    --cache_dir="cache_yue_cs-yue" \
#    --output_dir="./save_yue_cs-yue"