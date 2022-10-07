################# ALL DATA

# # Wav2Vec 2.0 English
# CUDA_VISIBLE_DEVICES=0 python eval.py \
#     --model_name_or_path=save/jonatasgrosman/wav2vec2-large-xlsr-53-english/checkpoint-25155 \
#     --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#     --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#     --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#     --cache_dir_name cache/jonatasgrosman/wav2vec2-large-xlsr-53-english \
#     --preprocessing_num_workers=16 \
#     --audio_column_name=audio_path --text_column_name=text_path \
#     --eval_accumulation_steps=10 \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./eval/eng \
#     --lang="cs-eng"
    
# # Wav2Vec 2.0 Cantonese
# CUDA_VISIBLE_DEVICES=0 python eval.py \
#     --model_name_or_path=./save_cs-yue_cs-eng/./pretrain/save/facebook/wav2vec2-large-xlsr-53/checkpoint-63424/checkpoint-20335 \
#     --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#     --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#     --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#     --cache_dir_name cache_cs-yue_cs-eng/pretrain/save/facebook/wav2vec2-large-xlsr-53/checkpoint-63424 \
#     --preprocessing_num_workers=16 \
#     --audio_column_name=audio_path --text_column_name=text_path \
#     --eval_accumulation_steps=10 \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./eval/yue \
#     --lang="yue,eng,cs-yue,cs-eng"

# Wav2Vec 2.0 Cantonese
CUDA_VISIBLE_DEVICES=1 python eval.py \
    --model_name_or_path=baselines/save/scottykwok/wav2vec2-large-xlsr-cantonese/checkpoint-112230 \
    --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
    --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
    --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
    --cache_dir_name baselines/cache/scottykwok/wav2vec2-large-xlsr-cantonese/checkpoint-112230 \
    --preprocessing_num_workers=16 \
    --audio_column_name=audio_path --text_column_name=text_path \
    --eval_accumulation_steps=50 \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
    --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./baselines/eval/scotty \
    --lang="cs-eng"

# # Wav2Vec 2.0 Chinese
# CUDA_VISIBLE_DEVICES=1 python eval.py \
#     --model_name_or_path=save/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn/checkpoint-32895 \
#     --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#     --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#     --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#     --cache_dir_name cache/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn \
#     --preprocessing_num_workers=16 \
#     --audio_column_name=audio_path --text_column_name=text_path \
#     --eval_accumulation_steps=10 \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./eval/zh \
#     --lang="cs-eng"

# # Wav2Vec 2.0 Multilingual
# CUDA_VISIBLE_DEVICES=1 python eval.py \
#     --model_name_or_path=save/facebook/wav2vec2-large-xlsr-53/checkpoint-38700 \
#     --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#     --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#     --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#     --cache_dir_name cache/facebook/wav2vec2-large-xlsr-53 \
#     --preprocessing_num_workers=16 \
#     --audio_column_name=audio_path --text_column_name=text_path \
#     --eval_accumulation_steps=10 \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./eval/multilingual \
#     --lang="yue,eng,cs-yue,cs-eng"

################# CANTONESE AND CODE-SWITCHING ONLY

# # Wav2Vec 2.0 Cantonese
# CUDA_VISIBLE_DEVICES=1 python eval.py \
#     --model_name_or_path=save_yue_cs-yue/ctl/wav2vec2-large-xlsr-cantonese/checkpoint-24384 \
#     --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#     --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#     --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#     --cache_dir_name cache_yue_cs-yue/ctl/wav2vec2-large-xlsr-cantonese \
#     --preprocessing_num_workers=16 \
#     --audio_column_name=audio_path --text_column_name=text_path \
#     --eval_accumulation_steps=10 \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./eval_yue_cs-yue/yue \
#     --lang="yue,cs-yue"

# # Wav2Vec 2.0 English
# CUDA_VISIBLE_DEVICES=0 python eval.py \
#     --model_name_or_path=save_yue_cs-yue/jonatasgrosman/wav2vec2-large-xlsr-53-english/checkpoint-39624 \
#     --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#     --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#     --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#     --cache_dir_name cache_yue_cs-yue/jonatasgrosman/wav2vec2-large-xlsr-53-english \
#     --preprocessing_num_workers=16 \
#     --audio_column_name=audio_path --text_column_name=text_path \
#     --eval_accumulation_steps=10 \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./eval_yue_cs-yue/eng \
#     --lang="yue,cs-yue"

# # Wav2Vec 2.0 Multilingual
# CUDA_VISIBLE_DEVICES=0 python eval.py \
#     --model_name_or_path=save_yue_cs-yue/facebook/wav2vec2-large-xlsr-53/checkpoint-31496 \
#     --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#     --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#     --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#     --cache_dir_name cache_yue_cs-yue/facebook/wav2vec2-large-xlsr-53 \
#     --preprocessing_num_workers=16 \
#     --audio_column_name=audio_path --text_column_name=text_path \
#     --eval_accumulation_steps=10 \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./eval_yue_cs-yue/multilingual \
#     --lang="yue,cs-yue"

# # Wav2Vec 2.0 Multilingual
# CUDA_VISIBLE_DEVICES=0 python eval.py \
#     --model_name_or_path=save_yue_cs-yue/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn/checkpoint-49784 \
#     --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#     --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#     --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#     --cache_dir_name cache_yue_cs-yue/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn \
#     --preprocessing_num_workers=16 \
#     --audio_column_name=audio_path --text_column_name=text_path \
#     --eval_accumulation_steps=10 \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./eval_yue_cs-yue/zh \
#     --lang="yue,cs-yue"

################# CANTONESE CODE-SWITCHING AND ENGLISH CODE-SWITCHING ONLY

# # Wav2Vec 2.0 Cantonese
# CUDA_VISIBLE_DEVICES=0 python eval.py \
#     --model_name_or_path=save_cs-yue_cs-eng/ctl/wav2vec2-large-xlsr-cantonese/checkpoint-10790 \
#     --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#     --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#     --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#     --cache_dir_name cache_cs-yue_cs-eng/ctl/wav2vec2-large-xlsr-cantonese/checkpoint-10790 \
#     --preprocessing_num_workers=16 \
#     --audio_column_name=audio_path --text_column_name=text_path \
#     --eval_accumulation_steps=10 \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./eval_cs-yue_cs-eng/yue \
#     --lang="cs-eng"

# # Wav2Vec 2.0 Cantonese
# CUDA_VISIBLE_DEVICES=7 python eval.py \
#     --model_name_or_path=baselines/save_cs-yue_cs-eng/scottykwok/wav2vec2-large-xlsr-cantonese/checkpoint-22383 \
#     --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#     --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#     --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#     --cache_dir_name baselines/cache_cs-yue_cs-eng/scottykwok/wav2vec2-large-xlsr-cantonese/checkpoint-22383 \
#     --preprocessing_num_workers=16 \
#     --audio_column_name=audio_path --text_column_name=text_path \
#     --eval_accumulation_steps=30 \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./baselines/eval_cs-yue_cs-eng/scotty \
#     --lang="cs-eng"

# # Wav2Vec 2.0 English
# CUDA_VISIBLE_DEVICES=1 python eval.py \
#     --model_name_or_path=save_cs-yue_cs-eng/jonatasgrosman/wav2vec2-large-xlsr-53-english/checkpoint-15770 \
#     --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#     --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#     --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#     --cache_dir_name cache_cs-yue_cs-eng/jonatasgrosman/wav2vec2-large-xlsr-53-english \
#     --preprocessing_num_workers=16 \
#     --audio_column_name=audio_path --text_column_name=text_path \
#     --eval_accumulation_steps=10 \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./eval_cs-yue_cs-eng/eng \
#     --lang="cs-eng"

# # Wav2Vec 2.0 Multilingual
# CUDA_VISIBLE_DEVICES=1 python eval.py \
#     --model_name_or_path=save_cs-yue_cs-eng/facebook/wav2vec2-large-xlsr-53/checkpoint-20750 \
#     --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#     --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#     --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#     --cache_dir_name cache_cs-yue_cs-eng/facebook/wav2vec2-large-xlsr-53 \
#     --preprocessing_num_workers=16 \
#     --audio_column_name=audio_path --text_column_name=text_path \
#     --eval_accumulation_steps=10 \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./eval_cs-yue_cs-eng/multilingual \
#     --lang="cs-eng"

# # Wav2Vec 2.0 Chinese
# CUDA_VISIBLE_DEVICES=1 python eval.py \
#     --model_name_or_path=save_cs-yue_cs-eng/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn/checkpoint-18260 \
#     --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#     --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#     --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#     --cache_dir_name cache_cs-yue_cs-eng/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn \
#     --preprocessing_num_workers=16 \
#     --audio_column_name=audio_path --text_column_name=text_path \
#     --eval_accumulation_steps=10 \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./eval_cs-yue_cs-eng/zh \
#     --lang="cs-eng"

############# ZERO-SHOT

# # Wav2Vec 2.0 Multilingual
# CUDA_VISIBLE_DEVICES=0 python eval.py \
#     --model_name_or_path=jonatasgrosman/wav2vec2-large-xlsr-53-english \
#     --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#     --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#     --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#     --cache_dir_name cache/jonatasgrosman/wav2vec2-large-xlsr-53-english \
#     --preprocessing_num_workers=16 \
#     --audio_column_name=audio_path --text_column_name=text_path \
#     --eval_accumulation_steps=10 \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./eval/zero-shot/eng \
#     --lang="cs-eng"

# # Wav2Vec 2.0 Multilingual
# CUDA_VISIBLE_DEVICES=0 python eval.py \
#     --model_name_or_path=CAiRE/wav2vec2-large-xlsr-53-cantonese \
#     --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#     --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#     --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#     --cache_dir_name cache_cs-yue_cs-eng/zero-shot/CAiRE/wav2vec2-large-xlsr-53-cantonese \
#     --preprocessing_num_workers=16 \
#     --audio_column_name=audio_path --text_column_name=text_path \
#     --eval_accumulation_steps=10 \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./eval/zero-shot/yue \
#     --lang="cs-eng"

# # Wav2Vec 2.0 Multilingual
# CUDA_VISIBLE_DEVICES=0 python eval.py \
#     --model_name_or_path=jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn \
#     --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#     --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#     --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#     --cache_dir_name cache_cs-yue_cs-eng/zero-shot/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn \
#     --preprocessing_num_workers=16 \
#     --audio_column_name=audio_path --text_column_name=text_path \
#     --eval_accumulation_steps=10 \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./eval/zero-shot/zh \
#     --lang="cs-eng"