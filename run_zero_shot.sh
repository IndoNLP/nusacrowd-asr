# # Wav2Vec 2.0 Multilingual
# CUDA_VISIBLE_DEVICES=0 python zero_shot.py \
#     --model_name_or_path=CAiRE/wav2vec2-large-xlsr-53-cantonese \
#     --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#     --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#     --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#     --cache_dir_name ./baselines/cache/CAiRE/wav2vec2-large-xlsr-53-cantonese \
#     --preprocessing_num_workers=16 \
#     --audio_column_name=audio_path --text_column_name=text_path \
#     --eval_accumulation_steps=10 \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./baselines/eval/zero-shot/yue \
#     --lang="cs-eng,cs-yue,eng,yue"

CUDA_VISIBLE_DEVICES=7 python zero_shot.py \
    --model_name_or_path=scottykwok/wav2vec2-large-xlsr-cantonese \
    --train_manifest_path=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
    --valid_manifest_path=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
    --test_manifest_path=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
    --cache_dir_name ./baselines/cache/scottykwok/wav2vec2-large-xlsr-cantonese \
    --preprocessing_num_workers=16 \
    --audio_column_name=audio_path --text_column_name=text_path \
    --eval_accumulation_steps=30 \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
    --seed=14045 --num_train_epochs=5 --learning_rate=5e-5 --output_dir=./baselines/eval/zero-shot/scotty \
    --lang="cs-eng,cs-yue,eng,yue"

# CUDA_VISIBLE_DEVICES=4 python harmonize_train.py --model=CAiRE/wav2vec2-large-xlsr-53-cantonese \
#    --name="Wav2Vec2" \
#    --train-manifest-list=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#    --valid-manifest-list=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#    --test-manifest-list=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#    --num-workers 4 --logging-strategy steps --logging-steps 10 --report-to "tensorboard" --lr 1e-20 --meta-lr 1e-20 \
#    --k-train=4 --k-valid=4 --epochs 50  --save-every 10 --save-total-limit 3 --evaluate-every 5 \
#    --lang="cs-yue,cs-eng,yue,eng" --loss="ctc" \
#    --cache-dir="cache_z/CAiRE/wav2vec2-large-xlsr-53-cantonese" --dropout 0.1 --clip  \
#    --output-dir="save_z" \
#    --cuda --verbose --copy-grad


# CUDA_VISIBLE_DEVICES=6 python harmonize_train.py --model=CAiRE/wav2vec2-large-xlsr-53-cantonese \
#    --name="Wav2Vec2" \
#    --train-manifest-list=data/ubc_cantonese_english_asr/preprocessed_train_metadata.csv \
#    --valid-manifest-list=data/ubc_cantonese_english_asr/preprocessed_valid_metadata.csv \
#    --test-manifest-list=data/ubc_cantonese_english_asr/preprocessed_test_metadata.csv \
#    --num-workers 4 --logging-strategy steps --logging-steps 10 --report-to "tensorboard" --lr 1e-20 --meta-lr 1e-20 \
#    --k-train=4 --k-valid=4 --epochs 50  --save-every 10 --save-total-limit 3 --evaluate-every 5 \
#    --lang="cs-yue,cs-eng,yue,eng" --loss="ctc" \
#    --cache-dir="cache_z/CAiRE/wav2vec2-large-xlsr-53-cantonese" --dropout 0.1 --clip  \
#    --output-dir="save_z" \
#    --cuda --verbose --copy-grad --pcgrad