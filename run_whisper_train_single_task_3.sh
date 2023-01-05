## SINGLE-TASK TRAINING

# # librivox_indonesia_jav
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 python src/train_seq2seq.py \
#    --model_name_or_path=openai/whisper-base \
#    --task_config_name librivox_indonesia_jav_nusantara_sptext \
#    --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=16 --dataloader_pin_memory \
#    --seed=14045 --num_train_epochs=30 --learning_rate=1e-5 --fp16 \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=1 --load_best_model_at_end \
#    --metric_for_best_model=wer --greater_is_better=False \
#    --gradient_checkpointing=True \
#    --cache_dir=./cache/librivox_indonesia_jav/ \
#    --output_dir=./save/librivox_indonesia_jav/ 2>&1 | tee ./log/librivox_indonesia_jav_exp.logs

# # librivox_indonesia_min
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 python src/train_seq2seq.py \
#    --model_name_or_path=openai/whisper-base \
#    --task_config_name librivox_indonesia_min_nusantara_sptext \
#    --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=16 --dataloader_pin_memory \
#    --seed=14045 --num_train_epochs=30 --learning_rate=1e-5 --fp16 \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=1 --load_best_model_at_end \
#    --metric_for_best_model=wer --greater_is_better=False \
#    --gradient_checkpointing=True \
#    --cache_dir=./cache/librivox_indonesia_min/ \
#    --output_dir=./save/librivox_indonesia_min/ 2>&1 | tee ./log/librivox_indonesia_min_exp.logs

# # indspeech_news_ethnicsr_su
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 python src/train_seq2seq.py \
#    --model_name_or_path=openai/whisper-base \
#    --task_config_name indspeech_news_ethnicsr_su_nooverlap_nusantara_sptext \
#    --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=16 --dataloader_pin_memory \
#    --seed=14045 --num_train_epochs=30 --learning_rate=1e-5 --fp16 \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=1 --load_best_model_at_end \
#    --metric_for_best_model=wer --greater_is_better=False \
#    --gradient_checkpointing=True \
#    --cache_dir=./cache/indspeech_news_ethnicsr_su/ \
#    --output_dir=./save/indspeech_news_ethnicsr_su/ 2>&1 | tee ./log/indspeech_news_ethnicsr_su_exp.logs

# # indspeech_newstra_ethnicsr_sun
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 python src/train_seq2seq.py \
#    --model_name_or_path=openai/whisper-base \
#    --task_config_name indspeech_newstra_ethnicsr_nooverlap_sun_nusantara_sptext \
#    --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=16 --dataloader_pin_memory \
#    --seed=14045 --num_train_epochs=30 --learning_rate=1e-5 --fp16 \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=1 --load_best_model_at_end \
#    --metric_for_best_model=wer --greater_is_better=False \
#    --gradient_checkpointing=True \
#    --cache_dir=./cache/indspeech_newstra_ethnicsr_sun/ \
#    --output_dir=./save/indspeech_newstra_ethnicsr_sun/ 2>&1 | tee ./log/indspeech_newstra_ethnicsr_sun_exp.logs

# # librivox_indonesia_sun
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 python src/train_seq2seq.py \
#    --model_name_or_path=openai/whisper-base \
#    --task_config_name librivox_indonesia_sun_nusantara_sptext \
#    --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=16 --dataloader_pin_memory \
#    --seed=14045 --num_train_epochs=30 --learning_rate=1e-5 --fp16 \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=1 --load_best_model_at_end \
#    --metric_for_best_model=wer --greater_is_better=False \
#    --gradient_checkpointing=True \
#    --cache_dir=./cache/librivox_indonesia_sun/ \
#    --output_dir=./save/librivox_indonesia_sun/ 2>&1 | tee ./log/librivox_indonesia_sun_exp.logs