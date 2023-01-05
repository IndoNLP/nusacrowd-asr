## SINGLE-TASK TRAINING

# # indspeech_digit_cdsr
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python src/train_seq2seq.py \
#    --model_name_or_path=openai/whisper-base \
#    --task_config_name indspeech_digit_cdsr_nusantara_sptext \
#    --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=16 --dataloader_pin_memory \
#    --seed=14045 --num_train_epochs=30 --learning_rate=1e-5 --fp16 \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=1 --load_best_model_at_end \
#    --metric_for_best_model=wer --greater_is_better=False \
#    --gradient_checkpointing=True \
#    --cache_dir=./cache/indspeech_digit_cdsr/ \
#    --output_dir=./save/indspeech_digit_cdsr/ 2>&1 | tee ./log/indspeech_digit_cdsr_exp.logs

# # indspeech_news_lvcsr
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python src/train_seq2seq.py \
#    --model_name_or_path=openai/whisper-base \
#    --task_config_name indspeech_news_lvcsr_nusantara_sptext \
#    --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=16 --dataloader_pin_memory \
#    --seed=14045 --num_train_epochs=30 --learning_rate=1e-5 --fp16 \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=1 --load_best_model_at_end \
#    --metric_for_best_model=wer --greater_is_better=False \
#    --gradient_checkpointing=True \
#    --cache_dir=./cache/indspeech_news_lvcsr/ \
#    --output_dir=./save/indspeech_news_lvcsr/ 2>&1 | tee ./log/indspeech_news_lvcsr_exp.logs

# indspeech_teldialog_lvcsr
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python src/train_seq2seq.py \
   --model_name_or_path=openai/whisper-base \
   --task_config_name indspeech_teldialog_lvcsr_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=1e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=1 --load_best_model_at_end \
   --metric_for_best_model=wer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/indspeech_teldialog_lvcsr/ \
   --output_dir=./save/indspeech_teldialog_lvcsr/ 2>&1 | tee ./log/indspeech_teldialog_lvcsr_exp.logs

# # librivox_indonesia_ind
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python src/train_seq2seq.py \
#    --model_name_or_path=openai/whisper-base \
#    --task_config_name librivox_indonesia_ind_nusantara_sptext \
#    --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=16 --dataloader_pin_memory \
#    --seed=14045 --num_train_epochs=30 --learning_rate=1e-5 --fp16 \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=1 --load_best_model_at_end \
#    --metric_for_best_model=wer --greater_is_better=False \
#    --gradient_checkpointing=True \
#    --cache_dir=./cache/librivox_indonesia_ind/ \
#    --output_dir=./save/librivox_indonesia_ind/ 2>&1 | tee ./log/librivox_indonesia_ind_exp.logs

# # librivox_indonesia_ace
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python src/train_seq2seq.py \
#    --model_name_or_path=openai/whisper-base \
#    --task_config_name librivox_indonesia_ace_nusantara_sptext \
#    --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=16 --dataloader_pin_memory \
#    --seed=14045 --num_train_epochs=30 --learning_rate=1e-5 --fp16 \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=1 --load_best_model_at_end \
#    --metric_for_best_model=wer --greater_is_better=False \
#    --gradient_checkpointing=True \
#    --cache_dir=./cache/librivox_indonesia_ace/ \
#    --output_dir=./save/librivox_indonesia_ace/ 2>&1 | tee ./log/librivox_indonesia_ace_exp.logs