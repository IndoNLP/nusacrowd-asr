# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python src/train_seq2seq.py \
#    --model_name_or_path=openai/whisper-base \
#    --task_config_name indspeech_digit_cdsr_nusantara_sptext \
#    --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=16 --dataloader_pin_memory \
#    --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
#    --metric_for_best_model=wer --greater_is_better=False \
#    --gradient_checkpointing=True \
#    --cache_dir=./cache/indspeech_digit_cdsr/ \
#    --output_dir=./save/indspeech_digit_cdsr/ 2>&1 | tee indspeech_digit_cdsr_exp.logs


OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python src/train_seq2seq.py \
   --model_name_or_path=openai/whisper-base \
   --task_config_name librivox_indonesia_ind_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=wer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/librivox_indonesia_ind/ \
   --output_dir=./save/librivox_indonesia_ind/ 2>&1 | tee librivox_indonesia_ind_exp.logs