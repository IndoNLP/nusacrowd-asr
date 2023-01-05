## MONOLINGUAL MULTI-TASK TRAINING

# # ind_lang
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=6 python src/train_seq2seq.py \
#    --model_name_or_path=openai/whisper-small \
#    --lang ind \
#    --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=16 --dataloader_pin_memory \
#    --seed=14045 --num_train_epochs=30 --learning_rate=1e-4 --fp16 \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=1 --load_best_model_at_end \
#    --metric_for_best_model=wer --greater_is_better=False \
#    --gradient_checkpointing=True \
#    --cache_dir=./cache/ind_lang/ \
#    --output_dir=./save/ind_lang/ 2>&1 | tee ./log/ind_lang_exp.logs

# sun_lang
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python src/train_seq2seq.py \
   --model_name_or_path=openai/whisper-small \
   --lang sun \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=1e-4 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=1 --load_best_model_at_end \
   --metric_for_best_model=wer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/sun_lang/ \
   --output_dir=./save/sun_lang/ 2>&1 | tee ./log/sun_lang_exp.logs

# # jav_lang
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=6 python src/train_seq2seq.py \
#    --model_name_or_path=openai/whisper-small \
#    --lang jav \
#    --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=16 --dataloader_pin_memory \
#    --seed=14045 --num_train_epochs=30 --learning_rate=1e-4 --fp16 \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=1 --load_best_model_at_end \
#    --metric_for_best_model=wer --greater_is_better=False \
#    --gradient_checkpointing=True \
#    --cache_dir=./cache/jav_lang/ \
#    --output_dir=./save/jav_lang/ 2>&1 | tee ./log/jav_lang_exp.logs