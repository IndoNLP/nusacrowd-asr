## MULTILINGUAL MULTI-TASK TRAINING

# all_lang
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 python src/train_seq2seq.py \
   --model_name_or_path=openai/whisper-base \
   --lang all \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=1e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=1 --load_best_model_at_end \
   --metric_for_best_model=wer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/all_lang/ \
   --output_dir=./save/all_lang/ 2>&1 | tee ./log/all_lang_exp.logs