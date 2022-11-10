################# ALL DATA

#######
## Multilingual
#######
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python src/train.py \
   --model_name_or_path=facebook/wav2vec2-large-xlsr-53 --lang all \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/all_lang/ \
   --output_dir=./save/all_lang/ 2>&1 | tee multilingual_exp.logs &

#######
## Monolingual Indonesian
#######
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=1 python src/train.py \
   --model_name_or_path=facebook/wav2vec2-large-xlsr-53 --lang ind \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/ind_lang/ \
   --output_dir=./save/ind_lang/ 2>&1 | tee indonesian_exp.logs &
   
#######
## Monolingual Sundanese
#######
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=2 python src/train.py \
   --model_name_or_path=facebook/wav2vec2-large-xlsr-53 --lang sun \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/sun_lang/ \
   --output_dir=./save/sun_lang/ 2>&1 | tee sundanese_exp.logs &
   
#######
## Monolingual Javanese
#######
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=3 python src/train.py \
   --model_name_or_path=facebook/wav2vec2-large-xlsr-53 --lang jav \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/jav_lang/ \
   --output_dir=./save/jav_lang/ 2>&1 | tee javanese_exp.logs
