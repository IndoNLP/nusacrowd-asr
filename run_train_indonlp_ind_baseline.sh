#######
## Task Specific
#######
CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name indspeech_digit_cdsr_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/indspeech_digit_cdsr/ \
   --output_dir=./save/indspeech_digit_cdsr/ &
    
CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name indspeech_news_lvcsr_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/indspeech_news_lvcsr/ \
   --output_dir=./save/indspeech_news_lvcsr/
    

CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name indspeech_teldialog_lvcsr_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/indspeech_teldialog_lvcsr/ \
   --output_dir=./save/indspeech_teldialog_lvcsr/
    
    
CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name indspeech_teldialog_svcsr_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/indspeech_teldialog_svcsr/ \
   --output_dir=./save/indspeech_teldialog_svcsr/
    

CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name librivox_indonesia_ind_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/librivox_indonesia_ind/ \
   --output_dir=./save/librivox_indonesia_ind/
    

CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name indspeech_newstra_ethnicsr_nooverlap_sun_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/indspeech_newstra_ethnicsr_nooverlap_sun/ \
   --output_dir=./save/indspeech_newstra_ethnicsr_nooverlap_sun/
    

CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name indspeech_news_ethnicsr_su_nooverlap_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/indspeech_news_ethnicsr_su_nooverlap/ \
   --output_dir=./save/indspeech_news_ethnicsr_su_nooverlap/
    

CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name librivox_indonesia_sun_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/librivox_indonesia_sun/ \
   --output_dir=./save/librivox_indonesia_sun/
    

CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name su_id_asr_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/su_id_asr/ \
   --output_dir=./save/su_id_asr/
    

CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name indspeech_newstra_ethnicsr_nooverlap_jav_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/indspeech_newstra_ethnicsr_nooverlap_jav/ \
   --output_dir=./save/indspeech_newstra_ethnicsr_nooverlap_jav/
    

CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name indspeech_news_ethnicsr_jv_nooverlap_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/indspeech_news_ethnicsr_jv_nooverlap/ \
   --output_dir=./save/indspeech_news_ethnicsr_jv_nooverlap/
    

CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name librivox_indonesia_jav_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/librivox_indonesia_jav/ \
   --output_dir=./save/librivox_indonesia_jav/
    

CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name jv_id_asr_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/jv_id_asr/ \
   --output_dir=./save/jv_id_asr/
    

CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name indspeech_newstra_ethnicsr_nooverlap_ban_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/indspeech_newstra_ethnicsr_nooverlap_ban/ \
   --output_dir=./save/indspeech_newstra_ethnicsr_nooverlap_ban/
    

CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name librivox_indonesia_ban_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/librivox_indonesia_ban/ \
   --output_dir=./save/librivox_indonesia_ban/
    

CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name indspeech_newstra_ethnicsr_nooverlap_btk_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/indspeech_newstra_ethnicsr_nooverlap_btk/ \
   --output_dir=./save/indspeech_newstra_ethnicsr_nooverlap_btk/
    

CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name librivox_indonesia_ace_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/librivox_indonesia_ace/ \
   --output_dir=./save/librivox_indonesia_ace/
    

CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name librivox_indonesia_bug_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/librivox_indonesia_bug/ \
   --output_dir=./save/librivox_indonesia_bug/
    

CUDA_VISIBLE_DEVICES=0 python src/train.py --model_name_or_path=indonesian-nlp/wav2vec2-large-xlsr-indonesian-baseline \
   --task_config_name librivox_indonesia_min_nusantara_sptext \
   --preprocessing_num_workers=16 --audio_column_name=audio --text_column_name=text \
   --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
   --dataloader_num_workers=16 --dataloader_pin_memory \
   --seed=14045 --num_train_epochs=30 --learning_rate=5e-5 --fp16 \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --metric_for_best_model=mer --greater_is_better=False \
   --gradient_checkpointing=True \
   --cache_dir=./cache/librivox_indonesia_min/ \
   --output_dir=./save/librivox_indonesia_min/