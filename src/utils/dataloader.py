import os, sys
from itertools import chain
from nusacrowd import NusantaraConfigHelper
from datasets import concatenate_datasets, load_dataset, DatasetDict

SPEECH_RECOGNITION_TASKS = {
    'ind': [
        'indspeech_digit_cdsr_nusantara_sptext',
        'indspeech_news_lvcsr_nusantara_sptext',
        'indspeech_teldialog_lvcsr_nusantara_sptext',
        'indspeech_teldialog_svcsr_nusantara_sptext',
        'librivox_indonesia_ind_nusantara_sptext',
        'titml_idn_nusantara_sptext'
    ], 
    'sun': [
        'indspeech_newstra_ethnicsr_nooverlap_sun_nusantara_sptext',
        'indspeech_news_ethnicsr_su_nooverlap_nusantara_sptext',
        'librivox_indonesia_sun_nusantara_sptext',
        'su_id_asr_nusantara_sptext',
    ],
    'jav': [
        'indspeech_newstra_ethnicsr_nooverlap_jav_nusantara_sptext',
        'indspeech_news_ethnicsr_jv_nooverlap_nusantara_sptext',
        'librivox_indonesia_jav_nusantara_sptext',
        'jv_id_asr_nusantara_sptext',
    ],
    'ban': [
        'indspeech_newstra_ethnicsr_nooverlap_ban_nusantara_sptext',
        'librivox_indonesia_ban_nusantara_sptext',
    ],
    'btk': [
        'indspeech_newstra_ethnicsr_nooverlap_btk_nusantara_sptext',
    ],
    'ace': [
        'librivox_indonesia_ace_nusantara_sptext',
    ],
    'bug': [
        'librivox_indonesia_bug_nusantara_sptext',
    ], 
    'min': [
        'librivox_indonesia_min_nusantara_sptext',
    ]
}

TTS_TASKS = {
    'ind': [
        'indspeech_news_tts_nusantara_sptext'
    ],
    'jav': [
        'jv_id_tts_nusantara_sptext'
    ],
    'sun': [
        'su_id_tts_nusantara_sptext'
    ]
}

SLU_TASKS = [
    'xsid_nusantara_text', 
    'xsid_nusantara_seq_label'
]

SPEECH_TRANSLATION_TASKS = [
    'covost2_ind_eng_nusantara_s2t',
    'covost2_eng_ind_nusantara_s2t',
]

SPEECH_TO_SPEECH_TASKS = [
    'cvss_c_nusantara_s2s',
    'cvss_t_nusantara_s2s'
]

# Global Config Helper
conhelps = NusantaraConfigHelper()

# Basic Loader
def load_asr_tasks(langs=['ind']):
    tasks = []
    for lang in langs:
        tasks += SPEECH_RECOGNITION_TASKS[lang] if lang in SPEECH_RECOGNITION_TASKS else []
        
    asr_datasets = {
        helper.config.name: helper.load_dataset() for helper in conhelps.filtered(lambda x: x.config.name in tasks)
    }
    return asr_datasets

def load_tts_tasks(lang=['ind']):
    tasks = []
    for lang in langs:
        tasks += TTS_TASKS[lang] if lang in TTS_TASKS else []
        
    tts_datasets = {
        helper.config.name: helper.load_dataset() for helper in conhelps.filtered(lambda x: x.config.name in tasks)
    }
    return tts_datasets

def load_slu_tasks():
    slu_datasets = {
        helper.config.name: helper.load_dataset() for helper in conhelps.filtered(lambda x: x.config.name in SLU_TASKS)
    }
    return slu_datasets

def load_s2t_tasks():
    s2t_datasets = {
        helper.config.name: helper.load_dataset() for helper in conhelps.filtered(lambda x: x.config.name in SPEECH_TRANSLATION_TASKS)
    }
    return s2t_datasets

def load_s2s_tasks():
    s2s_datasets = {
        helper.config.name: helper.load_dataset() for helper in conhelps.filtered(lambda x: x.config.name in SPEECH_TO_SPEECH_TASKS)
    }
    return s2s_datasets

# Experiment Data Loader
# Input:
#    asr (bool): whether to load from ASR datasets or not
#    tts (bool): whether to load from TTS datasets or not
#    train_lang (str or list[str]): list of languages to load for training & validation, 'all' will load all languages
# Output:
#    train_dataset (Dataset): training dataset, instance of HuggingFace datasets.Dataset
#    train_dataset (Dataset): training dataset, instance of HuggingFace datasets.Dataset
#    test_datasets (dict<str: Dataset>): a map from config_name to 
def load_speech_datasets(asr=True, tts=True, train_lang='ind'):
    all_langs = list(set(list(SPEECH_RECOGNITION_TASKS.keys()) + list(TTS_TASKS.keys())))
    if type(train_lang) is str:
        train_lang = [train_lang]
    if train_lang == 'all':
        train_lang = all_langs
        
    train_configs = list(chain(*(
                        [[cfg for cfg in SPEECH_RECOGNITION_TASKS[lang]] for lang in train_lang if lang in SPEECH_RECOGNITION_TASKS] + 
                        [[cfg for cfg in TTS_TASKS[lang]] for lang in train_lang if lang in TTS_TASKS]
                    )))

    datasets = {}
    if asr:
        datasets.update(load_asr_tasks(langs=all_langs))
    if tts:
        datasets.update(load_tts_tasks(langs=all_langs))
        
    train_datasets = []
    test_dataset_dict = {}
    for config_name, dataset in datasets.items():
        if 'train' in dataset and config_name in train_configs:
            train_datasets.append(dataset['train'])
        if 'test' in dataset:
            test_dataset_dict[config_name] = dataset['test']
        
    train_dataset = concatenate_datasets([train_datasets])
    
    tr_val_dataset = train_dataset.train_test_split(test_size=100 * len(datasets), seed=20221010)
    train_dataset, valid_dataset = tr_val_dataset['train'], tr_val_dataset['test']
    
    return train_dataset, valid_dataset, test_dataset_dict    

if __name__ == '__main__':
    print('Load Buginese Speech Datasets...')
    asr_datasets = load_speech_datasets(asr=True, tts=True, train_lang='bug')
    
    print('Load All Speech Datasets...')
    asr_datasets = load_speech_datasets(asr=True, tts=True, train_lang='bug')