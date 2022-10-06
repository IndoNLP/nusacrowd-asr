import os, sys
from nusacrowd import NusantaraConfigHelper

SPEECH_RECOGNITION_TASKS = [
	'indspeech_news_ethnicsr_jv_nooverlap_nusantara_sptext',
	'indspeech_news_ethnicsr_su_nooverlap_nusantara_sptext',
	'indspeech_newstra_ethnicsr_nooverlap_ban_nusantara_sptext',
	'indspeech_newstra_ethnicsr_nooverlap_btk_nusantara_sptext',
	'indspeech_newstra_ethnicsr_nooverlap_jav_nusantara_sptext',
	'indspeech_newstra_ethnicsr_nooverlap_sun_nusantara_sptext',
	'indspeech_digit_cdsr_nusantara_sptext',
	'indspeech_news_lvcsr_nusantara_sptext',
	'indspeech_teldialog_lvcsr_nusantara_sptext',
	'indspeech_teldialog_svcsr_nusantara_sptext',
	'indspeech_teldialog_svcsr_nusantara_sptext',
	'librivox_indonesia_nusantara_sptext',
	'librivox_indonesia_ace_nusantara_sptext',
	'librivox_indonesia_bug_nusantara_sptext',
	'librivox_indonesia_sun_nusantara_sptext',
	'librivox_indonesia_ban_nusantara_sptext',
	'librivox_indonesia_min_nusantara_sptext',
	'librivox_indonesia_ind_nusantara_sptext',
	'librivox_indonesia_jav_nusantara_sptext',
	'jv_id_asr_nusantara_sptext',
	'su_id_asr_nusantara_sptext',
	'titml_idn_nusantara_sptext'
]

TTS_TASKS = [
	'indspeech_news_tts_nusantara_sptext',
	'jv_id_tts_nusantara_sptext',
	'su_id_tts_nusantara_sptext',   
]

SLU_TASKS = [
    'xsid_nusantara_text', 
    'xsid_nusantara_seq_label'
]

SPEECH_TRANSLATION_TASKS = [
    'covost2_ind_eng_nusantara_t2t',
    'covost2_eng_ind_nusantara_t2t',
]

SPEECH_TO_SPEECH_TASKS = [
    'cvss_c_nusantara_s2s',
    'cvss_t_nusantara_s2s'
]

def load_asr_tasks():
    conhelps = NusantaraConfigHelper()
    asr_datasets = {
        helper.config.name: helper.load_dataset() for helper in conhelps.filtered(lambda x: x.config.name in SPEECH_RECOGNITION_TASKS)
    }
    return asr_datasets

def load_tts_tasks():
    conhelps = NusantaraConfigHelper()
    tts_datasets = {
        helper.config.name: helper.load_dataset() for helper in conhelps.filtered(lambda x: x.config.name in TTS_TASKS)
    }
    return tts_datasets

def load_slu_tasks():
    conhelps = NusantaraConfigHelper()
    slu_datasets = {
        helper.config.name: helper.load_dataset() for helper in conhelps.filtered(lambda x: x.config.name in SLU_TASKS)
    }
    return slu_datasets

def load_s2t_tasks():
    conhelps = NusantaraConfigHelper()
    s2t_datasets = {
        helper.config.name: helper.load_dataset() for helper in conhelps.filtered(lambda x: x.config.name in SPEECH_TRANSLATION_TASKS)
    }
    return s2t_datasets

def load_s2s_tasks():
    conhelps = NusantaraConfigHelper()
    s2s_datasets = {
        helper.config.name: helper.load_dataset() for helper in conhelps.filtered(lambda x: x.config.name in SPEECH_TO_SPEECH_TASKS)
    }
    return s2s_datasets

if __name__ == '__main__':
    print('Load ASR Datasets...')
    asr_datasets = asr_datasets()
    
    print(f'Loaded {len(nlu_datasets)} ASR datasets')
    for i, dset_subset in enumerate(asr_datasets.keys()):
        print(f'{i} {dset_subset}')
    
    print('Load TTS Datasets...')
    tts_datasets = load_tts_tasks()
    
    print(f'Loaded {len(nlg_datasets)} TTS datasets')
    for i, dset_subset in enumerate(tts_datasets.keys()):
        print(f'{i} {dset_subset}')
