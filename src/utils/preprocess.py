from tqdm import trange

import numpy as np
import os
import pandas as pd


class Preprocess:

    def __init__(self, raw_metadata_path, preprocessed_metadata_path=None):

        print('Start preprocessing: {}.'.format(raw_metadata_path))

        self.raw_metadata_path = raw_metadata_path

        raw_p = os.path.normpath(self.raw_metadata_path).split(os.sep)
        self.dir_path = os.path.join(*raw_p[:-1])

        if preprocessed_metadata_path is None:
            file_split = "_".join(raw_p[-1].split("_")[-2:])
            self.preprocessed_metadata_path = os.path.join(self.dir_path, "preprocessed_{}".format(file_split))
        else:
            self.preprocessed_metadata_path = preprocessed_metadata_path

        if not os.path.isfile(self.preprocessed_metadata_path):
            self.reformat_csv()

        print('Finished preprocessing: {}.'.format(self.preprocessed_metadata_path))

    def reformat_csv(self):
        raw_df = pd.read_csv(self.raw_metadata_path)
        df = raw_df.copy()

        for i in trange(len(df)):
            transcription_relative_path = df["text_path"][i]
            text_path = os.path.join(self.dir_path, transcription_relative_path)
            with open(text_path, "r") as f:
                df.loc[i, ["text_path"]] = f.readline()

        df.replace('', np.nan, inplace=True)
        df = df.dropna()
        df.to_csv(self.preprocessed_metadata_path)


if __name__ == '__main__':
    full_preprocess = Preprocess("../data/ubc_cantonese_english_asr/cantonese_english_asr_metadata_full.csv")
    test_preprocess = Preprocess("../data/ubc_cantonese_english_asr/cantonese_english_asr_test_metadata.csv")
    train_preprocess = Preprocess("../data/ubc_cantonese_english_asr/cantonese_english_asr_train_metadata.csv")
    valid_preprocess = Preprocess("../data/ubc_cantonese_english_asr/cantonese_english_asr_valid_metadata.csv")