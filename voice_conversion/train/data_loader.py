import os
import torch
import numpy as np

from torch.utils.data import Dataset


class LoadDataset(Dataset):
    def __init__(self, data_male, data_female, padding_mel_size=96):
        super().__init__()
        self.data1 = data_male
        self.data2 = data_female
        self.padding_mel_size = padding_mel_size

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        audio_name1 = self.data1[index]
        audio_name2 = self.data2[index % len(self.data2)]

        mel_spectrogram1 = torch.load(audio_name1)[0]
        mel_spectrogram2 = torch.load(audio_name2)[0]

        if self.padding_mel_size is not None:
            mel_spectrogram1 = self.padding(mel_spectrogram1)
            mel_spectrogram2 = self.padding(mel_spectrogram2)

        return mel_spectrogram1, mel_spectrogram2

    def padding(self, mel):
        padding = np.random.randint(mel.shape[1] - self.padding_mel_size + 1)
        res = mel[:, padding:self.padding_mel_size + padding]

        return res


def get_transcript_speech(config):
    for _, _, files in os.walk(os.path.join(config["path"], "male")):
        transcript_male_speech_ = files

    transcript_male_speech = [os.path.join(config["path"], "male", file) for file in transcript_male_speech_]

    for _, _, files in os.walk(os.path.join(config["path"], "female")):
        transcript_female_speech_ = files

    transcript_female_speech = [os.path.join(config["path"], "female", file) for file in transcript_female_speech_]

    return transcript_male_speech, transcript_female_speech
