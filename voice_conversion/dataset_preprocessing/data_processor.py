import os
import tarfile
from dataclasses import dataclass

from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils.data import Dataset, DataLoader

import torch
from torch import nn

import torchaudio
import librosa


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):

    def __init__(self, config: MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """

        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel


class LoadDataset(Dataset):
    def __init__(self, data_male, data_female, transform, padding_mel_size=96):
        super().__init__()
        self.data1 = data_male
        self.data2 = data_female
        self.transform = transform
        self.padding_mel_size = padding_mel_size

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        audio_name1 = self.data1[index]
        audio_name2 = self.data2[index % len(self.data2)]

        wav1, _ = librosa.load(audio_name1, 22050)
        wav1 = torch.tensor(wav1).squeeze()

        wav2, _ = librosa.load(audio_name2, 22050)
        wav2 = torch.tensor(wav2).squeeze()

        mel_spectrogram1 = self.transform.forward(wav1)
        mel_spectrogram2 = self.transform.forward(wav2)

        return mel_spectrogram1, mel_spectrogram2


def download_data_set():
    # LJ speech
    os.system("wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2")
    tar = tarfile.open("LJSpeech-1.1.tar.bz2", "r:bz2")
    tar.extractall()
    tar.close()

    # The World English Bible
    os.system("kaggle datasets download -d bryanpark/the-world-english-bible-speech-dataset")
    os.system("mkdir maleSpeech")
    os.system("mv the-world-english-bible-speech-dataset.zip maleSpeech/")
    os.system("unzip maleSpeech/the-world-english-bible-speech-dataset.zip")

    # create dir for result dataset
    os.system("mkdir specData")


def download_vovoder():
    os.system("git clone https://github.com/NVIDIA/waveglow.git")

    gdd.download_file_from_google_drive(
        file_id='1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF',
        dest_path='./waveglow_256channels_universal_v5.pt'
    )


if __name__ == "__main__":
    featurizer = MelSpectrogram(MelSpectrogramConfig())

    lj_speech = "LJSpeech-1.1/wavs"
    male_speech = "maleSpeech"

    f = open(os.path.join(male_speech, "transcript.txt"), )
    transcript_male_speech = [os.path.join(male_speech, x.split("\t")[0] + ".wav") for x in f.read().split("\n")
                              if x.split("/")[0] != "Isaiah" and float(x.split("\t")[2]) > 2]
    f.close()

    for _, _, files in os.walk(lj_speech):
        transcript_female_speech_ = files

    transcript_female_speech = [os.path.join(lj_speech, file) for file in transcript_female_speech_]

    train_loader = DataLoader(dataset=LoadDataset(transcript_male_speech, transcript_female_speech, featurizer),
                              batch_size=1, shuffle=False, num_workers=20)

    path = "/specData"

    for i, (male, female) in enumerate(train_loader):
        torch.save(male, path + "male/" + "file_" + str(i) + ".pt")
        torch.save(female, path + "female/" + "file_" + str(i) + ".pt")
