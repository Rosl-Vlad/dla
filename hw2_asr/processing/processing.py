import re
import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


LETTERS = "qwertyuiopasdfghjklzxcvbnm "
END_TOKEN = 27


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def processing_LJ(path):
    id_latter = {}
    latter_id = {}
    for i, char in enumerate(LETTERS):
        id_latter[i] = char
        latter_id[char] = i

    id_chars = []
    values = {"id": [], "text": []}
    with open(path + "metadata.csv", "r") as rdr:
        for line in rdr:
            line = line.split("|")
            sub_text = re.sub(r'[^a-z ]+', '', line[2].lower()[:-1])
            values["text"].append(sub_text)
            values["id"].append(line[0])
            id_chars.append([latter_id[char] for char in sub_text])

    pd_values = pd.DataFrame(values)
    pd_values["id_chars"] = id_chars

    return pd_values, id_latter, latter_id


def filter_df(df, quantile=0.05):
    lens = []
    for i in df["text"]:
        lens.append(len(i))
    l_ = np.quantile(lens, quantile)
    r_ = np.quantile(lens, 1 - quantile)
    mask = (df["text"].str.len() > l_) & (df["text"].str.len() < r_)
    df = df[mask]

    return df, int(r_)


class LoadDataset(Dataset):
    def __init__(self, data, path, transform, padding_mel, padding_text):
        super().__init__()
        self.data = data.values
        self.path = path + "wavs"
        self.transform = transform
        self.padding_mel = padding_mel
        self.padding_text = padding_text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        audio_name, _, tokens = self.data[index]
        audio_path = os.path.join(self.path, audio_name + ".wav")
        wav, sr = torchaudio.load(audio_path)
        wav = wav.squeeze()
        mel_spectrogram = self.transform.forward(wav)
        log_mel = torch.log(mel_spectrogram + 1e-9)

        if log_mel.shape[1] < self.padding_mel:
            res = torch.cat((log_mel, torch.zeros((log_mel.shape[0], self.padding_mel - log_mel.shape[1]))),
                            dim=1)
        else:
            res = log_mel[:, :self.padding_mel]

        target = torch.tensor(tokens + [END_TOKEN] * (self.padding_text - len(tokens)))
        target_len = torch.tensor(len(tokens))
        input_len = torch.tensor(435)

        return res.squeeze(0).transpose(0, 1), target, target_len, input_len


def _get_loader(df, config, train: bool):
    melspec = torchaudio.transforms.MelSpectrogram(sample_rate=config["sample_rate"],
                                                   n_mels=config["n_mels"],
                                                   n_fft=config["n_fft"],
                                                   hop_length=config["hop_length"],
                                                   f_max=config["f_max"])
    if train:
        train_audio_transforms = nn.Sequential(
            melspec,
            torchaudio.transforms.FrequencyMasking(freq_mask_param=config["freq_mask_param"]),
            torchaudio.transforms.TimeMasking(time_mask_param=config["time_mask_param"])
        )
        return DataLoader(dataset=LoadDataset(df, config["path"], train_audio_transforms,
                                              config["padding_spec"], config["padding_text"]),
                          batch_size=config["batch_size"], shuffle=True, num_workers=5)

    return DataLoader(dataset=LoadDataset(df, config["path"], melspec, config["padding_spec"], config["padding_text"]),
                      batch_size=config["batch_size"], shuffle=False, num_workers=3)


def get_loader(config):
    df, id_latter, _ = processing_LJ(config["path"])
    df, _ = filter_df(df)
    train_df, valid_df = train_test_split(df, test_size=config["train_test_split"])

    train_loader = _get_loader(train_df, config, True)
    valid_loader = _get_loader(valid_df, config, False)
    return train_loader, valid_loader, id_latter
