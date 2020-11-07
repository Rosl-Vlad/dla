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
