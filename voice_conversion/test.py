from IPython import display

import librosa

import torch
import torchaudio

from model.models import Encoder, Decoder
from model.vocoder import Vocoder
from dataset_preprocessing.data_processor import MelSpectrogram, MelSpectrogramConfig


def init(config):
    encoder = Encoder(80, [160, 320, 640, 1280, 1024], [64, 32, 16, 8, 4])
    decoder_m = Decoder(1024, [1280, 640, 320, 160, 80], [4, 8, 16, 32, 64])
    decoder_f = Decoder(1024, [1280, 640, 320, 160, 80], [4, 8, 16, 32, 64])

    decoder_m = decoder_m.to(config["device"])
    decoder_f = decoder_f.to(config["device"])
    encoder = encoder.to(config["device"])

    return encoder, decoder_m, decoder_f


def load_models(config):
    encoder, decoder_m, decoder_f = init(config)

    encoder.load_state_dict(torch.load("model_checkpoint/encoder.model.pth"), strict=False)
    decoder_m.load_state_dict(torch.load("model_checkpoint/decoder_male.model.pth"), strict=False)
    decoder_f.load_state_dict(torch.load("model_checkpoint/decoder_female.model.pth"), strict=False)

    return encoder, decoder_m, decoder_f


def load_vocoder():
    vocoder = Vocoder()
    vocoder = vocoder.eval()

    return vocoder


def test(config, wav_path):
    vocoder = load_vocoder()
    encoder, decoder_m, decoder_f = load_models(config)

    featurizer = MelSpectrogram(MelSpectrogramConfig())
    wav, _ = librosa.load(wav_path, 22050)
    wav = torch.tensor(wav).squeeze()

    wav = featurizer(wav)
    wav = wav.reshape(1, wav.shape[0], wav.shape[1])

    display.Audio(vocoder.inference(decoder_f(encoder(wav.to(config["device"])))[0].unsqueeze(0).cpu().detach()).squeeze(),
                  rate=22050)
