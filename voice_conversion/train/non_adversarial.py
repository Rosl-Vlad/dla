import torch
from torch import nn

from torch.utils.data import DataLoader
from model.models import Encoder, Decoder
from .data_loader import LoadDataset, get_transcript_speech


# TODO: перевести все на конфиг
def init(config):
    encoder = Encoder(80, [160, 320, 640, 1280, 64], [64, 32, 16, 8, 4])
    decoder_m = Decoder(64, [1280, 640, 320, 160, 80], [4, 8, 16, 32, 64])
    decoder_f = Decoder(64, [1280, 640, 320, 160, 80], [4, 8, 16, 32, 64])

    decoder_m = decoder_m.to(config["device"])
    decoder_f = decoder_f.to(config["device"])
    encoder = encoder.to(config["device"])

    return encoder, decoder_m, decoder_f


def train(config):
    encoder, decoder_m, decoder_f = init(config)

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    optim = torch.optim.Adam(list(decoder_m.parameters()) + list(encoder.parameters()) + list(decoder_f.parameters()),
                             lr=config["lr"])

    transcript_male_speech, transcript_female_speech = get_transcript_speech(config)
    train_loader = DataLoader(dataset=LoadDataset(transcript_male_speech, transcript_female_speech),
                              batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

    for i in range(config["epoch"]):
        decoder_f.train()
        decoder_m.train()
        encoder.train()

        for male, female in train_loader:
            male = male.to(config["device"])
            female = female.to(config["device"])

            optim.zero_grad()

            preds_male = decoder_m(encoder(male))
            preds_female = decoder_f(encoder(female))

            loss_male = (criterion_mse(male, preds_male) + criterion_l1(male, preds_male)) / 2
            loss_female = (criterion_mse(female, preds_female) + criterion_l1(female, preds_female)) / 2
            loss = (loss_female + loss_male) / 2
            loss.backward()
            optim.step()

    return encoder, decoder_m, decoder_f
