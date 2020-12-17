import os
import torch
from torch import nn

from torch.utils.data import DataLoader
from .model.models import Encoder, Decoder, Discriminator
from data_loader import LoadDataset, get_transcript_speech


# TODO: перевести все на конфиг
def init(config):
    encoder = Encoder(80, [160, 320, 640, 1280, 64], [64, 32, 16, 8, 4])
    decoder_m = Decoder(64, [1280, 640, 320, 160, 80], [4, 8, 16, 32, 64])
    decoder_f = Decoder(64, [1280, 640, 320, 160, 80], [4, 8, 16, 32, 64])

    decoder_m = decoder_m.to(config["device"])
    decoder_f = decoder_f.to(config["device"])
    encoder = encoder.to(config["device"])

    dis_f = Discriminator()
    dis_m = Discriminator()

    dis_f = dis_f.to(config["device"])
    dis_m = dis_m.to(config["device"])

    return encoder, decoder_m, decoder_f, dis_m, dis_f


def train(config):
    encoder, decoder_m, decoder_f, dis_m, dis_f = init(config)

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    optim_dis_f = torch.optim.Adam(dis_f.parameters(), lr=0.0002)
    optim_dis_m = torch.optim.Adam(dis_m.parameters(), lr=0.0002)

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

            # discriminator female
            optim_dis_f.zero_grad()

            d_real_preds = dis_f(female)
            labels = torch.ones_like(d_real_preds).to(config["device"])
            d_real_f_loss = criterion_mse(d_real_preds, labels)

            preds_female = decoder_f(encoder(female))

            d_fake_f_preds = dis_f(preds_female)
            labels = torch.zeros_like(d_fake_f_preds).to(config["device"])
            d_fake_f_loss = criterion_mse(d_fake_f_preds, labels)
            d_loss_f = (d_real_f_loss + d_fake_f_loss) / 2
            d_loss_f.backward()
            optim_dis_f.step()

            # discriminator male
            optim_dis_m.zero_grad()

            d_real_preds = dis_m(male)
            labels = torch.ones_like(d_real_preds).to(config["device"])
            d_real_m_loss = criterion_mse(d_real_preds, labels)

            preds_male = decoder_f(encoder(male))

            d_fake_m_preds = dis_m(preds_male)
            labels = torch.zeros_like(d_fake_m_preds).to(config["device"])
            d_fake_m_loss = criterion_mse(d_fake_m_preds, labels)
            d_loss_m = (d_real_m_loss + d_fake_m_loss) / 2
            d_loss_m.backward()
            optim_dis_m.step()

            # encoder backward

            optim.zero_grad()

            preds_male = decoder_m(encoder(male))
            preds_female = decoder_f(encoder(female))

            preds_real_fake_m = dis_m(preds_male)
            preds_real_fake_f = dis_f(preds_female)
            labels = torch.zeros_like(preds_real_fake_m).to(config["device"])

            loss_male = config["l1_wight"] * criterion_l1(male, preds_male) + criterion_mse(preds_real_fake_m, labels)
            loss_female = config["l1_wight"] * criterion_l1(female, preds_female) + criterion_mse(preds_real_fake_f, labels)
            loss = (loss_female + loss_male) / 2
            loss.backward()
            optim.step()

    return encoder, decoder_m, decoder_f
