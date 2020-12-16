import sys
import json
import torch

from .train.non_adversarial import train
from .test import test


def set_config():
    with open("config/config.json", "r") as rdr:
        config = json.load(rdr)
    return config


if __name__ == "__main__":
    config = set_config()
    if sys.argv[1] == "--train":
        encoder, decoder_m, decoder_f = train(config)

        torch.save(decoder_m, "decoder_male.model.pth")
        torch.save(decoder_f, "decoder_female.model.pth")
        torch.save(encoder, "encoder.model.pth")
    elif sys.argv[1] == "--test":
        test(config, sys.argv[2])
    else:
        print("something go wrong")