import torch
from processing.dataset import get_loader
from config.config import set_config
from train import train
from eval import evaluate


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    set_seed()
    config = set_config()
    train_loader, valid_loader, id_letters = get_loader(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    asr, criterion = train(config, train_loader, device)
    evaluate(valid_loader, asr, criterion, device, id_letters)


if __name__ == '__main__':
    main()
