from .processing.dataset import *
from .config.config import set_config
from .train import train
from .eval import evaluate


def main():
    set_seed()
    config = set_config()
    train_loader, valid_loader, id_letters = get_loader(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    asr, criterion = train(config, train_loader, device)
    evaluate(valid_loader, asr, criterion, device, id_letters)


if __name__ == '__main__':
    main()
