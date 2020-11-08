import torch
import torch.nn as nn
from datetime import datetime
import torch.nn.functional as F
from model.model import ASR


def get_time(start_time):
    duration = datetime.now() - start_time
    days, seconds = duration.days, duration.seconds
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return minutes, seconds


def _train(train_loader, epoch, model, optimizer, criterion, device, batch_size):
    model.train()
    train_loss = 0.0
    start_time = datetime.now()
    for idx, (melspec, tokens, target_len, padded_len) in enumerate(train_loader):
        melspec, tokens = melspec.to(device), tokens.to(device)

        optimizer.zero_grad()

        outputs = model(melspec.unsqueeze(1).transpose(2, 3).to(device))
        outputs = F.log_softmax(outputs, dim=2)

        loss = criterion(outputs.transpose(0, 1), tokens, padded_len, target_len)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        minutes, seconds = get_time(start_time)

        print("\r Train Epoch {} [{}/{} ({:.0f}%)] loss: {:.7f} time: {}m:{}s".format(epoch,
                                                                                      (idx + 1) * batch_size,
                                                                                      len(train_loader.dataset),
                                                                                      100. * idx / len(train_loader),
                                                                                      loss.item(),
                                                                                      minutes,
                                                                                      seconds), end='')
        if (idx + 1) % 10 == 0:
            pass
            # wandb.log({"Loss": loss})
    print()


def train(config, train_loader, device):
    model = ASR(config["rnn_dim"], config["n_class"], config["n_feats"])
    model = model.to(device)

    criterion = nn.CTCLoss(blank=28).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["epoch"]):
        _train(train_loader, epoch, model, optimizer, criterion, device, config["batch_size"])

    return model, criterion

