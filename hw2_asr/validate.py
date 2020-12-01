import jiwer
import torch
import torch.nn.functional as F


def int_to_text(labels, id_latter):
    string = ""
    for i in labels:
        string += id_latter[i]
    return string


def greedy_decoder(output, labels, label_lengths, id_letters, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []

    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(int_to_text(labels[i][:label_lengths[i]].tolist(), id_letters))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(int_to_text(decode, id_letters))
    return decodes, targets


def evaluate(valid_loader, model, criterion, device, id_letters):
    model.eval()
    loss = 0
    test_wer = []

    with torch.no_grad():
        for i, (melspec, tokens, target_len, padded_len) in enumerate(valid_loader):
            melspec, tokens = melspec.to(device), tokens.to(device)

            outputs = model(melspec.unsqueeze(1).transpose(2, 3))
            outputs = F.log_softmax(outputs, dim=2)

            loss += criterion(outputs.transpose(0, 1), tokens, padded_len, target_len).item()

            decoded_preds, decoded_targets = greedy_decoder(outputs, tokens, target_len, id_letters)

            for j in range(len(decoded_preds)):
                test_wer.append(jiwer.wer(decoded_targets[j], decoded_preds[j]))

    loss /= len(valid_loader)
    #wandb.log({"Validation loss": loss})

    avg_wer = sum(test_wer) / len(test_wer)
    #wandb.log({"WER": avg_wer})
    print('Validation: Average loss: {:.4f}, Average WER: {:.4f}\n'.format(loss, avg_wer))
