import torch.nn as nn


class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2, 3).contiguous()
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()


class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x


class BidirectionalLSTM(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout):
        super(BidirectionalLSTM, self).__init__()

        self.BiLSTM = nn.LSTM(input_size=rnn_dim, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiLSTM(x)
        x = self.dropout(x)
        return x


class Classifier(nn.Module):
    def __init__(self, dim, n_classes, dropout):
        super(Classifier, self).__init__()
        self.clf1 = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout)
        self.clf2 = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.clf1(x)
        x = self.drop(x)
        return self.clf2(x)


class ASR(nn.Module):
    def __init__(self, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(ASR, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)

        self.rescnn_layers1 = ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
        self.rescnn_layers2 = ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)

        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)

        self.birnn_layers = BidirectionalLSTM(rnn_dim=rnn_dim, hidden_size=rnn_dim, dropout=dropout)

        self.classifier = Classifier(rnn_dim, n_class, dropout)

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers1(x)
        x = self.rescnn_layers2(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x