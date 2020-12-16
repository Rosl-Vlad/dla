from torch import nn


class LSTMBlock(nn.Module):
    def __init__(self, inp, outp, n_layers=1, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = outp

        self.lstm = nn.GRU(inp, hidden, num_layers=n_layers, bidirectional=True)
        self.clf = nn.Linear(hidden * 2, outp)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.clf(x)
        x = x.permute(1, 2, 0)
        return x


class ConvBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=1, padding=(kernel_size) // 2, bias=False)

        self.conv1_drop1 = nn.Dropout2d(0.1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=2, padding=(kernel_size - 1) // 2, bias=False)

        self.conv1_drop2 = nn.Dropout2d(0.1)

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_drop1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.conv1_drop2(x)
        x = self.bn2(x)

        x = self.relu(x)
        return x


class ConvDis(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)