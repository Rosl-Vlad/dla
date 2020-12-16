from blocks import ConvBasicBlock, ConvDis
from torch import nn


class Encoder(nn.Module):
    def block(self, inp, outp, kernel):
        return nn.Sequential(
            ConvBasicBlock(inp, outp, kernel)
        )

    def __init__(self, inp, ch, ks):
        super().__init__()
        self.conv = nn.Sequential(*[self.block(ch[i - 1] if i > 0 else inp,
                                               ch[i],
                                               ks[i],
                                               1
                                               ) for i in range(len(ks))])

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def block(self, inp, outp, kernel, stride, bnrelu=True, bn=True, lstm=False):
        return nn.Sequential(
            nn.ConvTranspose1d(inp, outp, kernel, stride=stride, padding=(kernel - 1) // 2),
            nn.BatchNorm1d(outp) if bnrelu else nn.Identity(),
            nn.ReLU() if bnrelu else nn.Identity(),
        )

    def __init__(self, inp, ch, ks):
        super().__init__()
        self.conv = nn.Sequential(*[self.block(ch[i - 1] if i > 0 else inp,
                                               ch[i],
                                               ks[i],
                                               2,
                                               bnrelu=(i + 1 < len(ks)),
                                               ) for i in range(len(ks))])

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, n_channel=80):
        super().__init__()
        self.conv1 = ConvDis(n_channel, n_channel * 2, 32)
        self.conv2 = ConvDis(n_channel * 2, n_channel * 4, 16)
        self.conv3 = ConvDis(n_channel * 4, n_channel * 8, 8)
        self.conv4 = ConvDis(n_channel * 8, n_channel * 10, 4)
        self.clf = nn.Linear(n_channel * 10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.clf(x.reshape(x.shape[0], x.shape[1]))

        return x
