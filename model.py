import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.05


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=4, kernel_size=(3, 3), padding=0, bias=False
            ),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )  # output_size = 26, RF = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=4, out_channels=4, kernel_size=(3, 3), padding=0, bias=False
            ),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )  # output_size = 24, RF = 5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=4, out_channels=8, kernel_size=(3, 3), padding=0, bias=False
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )  # output_size = 22, RF = 7

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 11, RF = 8
        self.convblock4 = nn.Sequential(
            nn.Conv2d(
                in_channels=8, out_channels=8, kernel_size=(1, 1), padding=0, bias=False
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )  # output_size = 11, RF = 8

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )  # output_size = 9, RF = 12
        self.convblock6 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )  # output_size = 7, RF = 16

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )  # output_size = 7, RF = 20
        self.convblock7a = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )  # output_size = 7, RF = 24
        self.convblock7b = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )  # output_size = 7, RF = 28

        self.convblock8 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )  # output_size = 7, RF = 28
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7),
        )  # output_size = 1, RF = 38

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock7a(x)
        x = self.convblock8(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
