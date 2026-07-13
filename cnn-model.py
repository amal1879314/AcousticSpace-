import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNBaseline, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(64,num_classes)
        )

    def forward(self,x):

        x=self.features(x)

        x=self.classifier(x)

        return x
