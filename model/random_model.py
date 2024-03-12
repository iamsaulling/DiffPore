import torch
import torch.nn as nn
from config import parse_args
args = parse_args()


randomModel = nn.Sequential(
    nn.Conv2d(3, 128, 3, padding=1),
    nn.ReLU(),  # 1

    nn.Conv2d(3, 128, 5, padding=2),
    nn.ReLU(),  # 3

    nn.Conv2d(3, 128, 7, padding=3),
    nn.ReLU(),  # 5

    nn.Conv2d(3, 128, 11, padding=5),
    nn.ReLU(),  # 7

    nn.Conv2d(3, 128, 15, padding=7),
    nn.ReLU(),  # 9

    nn.Conv2d(3, 128, 23, padding=11),
    nn.ReLU(),  # 11

    nn.Conv2d(3, 128, 37, padding=18),
    nn.ReLU(),  # 13

    nn.Conv2d(3, 128, 55, padding=27),
    nn.ReLU()   # 15
)
