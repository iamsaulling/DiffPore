import torch.nn as nn

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),  # conv1_1
    nn.ReLU(inplace=True),  # relu1_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),  # conv1_2
    nn.ReLU(inplace=True),  # relu1_2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # pool1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),  # conv2_1
    nn.ReLU(inplace=True),  # relu2_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),  # conv2_2
    nn.ReLU(inplace=True),  # relu2_2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # pool2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),  # conv3_1
    nn.ReLU(inplace=True),  # relu3_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),  # conv3_2
    nn.ReLU(inplace=True),  # relu3_2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),  # conv3_3
    nn.ReLU(inplace=True),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),  # conv3_4
    nn.ReLU(inplace=True),  # relu3_4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # pool3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),  # conv4_1
    nn.ReLU(inplace=True),  # relu4_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),  # conv4_2
    nn.ReLU(inplace=True),  # relu4_2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),  # conv4_3
    nn.ReLU(inplace=True),  # relu4_3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),  # conv4_4
    nn.ReLU(inplace=True),  # relu4_4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  #pool4
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),  # conv5_1
    nn.ReLU(inplace=True),  # relu5_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),  # conv5_2
    nn.ReLU(inplace=True),  # relu5_2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),  # conv5_3
    nn.ReLU(inplace=True),  # relu5_3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),  # conv5_4
    nn.ReLU(inplace=True)  # relu5_4
)