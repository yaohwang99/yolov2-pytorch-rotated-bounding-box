import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, need_pool=False):
        nn.Module.__init__(self)
        padding = 1 if kernel_size > 1 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if need_pool else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

class YOLOv2(nn.Module):
    def __init__(self, num_classes, anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                          (11.2364, 10.0071)]):
        super(YOLOv2, self).__init__()
        self.num_classes = num_classes
        self.num_boxes = len(anchors)
        self.anchors = anchors
        self.layer1 = Conv2d(3, 32, 3, True)
        self.layer2 = Conv2d(32, 64, 3, True)

        self.layer3 = Conv2d(64, 128, 3, False)
        self.layer4 = Conv2d(128, 64, 1, False)
        self.layer5 = Conv2d(64, 128, 3, True)
    
        self.layer6 = Conv2d(128, 256, 3, False)
        self.layer7 = Conv2d(256, 128, 1, False)
        self.layer8 = Conv2d(128, 256, 3, True)
    
        self.layer9 = Conv2d(256, 512, 3, False)
        self.layer10 = Conv2d(512, 256, 1, False)
        self.layer11 = Conv2d(256, 512, 3, False)
        self.layer12 = Conv2d(512, 256, 1, False)
        self.layer13 = Conv2d(256, 512, 3, True)

        self.layer14 = Conv2d(512, 1024, 3, False)
        self.layer15 = Conv2d(1024, 512, 1, False)
        self.layer16 = Conv2d(512, 1024, 3, False)
        self.layer17 = Conv2d(1024, 512, 1, False)
        self.layer18 = Conv2d(512, 1024, 3, False)

        self.head1 = nn.Sequential(Conv2d(1024, 1024, 3, False),
            Conv2d(1024, 1024, 1, False),
            Conv2d(1024, 1024, 3, False),
            Conv2d(1024, 1024, 1, False))
        self.head2 = nn.Sequential(
            Conv2d(2048, 1024, 3, False),
            Conv2d(1024, self.num_boxes * (self.num_classes + 5), 1, False))
    
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x_ = self.layer18(x)
        x = self.head1(x_)
        x = torch.cat([x, x_], 1)
        x = self.head2(x)

        return x

if __name__ == '__main__':
    model = YOLOv2(10)
    input = torch.randn(4, 3, 416, 416)
    pred = model.forward(input)
    print(pred.shape)
