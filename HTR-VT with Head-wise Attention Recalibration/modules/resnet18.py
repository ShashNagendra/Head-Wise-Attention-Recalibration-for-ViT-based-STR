import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# class ResNet18(nn.Module):
#     def __init__(self, nb_feat=768):  # Should match embed_dim
#         super(ResNet18, self).__init__()
#         self.inplanes = nb_feat // 4

        
        
        
#         # Initial convolution block
#         self.conv1 = nn.Conv2d(1, nb_feat//4, kernel_size=7, stride=4, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(nb_feat // 4, eps=1e-05)
#         self.relu = nn.ReLU(inplace=True)
#         #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         # Residual blocks
#         self.layer1 = self._make_layer(BasicBlock, nb_feat // 4, 2, stride=(2, 1))
#         self.layer2 = self._make_layer(BasicBlock, nb_feat // 2, 2, stride=2)
#         self.layer3 = self._make_layer(BasicBlock, nb_feat, 2, stride=2)
        
#         # Final pooling
#         self.final_pool = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)

class ResNet18(nn.Module):
    def __init__(self, nb_feat=768):
        super(ResNet18, self).__init__()
        self.inplanes = nb_feat // 4

        
        # First conv with stride 4 to quickly reduce dimensions
        self.conv1 = nn.Conv2d(1, nb_feat//4, kernel_size=7, stride=4, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(nb_feat//4)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # Modified stride
        
        # Subsequent layers with adjusted strides
        self.layer1 = self._make_layer(BasicBlock, nb_feat//4, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, nb_feat//2, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, nb_feat, 2, stride=2)
        
        # Final pooling with adjusted kernel/stride
        self.final_pool = nn.AdaptiveAvgPool2d((14, 14))  # Force 14x14 output

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if not any(p.is_cuda for p in self.parameters()):
            # Model is on CPU
            if x.is_cuda:
                x = x.cpu()
        else:
            # Model is on GPU
            if not x.is_cuda:
                x = x.cuda()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_pool(x)
        
        return x
    


