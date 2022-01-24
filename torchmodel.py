import torch
import math
import numpy as np
from torchvision import transforms

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

###TODO Look for getting an LSTM output directly instead of fully connected fc layer
class ChannelAttention(torch.nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.fc = torch.nn.Sequential(torch.nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = torch.nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes * 4)
        self.relu = torch.nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(torch.nn.Module):

    def __init__(self, block, layers, num_classes=512):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m,torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class physical_encoder(torch.nn.Module):
    def __init__(self, phys_shape):
        super(physical_encoder, self).__init__()

        self.phys_shape = phys_shape
        self.LSTM1 = torch.nn.LSTM(input_size=self.phys_shape,hidden_size=256,num_layers=1,batch_first=True)
        #self.fc = torch.nn.Linear(256, 256)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x, _status = self.LSTM1(x)
        #x = self.fc(x)
        return x

class error_encoder(torch.nn.Module):
    def __init__(self, error_shape):
        super(error_encoder, self).__init__()

        self.error_shape = error_shape
        self.LSTM2 = torch.nn.LSTM(input_size=self.error_shape,hidden_size=256,num_layers=1,batch_first=True)
        #self.fc = torch.nn.Linear(256,256)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,x):
        x, _status = self.LSTM2(x)
        #x = self.fc(x)
        return x

class self_attention(torch.nn.Module):
    def __init__(self):
        super(self_attention, self).__init__()

        self.fc_q = torch.nn.Linear(1024,1024)
        self.fc_k = torch.nn.Linear(1024,1024)
        self.fc_v = torch.nn.Linear(1024,1024)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)

        kt = torch.transpose(K, 2, 1)
        q_k = torch.matmul(Q, kt)

        dk = K.shape[1]# 3
        logits = q_k / np.sqrt(dk)
        att_wts = torch.softmax(logits, dim=-1)
        oupt = torch.matmul(att_wts, V)
        return oupt

class Soft_Mask(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Soft_Mask, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.output_dim),
            torch.nn.Sigmoid()
        )

        self.dropout = torch.nn.Dropout(0.2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat):
        feat_new = self.fc(feat)
        return feat_new

class BiDirectional(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super(BiDirectional,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.biLSTM = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(self.hidden_size*2, self.output_dim)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,x):
        out, _status = self.biLSTM(x)
        out = self.fc(out)

        return out

class viae_net(torch.nn.Module):
    def __init__(self, phy_shape, error_shape, attention=False):
        super(viae_net, self).__init__()

        self.attention = attention
        self.phy_encoder = physical_encoder(phy_shape)
        self.err_encoder = error_encoder(error_shape)
        self.self_attention = self_attention()
        self.cbam_resnet = ResNet(BasicBlock, [3, 4, 6, 3])
        self.Soft_Mask = Soft_Mask(1024,1024)
        self.bidirectionalLSTM = BiDirectional(1024, 1000, 2, 1)
        self.scale = 1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, ximg, xphy, xerr):
        err_out = self.err_encoder(xerr)
        phy_out = self.phy_encoder(xphy)
        physical_layer = torch.cat([err_out, phy_out], dim=2)
        cbam_out = []
        for i in range(0, len(ximg)):
            cbam_out.append(self.cbam_resnet(ximg[i]))
        cbam_out = torch.stack(cbam_out)
        cbam_out = torch.cat([physical_layer, cbam_out], dim=2)
        if self.attention == True:
            cbam_out = self.self_attention(cbam_out)
        out = self.Soft_Mask(cbam_out)
        out = self.bidirectionalLSTM(out)
        return out
