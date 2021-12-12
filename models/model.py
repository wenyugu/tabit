import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from layers import *
from layers import _gen_bias_mask, _gen_timing_signal
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*93, 128)
        self.fc2 = nn.Linear(128, 126)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout1(self.maxpool(x))
        x = torch.flatten(x, 1)
        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CNN_LSTM(nn.Module):
    def __init__(self, input_size=192, hidden_size=256, num_layers=1, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(1280, 512)
        if self.bidirectional:
            self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, 126)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional)

    def forward(self, x):
        bs, seqlen, num_features = x.size()
        x1 = x.unsqueeze(1)
        x1 = self.maxpool(F.relu(self.bn1(self.conv1(x1))))
        x1 = self.maxpool(F.relu(self.bn2(self.conv2(x1))))
        x1 = self.maxpool(F.relu(self.bn3(self.conv3(x1))))
        x1 = self.dropout1(self.maxpool(F.relu(self.conv4(x1))))
        x1 = x1.view(bs, -1).unsqueeze(1).repeat(1, seqlen, 1)
        x2, (hn, cn) = self.lstm(x)
        x = torch.cat((x1, x2), 2)
        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class Resnet_LSTM(nn.Module):
    def __init__(self, input_size=192, hidden_size=256, num_layers=1, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1.in_channels = 1
        self.resnet.fc.out_features = 1024
        self.resnet.conv1.weight = nn.parameter.Parameter(self.resnet.conv1.weight.mean(dim=1, keepdim=True))
        self.resnet.fc = nn.Linear(512, 1024)
        self.fc1 = nn.Linear(1280, 512)
        if self.bidirectional:
            self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, 126)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional)

    def forward(self, x):
        bs, seqlen, num_features = x.size()
        x1 = x.unsqueeze(1)
        x1 = self.resnet(x1)
        x1 = x1.view(bs, -1).unsqueeze(1).repeat(1, seqlen, 1)
        x2, (hn, cn) = self.lstm(x)
        x = torch.cat((x1, x2), 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # model = Net()
    # x = torch.rand(4, 1, 192, 9)
    # y = torch.max(torch.rand(4, 6, 21), 2)[1]
    # output = model(x)
    # print(output.size())
    # out = torch.reshape(output, (4, 6, 21))
    # loss = nn.CrossEntropyLoss()
    # z = -F.log_softmax(out, 2)
    # print(z[0])
    # print(y.unsqueeze(2)[0])
    # z = z.gather(2, y.unsqueeze(2))
    # print(z.shape)
    # print(z.sum(1))
    # print(torch.sum(z, 1).mean())
    # take 5 sec 
    config = {  'feature_size' : 192,
                'timestep' : 216,
                'num_chords' : 126,
                'input_dropout' : 0.2,
                'layer_dropout' : 0.2,
                'attention_dropout' : 0.2,
                'relu_dropout' : 0.2,
                'num_layers' : 8,
                'num_heads' : 4,
                'hidden_size' : 128,
                'total_key_depth' : 128,
                'total_value_depth' : 128,
                'filter_size' : 128,
                'loss' : 'ce',
                'probs_out' : False}
    model = CNN_LSTM(bidirectional=True)
    x = torch.rand(2, 216, 192)
    y = torch.max(torch.rand(2 * 216, 6, 21), 2)[1]
    output = model(x)
    print(output.size())
    out = torch.reshape(output, (-1, 6, 21))
    z = -F.log_softmax(out, -1)
    z = z.gather(2, y.unsqueeze(2))
    print(z.shape)
    print(torch.sum(z, 1).mean())