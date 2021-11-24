import torch
import torch.nn as nn
import torch.nn.functional as F

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

if __name__ == '__main__':
    model = Net()
    x = torch.rand(4, 1, 192, 9)
    y = torch.max(torch.rand(4, 6, 21), 2)[1]
    out = torch.reshape(model(x), (4, 6, 21))
    loss = nn.CrossEntropyLoss()
    z = -F.log_softmax(out, 2)
    print(z[0])
    print(y.unsqueeze(2)[0])
    z = z.gather(2, y.unsqueeze(2))
    print(z.sum(1))
    print(torch.sum(z, 1).mean())