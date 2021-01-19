import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, is_canceled=False, adr=False):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(77, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 8)
        # self.linear4 = nn.Linear(8, 3)
        self.linear4 = None
        if is_canceled:
            self.linear4 = nn.Linear(8, 2)
        if adr:
            self.linear4 = nn.Linear(8, 1)
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = self.linear4(out)
        return out