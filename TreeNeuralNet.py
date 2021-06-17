import torch
import torch.nn as nn
import math
from utils import *


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class Node(nn.Module):
    def __init__(self, c_in, c_out, _s, _op, _name):
        super(Node, self).__init__()
        self.name = _name
        self.relu = nn.ReLU()
        self.alpha = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        if _op == 'conv_3':
            self.op = nn.Conv2d(c_in, c_out, kernel_size=3, stride=_s, padding=1, bias=False)
        elif _op == 'conv_5':
            self.op = nn.Conv2d(c_in, c_out, kernel_size=5, stride=_s, padding=2, bias=False)
        elif _op == 'dil_3':
            self.op = nn.Conv2d(c_in, c_out, kernel_size=3, stride=_s, padding=2, dilation=2,
                                bias=False)
        elif _op == 'avg_3':
            if _s != 1:
                self.op = nn.Sequential(
                    nn.Conv2d(c_in, c_out, kernel_size=1, stride=_s, padding=0, bias=False),
                    nn.AvgPool2d(3, 1, 1)
                    )
            else:
                self.op = nn.AvgPool2d(3, 1, 1)
        elif _op == 'identity':  # code 3
            if _s !=1:
                self.op = nn.Conv2d(c_in, c_out, kernel_size=1, stride=_s, padding=0, bias=False)
            else:
                self.op = Identity()
        else:
            raise Exception()

    def forward(self, _x):
        return self.relu(self.op(_x))


class TreeNeuralNet(nn.Module):
    def __init__(self, c, max_depth, borders, _ops, num_classes=10):
        super(TreeNeuralNet, self).__init__()
        self.ops = _ops
        self.max_depth = max_depth
        self.nodes = nn.ModuleDict()
        self.relu = nn.ReLU()
        self.init_conv = nn.Conv2d(3, c, 3, stride=1, padding=1)
        self.gp = nn.AdaptiveAvgPool2d(1)
        # c_mlp = int(c * math.pow(2, len(borders)) * math.pow(len(_ops), max_depth))
        c_mlp = int(c * math.pow(len(_ops), max_depth))
        print(f"MLP: {c_mlp}")
        self.fc = nn.Linear(c_mlp, num_classes)
        self._generate_layer(c, 0, borders, self.nodes, '')

    def forward(self, _x):
        # x: [n, c, h, w]
        outputs = {}
        _x = self.init_conv(_x)
        _x = self.relu(_x)
        for i, _o in enumerate(self.ops):
            self._forward(_x, self.nodes[f'{i}'], outputs)
        _x = torch.cat(list(outputs.values()), dim=1) # tree output
        _x = self.gp(_x)
        _y = self.fc(_x.view(_x.size(0), -1))
        return _y

    def _generate_layer(self, c_in, depth, borders, _nodes, _name):
        if depth < self.max_depth:
            if depth in borders:
                # c_out = c_in * 2
                c_out = c_in
                _s = 2
            else:
                c_out = c_in
                _s = 1
            for i, _o in enumerate(self.ops):
                next_name = _name + str(i)
                _nodes[next_name] = Node(c_in, c_out, _s, _o, next_name)
                self._generate_layer(c_out, depth+1, borders, _nodes, next_name)

    def _forward(self, _x, _node, outputs):
        if len(_node.name) == self.max_depth:
            # print(_node.name)
            # leaf node
            outputs[_node.name] = _node.alpha * _node(_x)
        else:
            for i, _o in enumerate(self.ops):
                next_name = _node.name + str(i)
                self._forward(_node.alpha * _node(_x), self.nodes[next_name], outputs)


if __name__ == "__main__":
    x = torch.randn(5, 3, 32, 32)
    # ns = [Node(10, 10, 2, _o) for _o in ['conv_3', 'dil_3', 'avg_3', 'identity']]
    # for n in ns:
    #     print(n(x).shape)
    c = 4
    max_depth = 5
    borders = [2, 4]
    _ops = ['conv_3', 'dil_3', 'identity', 'avg_3']
    tnet = TreeNeuralNet(c, max_depth, borders, _ops)
    print(number_of_params(tnet))
    print(tnet(x).shape)
