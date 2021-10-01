from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
import time

from pynvml import *

def make_linear_grad_of(weight):
    def grad(x):
        # `(N, in\_features)`
        assert(len(x.shape)==2)
        weight_batch = weight.view(1, weight.size(0), weight.size(1))
        weight_batch = weight_batch.expand(x.size(0), weight.size(0), weight.size(1))
        return weight_batch
    return grad
def make_elu_grad_of(alpha):
    def grad(x):
        # `(N, in\_features)`
        assert(len(x.shape)==2)
        grad_batch = torch.where(x > 0.0, torch.ones_like(x), alpha * torch.exp(x))
        grad_batch = torch.diag_embed(grad_batch)
        return grad_batch
    return grad

def getGMem(i):
    h = nvmlDeviceGetHandleByIndex(i)
    info = nvmlDeviceGetMemoryInfo(h)
    total_memory_gb_used = round(info.used/1024**3, 1)
    total_memory_gb_total = round(info.total/1024**3, 1)
    total_memory_gb_free = round((info.total-info.used)/1024**3, 1)
    return total_memory_gb_total, total_memory_gb_used, total_memory_gb_free

class NetAutoDec(nn.Module):
    def __init__(self, net_automap):
        super(NetAutoDec, self).__init__()
        self.dec0 = net_automap.dec0
        self.dec1 = net_automap.dec1
        self.dec2 = net_automap.dec2
        self.dec3 = net_automap.dec3
        self.dec4 = net_automap.dec4
        self.enc = net_automap.enc
    
    def act(self, input):
        return F.elu(input)
    
    def forward(self, x):
        x = self.act(self.dec0(x))
        x = self.act(self.dec1(x))
        x = self.act(self.dec2(x))
        x = self.act(self.dec3(x))
        x = self.act(self.dec4(x))
        x = self.enc(x)  # identity activation
        return x

class NetGrad(nn.Module):
    def __init__(self, net):
        super(NetGrad, self).__init__()
        assert(net.layers is not None)
        self.layers = net.layers
        for layer in self.layers:
            if layer.__class__.__name__ == 'Linear':
                layer.grad_func = make_linear_grad_of(layer.weight)
            elif layer.__class__.__name__ == 'ELU':
                layer.grad_func = make_elu_grad_of(layer.alpha)
            else:
                print(layer.__class__.__name__)
                exit('invalid grad layer')

    def forward(self, x):
        with torch.no_grad():
            grad_inputs = []
            for layer in self.layers:
                grad_inputs.append(x)
                x = layer(x)
            grad = None
            for layer, grad_input in zip(self.layers, grad_inputs):
                if grad is None:
                    grad = layer.grad_func(grad_input)
                else:
                    grad = torch.matmul(layer.grad_func(grad_input), grad)
            return grad

def conv1dLayer(l_in, ks, strides):
    return math.floor(float(l_in -(ks-1)-1)/strides + 1)
class NetAutoEnc(nn.Module):
    def __init__(self, npoints=-1, lbllength=5):
        super(NetAutoEnc, self).__init__()
        if npoints == -1:
            self.npoints = 1368
        else:
            self.npoints = npoints
        self.dim = 3
        self.lbllength = lbllength

        ks = 6
        strides = 2

        # goal: [3, npoints] --> 32
        self.layers_conv = nn.ModuleList()
        l_in = npoints
        layer_cnt = 0

        while True:
            l_out = conv1dLayer(l_in, ks, strides)
            if 3*l_out >= 32:
                l_in = l_out
                layer_cnt += 1
                self.layers_conv.append(nn.Conv1d(3, 3, ks, strides))
            else:
                break
        self.enc10 = nn.Linear(3*l_in, 32)
        self.enc11 = nn.Linear(32, self.lbllength)

        self.lbllengthdim = self.lbllength + self.dim
        scale1 = 10
        scale2 = 1

        self.layers = []
        self.dec0 = nn.Linear(self.lbllengthdim, self.dim * scale1 * scale2)
        self.layers.append(self.dec0)
        self.layers.append(torch.nn.ELU())
        self.dec1 = nn.Linear(self.dim * scale1 * scale2,
                              self.dim * scale1 * scale2)
        self.layers.append(self.dec1)
        self.layers.append(torch.nn.ELU())
        self.dec2 = nn.Linear(self.dim * scale1 * scale2,
                              self.dim * scale1 * scale2)
        self.layers.append(self.dec2)
        self.layers.append(torch.nn.ELU())
        self.dec3 = nn.Linear(self.dim * scale1 * scale2,
                              self.dim * scale1 * scale2)
        self.layers.append(self.dec3)
        self.layers.append(torch.nn.ELU())
        self.dec4 = nn.Linear(self.dim * scale1 * scale2,
                              self.dim * scale1 * scale2)
        self.layers.append(self.dec4)
        self.layers.append(torch.nn.ELU())
        self.enc = nn.Linear(self.dim*scale1*scale2,
                              self.dim)
        self.layers.append(self.enc)
    
    def act(self, input):
        return F.elu(input)
    
    def forward(self, x):
        state = x[:,:, :self.dim]
        x0 = x[:, :, -self.dim:]

        state = torch.transpose(state, 1, 2)
        
        for layer in self.layers_conv:
            state = self.act(layer(state))
        
        state = torch.transpose(state, 1, 2)
        state = state.reshape(-1, state.size(1)*state.size(2))
        state = self.act(self.enc10(state))
        xhat = self.act(self.enc11(state))
        xhat = xhat.view(xhat.size(0), 1, xhat.size(1))

        xhat = xhat.expand(xhat.size(0), self.npoints, xhat.size(2))
        x = torch.cat((xhat, x0), 2)
        x = x.view(x.size(0)*x.size(1), x.size(2))

        x = self.act(self.dec0(x))
        x = self.act(self.dec1(x))
        x = self.act(self.dec2(x))
        x = self.act(self.dec3(x))
        x = self.act(self.dec4(x))

        x = self.enc(x)  # identity activation

        return x

class NetAutoEncEnc(nn.Module):
    def __init__(self, net_autoenc):
        super(NetAutoEncEnc, self).__init__()
        self.layers_conv = net_autoenc.layers_conv 
        
        self.enc10 = net_autoenc.enc10
        self.enc11 = net_autoenc.enc11
    
    def act(self, input):
        return F.elu(input)

    def forward(self, state):
        state = torch.transpose(state, 1, 2)
        for layer in self.layers_conv:
            state = self.act(layer(state))
        state = torch.transpose(state, 1, 2)
        state = state.reshape(-1, state.size(1)*state.size(2))
        state = self.act(self.enc10(state))
        xhat = self.act(self.enc11(state))
        xhat = xhat.view(xhat.size(0), 1, xhat.size(1))
        return xhat