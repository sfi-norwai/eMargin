import random
from src.models.augclass import *
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from torch.nn.modules.module import Module

class AUGs(Module):
    def __init__(self,caugs,p=0.2):
        super(AUGs,self).__init__()

        self.augs = caugs
        self.p = p
    def forward(self,x_torch):
        x = x_torch.clone()
        for a in self.augs:
            if random.random()<self.p:
                x = a(x)
        return x.clone(),x_torch.clone()

class RandomAUGs(Module):
    def __init__(self,caugs,p=0.2):
        super(RandomAUGs,self).__init__()

        self.augs = caugs
        self.p = p
    def forward(self,x_torch):
        x = x_torch.clone()

        if random.random()<self.p:
            x =random.choice(self.augs)(x)
        return x.clone(),x_torch.clone()

class AutoAUG(Module):
    def __init__(self, aug_p1=0.2, aug_p2 = 0.0, used_augs=None, device=None, dtype=None) -> None:
        super(AutoAUG,self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.device = device
        all_augs = [subsequence(),cutout(), jitter(), scaling(), time_warp(), window_slice(), window_warp()]

        if used_augs is not None:
            self.augs = []
            for i in range(len(used_augs)):
                if used_augs[i]:
                    self.augs.append(all_augs[i])
        else:
            self.augs = all_augs
        self.weight = Parameter(torch.empty((2,len(self.augs)), **factory_kwargs)).to(device)
        self.reset_parameters()
        self.aug_p1 = aug_p1
        self.aug_p2 = aug_p2

    def get_sampling(self,temperature=1.0, bias=0.0):

        if self.training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(self.weight.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(self.device)
            gate_inputs = (gate_inputs + self.weight) / temperature
            # para = torch.sigmoid(gate_inputs)
            para = torch.softmax(gate_inputs,-1)
            return para
        else:
            return torch.softmax(self.weight,-1)


    def reset_parameters(self) -> None:
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.01)
    def forward(self,xt):
        x,t = xt
        if self.aug_p1 ==0 and self.aug_p2==0:
            return x.clone(), x.clone()
        para = self.get_sampling()

        if random.random()>self.aug_p1 and self.training:
            aug1 = x.clone()
        else:
            xs1_list = []
            for aug in self.augs:
                xs1_list.append(aug(x))
            xs1 = torch.stack(xs1_list, 0)
            xs1_flattern = torch.reshape(xs1, (xs1.shape[0], xs1.shape[1] * xs1.shape[2] * xs1.shape[3]))
            aug1 = torch.reshape(torch.unsqueeze(para[0], -1) * xs1_flattern, xs1.shape)
            aug1 = torch.sum(aug1,0)

        aug2 = x.clone()

        return aug1,aug2
