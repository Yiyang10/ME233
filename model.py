import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import convert_P_ten, convert_I_TC, convert_I_shr, convert_P_shr

class Selfupdatingpara(nn.Module):
    def __init__(self):
        super(Selfupdatingpara,self).__init__()

        self.a = nn.Parameter(torch.tensor([np.e]), requires_grad=True)
        self.b = nn.Parameter(torch.tensor([np.e]), requires_grad=True)

    def forward(self, input):
        input2 = torch.clamp(input[:, 1:2], min=1e-6)
        b = F.softplus(self.b)
        # Assuming 'input' is of shape [batch, 3]
        acfn1 = torch.pow(self.a, input[:, 0:1]) - 1  # Shape: [batch, 1]
        acfn2 = - torch.log(1 - input2) / torch.log(b)  # Shape: [batch, 1]
        acfn3 = input[:, 2:3]  # Shape: [batch, 1]
        return acfn1, acfn2, acfn3

class Modeldiscover(nn.Module):
    def __init__(self, mode = "TC"):
        super(Modeldiscover, self).__init__()
        self.mode = mode
        # (I1 - 3)
        self.fcI11 = nn.Linear(1, 3, bias=False)
        self.fcI12 = nn.Linear(1, 3, bias=False)
        self.fcI13 = nn.Linear(1, 3, bias=False)
        self.fcI21 = nn.Linear(1, 3, bias=False)
        self.fcI22 = nn.Linear(1, 3, bias=False)
        self.fcI23 = nn.Linear(1, 3, bias=False)

        self.fc = nn.Linear(18, 1, bias=False)

        self.selfupdatingpara = Selfupdatingpara()

    def forward(self, x):
        # It is assumed that I1 and I2 are tensors of shape [batch, 1]
        # and that they require gradients (i.e. require_grad=True)

        feature_list = []
        stretch = x
        if self.mode == 'tension' or self.mode == 'compression':
            I1, I2 = convert_I_TC(x)
        else:
            I1, I2 = convert_I_shr(x)
        I1 = I1.unsqueeze(1).float(); I2 = I2.unsqueeze(1).float()
        I1 = I1.detach().clone().requires_grad_(True)
        I2 = I2.detach().clone().requires_grad_(True)
        # For I1: using powers 1, 2, and 3 (after subtracting 3)
        out1 = self.fcI11(I1 - 3)
        acf1, acf2, acf3 = self.selfupdatingpara(out1)
        feature_list.extend([acf1, acf2, acf3])

        out2 = self.fcI12(torch.pow(I1 - 3, 2))
        acf1, acf2, acf3 = self.selfupdatingpara(out2)
        feature_list.extend([acf1, acf2, acf3])

        out3 = self.fcI13(torch.pow(I1 - 3, 3))
        acf1, acf2, acf3 = self.selfupdatingpara(out3)
        feature_list.extend([acf1, acf2, acf3])

        # For I2: using powers 1, 2, and 3 (after subtracting 3)
        out4 = self.fcI21(I2 - 3)  # power 1
        acf1, acf2, acf3 = self.selfupdatingpara(out4)
        feature_list.extend([acf1, acf2, acf3])

        out5 = self.fcI22(torch.pow(I2 - 3, 2))
        acf1, acf2, acf3 = self.selfupdatingpara(out5)
        feature_list.extend([acf1, acf2, acf3])

        out6 = self.fcI23(torch.pow(I2 - 3, 3))
        acf1, acf2, acf3 = self.selfupdatingpara(out6)
        feature_list.extend([acf1, acf2, acf3])
        # Concatenate all 18 features along the feature dimension.
        # Each acf has shape [batch, 1] so the result is [batch, 18]
        x = torch.cat(feature_list, dim=1)
        # Compute the network output, interpreted here as the strain energy Ψ.
        Psi = self.fc(x)  # shape: [batch, 1]

        # Now, compute the gradients of Ψ with respect to I1 and I2.
        # (We assume that I1 and I2 are leaf nodes with requires_grad=True.)
        dPsi_dI1 = torch.autograd.grad(
            Psi, I1,
            grad_outputs=torch.ones_like(Psi),
            create_graph=True,
            retain_graph=True
        )[0]

        dPsi_dI2 = torch.autograd.grad(
            Psi, I2,
            grad_outputs=torch.ones_like(Psi),
            create_graph=True,
            retain_graph=True
        )[0]

        # Here we call convert_P_ten with the tuple (dPsi_dI1, dPsi_dI2, I1).
        # (If your “Stretch” should be I2 or some function of I1/I2, adjust accordingly.)
        # bounce between shear and TC
        if self.mode == "shear":
            stress = convert_P_shr((dPsi_dI1, dPsi_dI2, stretch.unsqueeze(1).to(torch.float32)))
        else:
            stress = convert_P_ten((dPsi_dI1, dPsi_dI2, stretch.unsqueeze(1).to(torch.float32)))
        # print("The type of I1 is", I1.dtype)
        return stress

def loss(f_actual, f_pred):
    return F.mse_loss(f_actual, f_pred)