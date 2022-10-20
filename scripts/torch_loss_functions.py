import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

    
class SequentiaSizeLoss(nn.Module):
    #boundを引数とする
    def __init__(self, bound, alpha=0.1):
        super(SequentiaSizeLoss, self).__init__()
        self.dice_loss = BCEDiceLoss()
        self.bound = bound
        self.alpha = alpha
    
    def forward(self, input, target):
        #dice = self.dice_loss(input[base_slice_index], target[base_slice_index])
        dice = self.dice_loss(input, target)
        ssl_batch = 0
        for i in range(len(input)):
            diff_sum = 0
            for j in range(len(input[0]) - 1):   
                diff = input[i, j + 1].sum() - input[i, j].sum()
                diff = diff.abs()
                diff_sum = diff_sum + (F.relu(diff - self.bound)**2)
            ssl_batch = ssl_batch + diff_sum
        print(dice)
        print(ssl_batch)
        return dice + (self.alpha * ssl_batch)
        