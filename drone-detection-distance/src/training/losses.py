
import torch.nn as nn

class DistanceLoss(nn.Module):
    def __init__(self, kind='smooth_l1'):
        super().__init__()
        if kind == 'smooth_l1':
            self.fn = nn.SmoothL1Loss(reduction='mean')
        else:
            self.fn = nn.MSELoss(reduction='mean')

    def forward(self, pred, target):
        return self.fn(pred, target)
