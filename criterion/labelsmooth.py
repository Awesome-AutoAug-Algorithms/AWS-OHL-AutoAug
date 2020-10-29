import torch


class LabelSmoothCELoss(torch.nn.Module):
    def __init__(self, smooth_ratio, num_classes):
        super(LabelSmoothCELoss, self).__init__()
        self.smooth_ratio = smooth_ratio
        self.v = self.smooth_ratio / num_classes

        self.logsoft = torch.nn.LogSoftmax(dim=1)

    def forward(self, inp, label):
        one_hot = torch.zeros_like(inp)
        one_hot.fill_(self.v)
        y = label.to(torch.long).view(-1, 1)
        one_hot.scatter_(1, y, 1-self.smooth_ratio+self.v)

        loss = - torch.sum(self.logsoft(inp) * (one_hot.detach())) / inp.size(0)
        return loss
