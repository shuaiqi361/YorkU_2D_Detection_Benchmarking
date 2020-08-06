import torch
import torch.nn as nn
import torch.nn.functional as F
from .iou_utils import bbox_overlaps_iou, bbox_overlaps_ciou, bbox_overlaps_diou, bbox_overlaps_giou, \
    decode
from torch.autograd import Variable


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha, config):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.device = config.device

    def forward(self, out, target):
        n_class = out.shape[1]  # excluded background class when using Sigmoid Focal Loss, 80 for COCO
        class_ids = torch.arange(
            1, n_class + 1, dtype=target.dtype, device=target.device
        ).unsqueeze(0)

        t = target.unsqueeze(1)
        p = torch.sigmoid(out).clamp(min=1e-5, max=1-1e-5)

        gamma = self.gamma
        alpha = self.alpha

        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)

        # t >= 0 for excluding anchors whose IOU in the range of (0.4, 0.5), labeled as -1
        loss = -(t == class_ids).float() * alpha * term1 - ((t != class_ids) * (t >= 0)).float() * (1 - alpha) * term2

        return loss.sum()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.05, dim=-1, reduce=True):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.reduce = reduce

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        if self.reduce:
            return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        else:
            return torch.sum(-true_dist * pred, dim=self.dim)


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion, it is class priors, can be learned
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        print('Initialize Focal Loss: gamma={}, alpha={}'.format(self.gamma, self.alpha))

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class IouLoss(nn.Module):

    def __init__(self, pred_mode='Corner', reduce='mean', variances=None, losstype='Diou'):
        super(IouLoss, self).__init__()
        self.reduce = reduce
        self.pred_mode = pred_mode  # predicted locs should be top-left and bottom-right corner conventions
        self.variances = variances
        self.loss = losstype

    def forward(self, loc_p, loc_t, prior_data=None, weights=None):
        num = loc_p.shape[0]

        if self.pred_mode == 'Center':
            assert prior_data is not None
            decoded_boxes = decode(loc_p, prior_data, self.variances)
        else:
            decoded_boxes = loc_p
        if self.loss == 'Iou':
            loss = 1.0 - bbox_overlaps_iou(decoded_boxes, loc_t)
        else:
            if self.loss == 'Giou':
                loss = 1.0 - bbox_overlaps_giou(decoded_boxes, loc_t)
            else:
                if self.loss == 'Diou':
                    loss = 1.0 - bbox_overlaps_diou(decoded_boxes, loc_t)
                else:
                    loss = 1.0 - bbox_overlaps_ciou(decoded_boxes, loc_t)

        if weights is not None and weights.sum() > 1e-6:
            return (loss * weights).sum() / weights.sum()
        else:
            if self.reduce == 'mean':
                loss = loss.sum() / num
            else:
                loss = loss.sum()

            return loss


class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss"""

    def __init__(self, beta=1.0 / 9.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred, target, weights=None):
        num = pred.size(0)

        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        l1_loss = torch.where(x >= self.beta, l1, l2)

        if weights is not None and weights.sum() > 1e-6:
            assert pred.size(0) == target.size(0) == weights.size(0)
            return (l1_loss * weights).sum() / weights.sum()
        else:
            if self.reduction == 'mean':
                return l1_loss.sum() / num
            else:
                return l1_loss.sum()
