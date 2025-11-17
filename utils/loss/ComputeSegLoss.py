# Inspired by https://github.com/alibaba/esod/blob/main/utils/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_BCE(eps=0.1):
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05, reduction='mean'):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.reduction = reduction  # Add reduction attribute

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss



def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()



def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def quality_dice_loss(inputs, targets, weight=None, gamma: float = 2, eps: float = 1e-6):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    if weight is not None:
        weight = weight.flatten(1)
        inputs = inputs * weight
        targets = targets * weight
    numerator = 2 * (inputs - targets).abs().sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = (numerator + 1) / (denominator + 1) + eps
    return loss.mean()


def sigmoid_quality_focal_loss(inputs, targets, weight=None, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weight, reduction="none")
    loss = ce_loss * ((prob - targets).abs() ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()

class ComputeSegLoss(nn.Module):
    """
    ComputeSegLoss:
    - pred: логиты маски (B, 1, H, W)
    - true: GT маска (B, 1, H, W), float {0,1}
    - weight: опциональная карта весов (B, 1, H, W)
    """
    def __init__(
        self,
        pixel_loss_type="blur_focal",  # 'bce' | 'blur' | 'focal' | 'blur_focal' | 'qfocal'
        use_dice=True,
        use_quality_dice=False,        # если True, заменит обычный dice
        use_sigmoid_focal=False,
        use_quality_focal=True,        # если True, заменит обычный sigmoid_focal_loss
        w_pix=1.0,
        w_area=0.5,
        w_dist=0.2,
        gamma=2.0,
        alpha=0.25,
        blur_alpha=0.05,
        eps=1e-6,
    ):
        super().__init__()
        self.w_pix = w_pix
        self.w_area = w_area
        self.w_dist = w_dist
        self.use_dice = use_dice
        self.use_quality_dice = use_quality_dice
        self.use_sigmoid_focal = use_sigmoid_focal
        self.use_quality_focal = use_quality_focal
        self.gamma = gamma
        self.alpha = alpha
        self.blur_alpha = blur_alpha
        self.eps = eps
        # --- пиксельный лосс ---
        if pixel_loss_type == "bce":
            base = nn.BCEWithLogitsLoss(reduction='mean')
        elif pixel_loss_type == "blur":
            base = BCEBlurWithLogitsLoss(alpha=self.blur_alpha)
        elif pixel_loss_type == "focal":
            base = FocalLoss(
                nn.BCEWithLogitsLoss(reduction='mean') ,
                gamma=gamma,
                alpha=alpha,
            )
        elif pixel_loss_type == "blur_focal":
            base = FocalLoss(
                BCEBlurWithLogitsLoss(alpha=self.blur_alpha),
                gamma=gamma,
                alpha=alpha,
            )
        elif pixel_loss_type == "qfocal":
            base = QFocalLoss(
                nn.BCEWithLogitsLoss(reduction='mean'),
                gamma=gamma,
                alpha=alpha,
            )
        else:
            raise ValueError(f"Unknown pixel_loss_type: {pixel_loss_type}")

        self.pixel_loss = base

    def forward(self, pred, true, weight=None):
        """
        pred: (B, 1, H, W) logits
        true: (B, 1, H, W) {0,1}
        weight: (B, 1, H, W) or None
        """
        # pixel term
        lpixl = self.pixel_loss(pred, true)

        # area (Dice) term
        if self.use_quality_dice:
            larea = quality_dice_loss(pred, true, weight=weight, gamma=self.gamma)
            if torch.isnan(larea):
                print("Warning: NaN detected in quality_dice_loss")
            if torch.isinf(larea):
                print("Warning: Inf detected in quality_dice_loss")
        elif self.use_dice:
            larea = dice_loss(pred, true)
            if torch.isnan(larea):
                print("Warning: NaN detected in quality_dice_loss")
            if torch.isinf(larea):
                print("Warning: Inf detected in quality_dice_loss")
        else:
            larea = pred.new_zeros(1)
        # distance/focal term
        if self.use_quality_focal:
            ldist = sigmoid_quality_focal_loss(
                pred, true, weight=weight,
                alpha=self.alpha, gamma=self.gamma
            )
            if torch.isnan(ldist):
                print("Warning: NaN detected in quality_dice_loss")
            if torch.isinf(ldist):
                print("Warning: Inf detected in quality_dice_loss")
        elif self.use_sigmoid_focal:
            ldist = sigmoid_focal_loss(
                pred, true,
                alpha=self.alpha, gamma=self.gamma
            )
            if torch.isnan(ldist):
                print("Warning: NaN detected in quality_dice_loss")
            if torch.isinf(ldist):
                print("Warning: Inf detected in quality_dice_loss")
        else:
            ldist = pred.new_zeros(1)

        
        lpixl = torch.clamp(lpixl, min=self.eps, max=100.0)
        larea = torch.clamp(larea, min=self.eps, max=100.0)
        ldist = torch.clamp(ldist, min=self.eps, max=100.0)
        
        loss = self.w_pix * lpixl + self.w_area * larea + self.w_dist * ldist
        
        # loss_items = torch.stack([lpixl.detach(), larea.detach(), ldist.detach(), loss.detach()])
        return loss 
