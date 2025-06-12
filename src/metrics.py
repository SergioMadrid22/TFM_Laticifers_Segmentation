def compute_iou(preds, targets, threshold=0.5, eps=1e-7):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def compute_dice(preds, targets, threshold=0.5, eps=1e-7):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean().item()


def dice_loss(pred, target, eps=1e-7):
    dice = compute_soft_dice(pred, target, eps)
    return 1 - dice


def compute_soft_dice(pred, target, eps=1e-7):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + eps) / (union + eps)


def compute_tversky(preds, targets, alpha=0.5, beta=0.5, threshold=0.5, eps=1e-7):
    preds = (preds > threshold).float()
    targets = targets.float()

    tp = (preds * targets).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return tversky.mean().item()


def tversky_loss(pred, target, alpha=0.5, beta=0.5, eps=1e-7):
    tversky = compute_soft_tversky(pred, target, alpha, beta, eps)
    return 1 - tversky


def compute_soft_tversky(pred, target, alpha=0.5, beta=0.5, eps=1e-7):
    pred = pred.view(-1)
    target = target.view(-1)

    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    fp = (pred * (1 - target)).sum()

    return (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def focal_tversky_loss(pred, target, alpha=0.5, beta=0.5, gamma=0.75, eps=1e-7):
    """Focal Tversky Loss: penalizes harder-to-segment pixels more."""
    tversky = compute_soft_tversky(pred, target, alpha, beta, eps)
    return (1 - tversky) ** gamma