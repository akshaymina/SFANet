import torch.nn.functional as F


class SFANetLoss:
    """Paper loss: L_total = 0.1 * L_rec + 1.0 * L_cls."""

    def __init__(self, lambda_rec=0.1, lambda_cls=1.0):
        self.lambda_rec = lambda_rec
        self.lambda_cls = lambda_cls

    def __call__(self, logits, target, original, reconstruction):
        cls = F.cross_entropy(logits, target)
        rec = F.mse_loss(reconstruction, original)
        total = self.lambda_rec * rec + self.lambda_cls * cls
        return total, rec, cls
