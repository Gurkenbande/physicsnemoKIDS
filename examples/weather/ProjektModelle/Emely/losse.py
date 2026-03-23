import torch
import torch.nn as nn

# Custom loss functions 


class QuantizedLoss(nn.Module):
    """
    Quantized Loss that computes MSE loss separately for different value ranges
    """
    def __init__(self, n_bins=10):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.n_bins = n_bins

    def forward(self, pred, target, bins=None):
        n_channels = target.shape[1]
        loss_quantized = 0

        for c in range(n_channels):
            boundaries = torch.linspace(
                target[:, c].min(),
                target[:, c].max(),
                self.n_bins,
                device=target.device
            )
            bins_c = torch.bucketize(target[:, c], boundaries).int()

            for b in torch.unique(bins_c):
                mask_b = (bins_c == b)
                loss_quantized += self.mse_loss(pred[mask_b, c], target[mask_b, c])

        return loss_quantized / n_channels 


class MSEGradientLoss(nn.Module):
    """
    MSE Loss combined with gradient loss 
    """
    def __init__(self, fine_shape, alpha=0.5):
        super().__init__()
        self.H, self.W = fine_shape
        self.alpha = alpha 
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        N = pred.shape[0] // (self.H * self.W)
        p = pred[:N*self.H*self.W].reshape(N, self.H, self.W, -1)
        t = target[:N*self.H*self.W].reshape(N, self.H, self.W, -1)

        mse = self.mse(p, t)
        grad_p_x = p[:, :, 1:, :] - p[:, :, :-1, :]  # Horizontal gradients
        grad_t_x = t[:, :, 1:, :] - t[:, :, :-1, :]
        grad_p_y = p[:, 1:, :, :] - p[:, :-1, :, :]  # Vertical gradients
        grad_t_y = t[:, 1:, :, :] - t[:, :-1, :, :]

        # MSE on gradients
        grad_loss = self.mse(grad_p_x, grad_t_x) + self.mse(grad_p_y, grad_t_y)
        return mse + self.alpha * grad_loss
