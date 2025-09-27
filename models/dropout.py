import torch
import torch.nn as nn


class DropoutUNet(nn.Module):
    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

        self.dropout_layers = [m for m in self.unet.modules() 
                               if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))]
        if not self.dropout_layers:
            raise ValueError("UNet dropout model does not contain any dropout layers.")

    def forward(self, x, timesteps, y=None, *, num_samples: int = 1, reduce: bool = True):
        was_training = self.training
        self.train()  # Ensure dropout is active

        preds = []
        for _ in range(num_samples):
            preds.append(self.unet(x, timesteps, y))

        self.train(was_training)  # Restore original training mode
        stack = torch.stack(preds)
        if not reduce:
            return stack
        
        return stack.mean(dim=0)