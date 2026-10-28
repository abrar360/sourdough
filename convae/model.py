"""A small convolutional autoencoder: one-hot class map -> low-dim bottleneck -> class logits."""
import torch.nn as nn

NUM_CLASSES = 5


class ConvAE(nn.Module):
    """Stride-2 conv encoder to a `latent`-channel bottleneck, mirrored decoder to class logits."""

    def __init__(self, widths, latent, num_classes=NUM_CLASSES):
        super().__init__()
        enc, ch = [], num_classes
        for w in widths:
            enc += [nn.Conv2d(ch, w, 3, stride=2, padding=1), nn.GroupNorm(8, w), nn.SiLU()]
            ch = w
        self.encoder = nn.Sequential(*enc, nn.Conv2d(ch, latent, 1), nn.Tanh())
        dec, ch = [], latent
        for w in reversed(widths):
            dec += [nn.Upsample(scale_factor=2), nn.Conv2d(ch, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU()]
            ch = w
        self.decoder = nn.Sequential(*dec, nn.Conv2d(ch, num_classes, 3, padding=1))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def build_model(cfg):
    return ConvAE(cfg.widths, cfg.latent)


def model_info(model):
    return {"enc_params": sum(p.numel() for p in model.encoder.parameters()),
            "dec_params": sum(p.numel() for p in model.decoder.parameters())}
