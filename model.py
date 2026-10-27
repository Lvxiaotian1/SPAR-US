"""Implicit intensity field, ultrasound probe geometry and checkpoint I/O."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from geometry import axis_angle_to_matrix


class DenseLayer(nn.Linear):
    """Fully connected layer with Xavier-uniform weights and zero bias."""

    def __init__(self, in_features, out_features, activation="relu"):
        self.activation = activation
        super().__init__(in_features, out_features)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(self.activation))
        nn.init.zeros_(self.bias)


class Field(nn.Module):
    """Maps a three-dimensional point to a scalar ultrasound intensity."""

    def __init__(self):
        super().__init__()
        dimensions = [63, 128, 128, 128, 128, 191, 128, 128]
        self.base_layers = nn.ModuleList()
        for size in dimensions:
            self.base_layers.append(nn.Sequential(DenseLayer(size, 128), nn.ReLU(inplace=True)))
        self.sigma_layers = nn.Sequential(DenseLayer(128, 1, "linear"))

    def forward(self, points):
        points = points.reshape(-1, 3)
        encoded = [points]
        for frequency in (2.0 ** torch.linspace(0, 9, 10)).tolist():
            encoded.extend([torch.sin(points * frequency), torch.cos(points * frequency)])
        encoded = torch.cat(encoded, dim=-1)
        hidden = encoded
        for index, layer in enumerate(self.base_layers):
            if index == 5:
                hidden = torch.cat((encoded, hidden), dim=-1)
            hidden = layer(hidden)
        return self.sigma_layers(hidden)


class UltrasoundProbe(nn.Module):
    """Per-frame rotation, translation and image geometry."""

    def __init__(self, phi, translation, height, width, sw, sh):
        super().__init__()
        self.phi = nn.Parameter(phi.clone())
        self.t = nn.Parameter(translation.clone())
        self.height, self.width = int(height), int(width)
        self.sw, self.sh = float(sw), float(sh)

    def forward(self, index):
        rotation = axis_angle_to_matrix(self.phi[index])
        return torch.cat((rotation, self.t[index].unsqueeze(-1)), dim=-1)


def render(field, probe, index, config):
    """Sample a frame along its scanlines and return the rendered intensities."""
    pose = probe(index)
    rotation, translation = pose[:, :3], pose[:, 3]
    width, height = probe.width, probe.height
    x = torch.arange(-width / 2, width / 2, dtype=torch.float32, device=pose.device) * probe.sw
    origins = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=1)
    origins = origins @ rotation.T + translation
    direction = torch.tensor([0.0, 1.0, 0.0], device=pose.device) @ rotation.T
    directions = direction.repeat(width, 1)
    fractions = torch.linspace(0.0, 1.0, height, device=pose.device)
    near = torch.full_like(directions[:, :1], config["near"])
    far = torch.full_like(directions[:, :1], config["far"])
    depths = near * (1.0 - fractions) + far * fractions
    points = origins.unsqueeze(1) + directions.unsqueeze(1) * depths.unsqueeze(-1)
    points = points.permute(1, 0, 2)
    return field(points).reshape(height, width)


def load_images(folder, device):
    """Load grayscale frames named 0.png through N-1.png."""
    folder = Path(folder)
    paths = sorted(folder.glob("*.png"), key=lambda path: int(path.stem))
    if not paths:
        raise ValueError(f"No PNG frames found in {folder}.")
    if [int(path.stem) for path in paths] != list(range(len(paths))):
        raise ValueError("Frame files must be numbered 0.png through N-1.png without gaps.")
    images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths]
    if any(image is None for image in images):
        raise ValueError("An input image could not be read.")
    if len({image.shape for image in images}) != 1:
        raise ValueError("All frames must share the same dimensions.")
    return torch.as_tensor(np.stack(images), dtype=torch.float32, device=device) / 255.0


def save_checkpoint(path, field, probe):
    """Write the fitted field and the estimated frame geometry."""
    torch.save(
        {
            "field": {name: value.detach().cpu() for name, value in field.state_dict().items()},
            "geometry": {
                "phi": probe.phi.detach().cpu(),
                "t": probe.t.detach().cpu(),
                "rotation_format": "axis-angle",
                "height": probe.height,
                "width": probe.width,
                "sw": probe.sw,
                "sh": probe.sh,
                "n_frames": int(probe.phi.shape[0]),
            },
        },
        path,
    )


def load_checkpoint(path, device):
    """Restore a field and its estimated frame geometry."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    field = Field()
    field.load_state_dict(checkpoint["field"], strict=True)
    geometry = checkpoint["geometry"]
    if geometry.get("rotation_format") != "axis-angle":
        raise ValueError("Unexpected rotation format in checkpoint.")
    probe = UltrasoundProbe(geometry["phi"], geometry["t"], geometry["height"], geometry["width"], geometry["sw"], geometry["sh"])
    return field.to(device).eval(), probe.to(device).eval()
