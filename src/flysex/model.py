"""Load the trained ensemble and score fly crops."""

import hashlib

import numpy as np
import torch
from torch import nn
from torchvision import models

TEST_TIME_AUGMENTATIONS = 6


def select_device(requested="auto"):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable on this computer.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable on this computer.")
    return torch.device(requested)


def build_model():
    network = models.resnet18(weights=None)
    network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    network.fc = nn.Linear(512, 1)
    return network


def load_models(model_directory, device):
    checkpoints = sorted(model_directory.glob("flysex_seed*.pt"))
    if not checkpoints:
        raise ValueError(f"No flysex_seed*.pt weights found in {model_directory}.")
    networks = []
    records = []
    for path in checkpoints:
        state = torch.load(path, map_location="cpu", weights_only=True)
        network = build_model()
        network.load_state_dict(state)
        network.to(device)
        network.eval()
        networks.append(network)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()
        records.append({"file": path.name, "sha256": digest})
    return networks, records


def augment_batch(images):
    if torch.rand(1) < 0.5:
        images = torch.flip(images, dims=[2])
    if torch.rand(1) < 0.5:
        images = torch.flip(images, dims=[3])
    gain = 0.90 + 0.20 * torch.rand(len(images), 1, 1, 1, device=images.device)
    return images * gain


def predict_probabilities(networks, crops, device):
    batch = torch.tensor(np.stack(crops), device=device).unsqueeze(1)
    batch = (batch - 0.5) / 0.25
    predictions = []
    with torch.inference_mode():
        for network in networks:
            predictions.append(torch.sigmoid(network(batch).squeeze(1)).cpu().numpy())
            for _ in range(TEST_TIME_AUGMENTATIONS - 1):
                predictions.append(torch.sigmoid(network(augment_batch(batch)).squeeze(1)).cpu().numpy())
    scores = np.mean(predictions, axis=0)
    if not np.isfinite(scores).all():
        raise ValueError("The model produced nonfinite scores.")
    return scores
