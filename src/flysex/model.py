"""Load the trained ensemble and score fly crops."""

import hashlib

import numpy as np
import torch
from torch import nn
from torchvision import models


def select_device(requested="auto"):
    """Select an available device for model inference."""
    # Validate the requested device
    if requested not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError("device must be auto, cpu, mps, or cuda.")

    # Select the first available accelerator
    if requested == "auto":
        if torch.cuda.is_available():
            # Return the selected compute device
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            # Return the selected compute device
            return torch.device("mps")
        # Return the selected compute device
        return torch.device("cpu")

    # Verify the explicitly selected accelerator
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable on this computer.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable on this computer.")

    # Return the selected device
    return torch.device(requested)


def build_model():
    """Build the single-channel ResNet architecture used by the checkpoints."""
    # Define the network layers
    network = models.resnet18(weights=None)
    network.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    network.fc = nn.Linear(
        in_features=512,
        out_features=1,
    )

    # Return the adapted network
    return network


def checkpoint_digest(path):
    """Calculate the SHA-256 checksum of one checkpoint."""
    # Read the checkpoint in fixed-size chunks
    digest = hashlib.sha256()
    with path.open(mode="rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)

    # Return the hexadecimal checksum
    return digest.hexdigest()


def load_models(model_directory, device):
    """Restore the checkpoint ensemble and record its checksums."""
    # Locate the saved weights
    checkpoints = sorted(model_directory.glob(pattern="flysex_seed*.pt"))
    if not checkpoints:
        raise ValueError(f"No flysex_seed*.pt weights found in {model_directory}.")

    # Restore each checkpoint on the selected device
    networks = []
    records = []
    for path in checkpoints:
        state = torch.load(
            f=path,
            map_location="cpu",
            weights_only=True,
        )
        network = build_model()
        network.load_state_dict(state_dict=state)
        network.to(device=device)
        network.eval()
        networks.append(network)
        records.append({"file": path.name, "sha256": checkpoint_digest(path=path)})

    # Return the ensemble and checkpoint records
    return networks, records


def augment_batch(images):
    """Flip fly crops and vary their brightness."""
    # Define the brightness range
    minimum_gain = 0.90
    gain_range = 0.20

    # Flip the batch along each image axis
    if torch.rand(1) < 0.5:
        images = torch.flip(
            input=images,
            dims=[2],
        )
    if torch.rand(1) < 0.5:
        images = torch.flip(
            input=images,
            dims=[3],
        )

    # Sample one brightness gain per crop
    gain = minimum_gain + gain_range * torch.rand(
        len(images),
        1,
        1,
        1,
        device=images.device,
    )

    # Return the augmented batch
    return images * gain


def network_probabilities(network, batch):
    """Convert one network's logits to CPU probability values."""
    # Apply the model and sigmoid transform
    logits = network(x=batch).squeeze(dim=1)
    probabilities = torch.sigmoid(input=logits)

    # Return the crop probabilities
    return probabilities.cpu().numpy()


def predict_probabilities(networks, crops, device):
    """Average model scores across the ensemble and image augmentations."""
    # Define the normalization and augmentation settings
    test_time_augmentations = 6
    intensity_centre = 0.5
    intensity_scale = 0.25

    # Validate the inference inputs
    if not networks or len(crops) == 0:
        raise ValueError("Inference requires at least one model and one crop.")

    # Normalize the crop batch
    crop_array = np.asarray(
        a=crops,
        dtype=np.float32,
    )
    batch = torch.tensor(
        data=crop_array,
        device=device,
    ).unsqueeze(dim=1)
    batch = (batch - intensity_centre) / intensity_scale

    # Score the clean and augmented batches
    predictions = []
    with torch.inference_mode():
        for network in networks:
            predictions.append(
                network_probabilities(
                    network=network,
                    batch=batch,
                )
            )
            for _ in range(test_time_augmentations - 1):
                augmented = augment_batch(images=batch)
                predictions.append(
                    network_probabilities(
                        network=network,
                        batch=augmented,
                    )
                )

    # Check the averaged scores
    scores = np.mean(
        a=predictions,
        axis=0,
    )
    if not np.isfinite(scores).all():
        raise ValueError("The model produced nonfinite scores.")

    # Return one score per crop
    return scores
