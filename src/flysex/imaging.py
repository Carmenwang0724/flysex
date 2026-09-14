"""Prepare body-axis crops and review images."""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

CROP_SIZE = 96
REFERENCE_PPM = 15.7
FIELD_OF_VIEW = 0.75
BACKGROUND_WINDOW = 121
INTENSITY_FLOOR = 0.35
INTENSITY_RANGE = 0.80


def read_still(path, width, height):
    with Image.open(path) as image:
        if image.size != (width, height):
            raise ValueError(f"{path.name}: expected {width} by {height} pixels, got {image.size}.")
        pixels = np.asarray(image.convert("L"), dtype=np.float32)
    # Detect empty image bands after decoding the PNG
    row_brightness = pixels.mean(axis=1)
    if np.any(np.convolve((row_brightness < 1).astype(int), np.ones(32), mode="valid") >= 32):
        raise ValueError(f"{path.name}: contains an empty horizontal band; re-extract the still.")
    return pixels


def flatten_illumination(image):
    background = ndimage.uniform_filter(image, BACKGROUND_WINDOW)
    return image / np.maximum(background, 1e-6)


def crop_along_body_axis(image, centre_x, centre_y, orientation, step):
    offsets = (np.arange(CROP_SIZE, dtype=np.float32) - (CROP_SIZE - 1) / 2) * step
    unit_x = np.cos(-orientation)
    unit_y = np.sin(-orientation)
    sample_x = centre_x + offsets[None, :] * unit_x - offsets[:, None] * unit_y
    sample_y = centre_y + offsets[None, :] * unit_y + offsets[:, None] * unit_x
    sampled = ndimage.map_coordinates(image, [sample_y, sample_x], order=1, mode="nearest")
    return np.clip((sampled - INTENSITY_FLOOR) / INTENSITY_RANGE, 0, 1).astype(np.float32)


def crop_flies(image, tracked, origin, scale_factor, pixels_per_mm):
    flattened = flatten_illumination(image)
    step = pixels_per_mm / REFERENCE_PPM * scale_factor * FIELD_OF_VIEW
    crops = []
    positions = []
    for position_x, position_y, orientation in tracked:
        centre_x = (position_x + origin[0]) * scale_factor
        centre_y = (position_y + origin[1]) * scale_factor
        if not (0 <= centre_x < image.shape[1] and 0 <= centre_y < image.shape[0]):
            raise ValueError("A tracked centre is outside the still; check well_origins and image geometry.")
        crops.append(crop_along_body_axis(
            image=flattened,
            centre_x=centre_x,
            centre_y=centre_y,
            orientation=orientation,
            step=step,
        ))
        positions.append((centre_x, centre_y))
    return crops, positions


def save_overlay(still_path, positions, scores, output_path):
    with Image.open(still_path) as source:
        image = source.convert("RGB")
    drawing = ImageDraw.Draw(image)
    order = np.argsort(-np.asarray(scores), kind="stable")
    for rank, index in enumerate(order, start=1):
        centre_x, centre_y = positions[index]
        colour = (230, 55, 55) if scores[index] > 0.5 else (40, 140, 240)
        drawing.ellipse((centre_x - 40, centre_y - 40, centre_x + 40, centre_y + 40), outline=colour, width=5)
        drawing.text((centre_x + 42, centre_y), f"rank {rank} / fly {index + 1} / {scores[index]:.3f}", fill=colour)
    image.save(output_path)


def save_review_sheet(examples, output_path):
    columns = 5
    cell_width = 200
    cell_height = 228
    rows = (len(examples) + columns - 1) // columns
    image = Image.new("RGB", (columns * cell_width, max(1, rows) * cell_height), "white")
    drawing = ImageDraw.Draw(image)
    for index, (crop, caption) in enumerate(examples):
        left = index % columns * cell_width
        top = index // columns * cell_height
        tile = Image.fromarray(np.clip(crop * 255, 0, 255).astype(np.uint8))
        tile = tile.resize((192, 192))
        image.paste(tile, (left + 4, top + 4))
        drawing.text((left + 4, top + 198), caption, fill="black")
    image.save(output_path)
