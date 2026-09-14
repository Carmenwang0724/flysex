"""Prepare body-axis crops and review images."""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# Define the shared crop and intensity settings
crop_size = 96
reference_pixels_per_mm = 15.7
field_of_view = 0.75
background_window = 121
intensity_floor = 0.35
intensity_range = 0.80


def read_still(path, width, height):
    """Decode a grayscale still and check its dimensions and empty bands."""
    # Define the empty-band detection settings
    minimum_brightness = 1
    empty_band_rows = 32

    # Decode the image at the expected resolution
    with Image.open(fp=path) as image:
        if image.size != (width, height):
            raise ValueError(
                f"{path.name}: expected {width} by {height} pixels, got {image.size}."
            )
        pixels = np.asarray(
            a=image.convert(mode="L"),
            dtype=np.float32,
        )

    # Count consecutive dark image rows
    row_brightness = pixels.mean(axis=1)
    dark_rows = (row_brightness < minimum_brightness).astype(dtype=np.int32)
    band_counts = np.convolve(
        a=dark_rows,
        v=np.ones(
            shape=empty_band_rows,
            dtype=np.int32,
        ),
        mode="valid",
    )
    if np.any(a=band_counts >= empty_band_rows):
        raise ValueError(
            f"{path.name}: contains an empty horizontal band; re-extract the still."
        )

    # Return the decoded pixel array
    return pixels


def flatten_illumination(image):
    """Divide the image intensity by its local background."""
    # Estimate the background illumination
    background = ndimage.uniform_filter(
        input=image,
        size=background_window,
    )

    # Return the background-normalized image
    return image / np.maximum(
        background,
        1e-6,
    )


def crop_along_body_axis(image, centre_x, centre_y, orientation, step):
    """Sample a normalized square crop along a fly's body axis."""
    # Define offsets from the crop centre
    offsets = (
        np.arange(
            crop_size,
            dtype=np.float32,
        )
        - (crop_size - 1) / 2
    ) * step

    # Rotate the sampling coordinates into the image plane
    unit_x = np.cos(-orientation)
    unit_y = np.sin(-orientation)
    sample_x = centre_x + offsets[None, :] * unit_x - offsets[:, None] * unit_y
    sample_y = centre_y + offsets[None, :] * unit_y + offsets[:, None] * unit_x

    # Interpolate the source pixels
    sampled = ndimage.map_coordinates(
        input=image,
        coordinates=[sample_y, sample_x],
        order=1,
        mode="nearest",
    )

    # Return the crop in the model's intensity range
    normalized = (sampled - intensity_floor) / intensity_range
    return np.clip(
        a=normalized,
        a_min=0,
        a_max=1,
    ).astype(dtype=np.float32)


def crop_flies(image, tracked, origin, scale_factor, pixels_per_mm):
    """Extract aligned crops and image positions for a tracked group."""
    # Normalize illumination and physical sampling scale
    flattened = flatten_illumination(image=image)
    step = pixels_per_mm / reference_pixels_per_mm * scale_factor * field_of_view

    # Map each tracked fly into the still
    crops = []
    positions = []
    for position_x, position_y, orientation in tracked:
        centre_x = (position_x + origin[0]) * scale_factor
        centre_y = (position_y + origin[1]) * scale_factor
        if not (0 <= centre_x < image.shape[1] and 0 <= centre_y < image.shape[0]):
            raise ValueError(
                "A tracked centre is outside the still; "
                "check well_origins and image geometry."
            )

        # Sample the aligned body crop
        crop = crop_along_body_axis(
            image=flattened,
            centre_x=centre_x,
            centre_y=centre_y,
            orientation=orientation,
            step=step,
        )
        crops.append(crop)
        positions.append((centre_x, centre_y))

    # Return the crops and their image centres
    return crops, positions


def save_overlay(still_path, positions, scores, output_path):
    """Draw score ranks and tracker positions onto a still."""
    # Define the overlay appearance
    radius = 40
    decision_threshold = 0.5
    male_colour = (230, 55, 55)
    female_colour = (40, 140, 240)

    # Open the image and order the scores
    with Image.open(fp=still_path) as source:
        image = source.convert(mode="RGB")
    drawing = ImageDraw.Draw(im=image)
    score_array = np.asarray(
        a=scores,
        dtype=np.float32,
    )
    order = np.argsort(
        a=-score_array,
        kind="stable",
    )

    # Draw a circle and caption for each fly
    for rank, index in enumerate(
        order,
        start=1,
    ):
        centre_x, centre_y = positions[index]
        colour = male_colour if scores[index] > decision_threshold else female_colour
        drawing.ellipse(
            xy=(
                centre_x - radius,
                centre_y - radius,
                centre_x + radius,
                centre_y + radius,
            ),
            outline=colour,
            width=5,
        )
        drawing.text(
            xy=(centre_x + radius + 2, centre_y),
            text=f"rank {rank} / fly {index + 1} / {scores[index]:.3f}",
            fill=colour,
        )

    # Save the annotated still
    image.save(fp=output_path)


def save_review_sheet(examples, output_path):
    """Arrange crop previews and score captions on one image."""
    # Define the review grid dimensions
    columns = 5
    cell_width = 200
    cell_height = 228
    tile_size = (192, 192)

    # Create the review canvas
    rows = (len(examples) + columns - 1) // columns
    image = Image.new(
        mode="RGB",
        size=(
            columns * cell_width,
            max(
                1,
                rows,
            )
            * cell_height,
        ),
        color="white",
    )
    drawing = ImageDraw.Draw(im=image)

    # Draw one crop and caption per grid cell
    for index, (crop, caption) in enumerate(examples):
        left = index % columns * cell_width
        top = index // columns * cell_height
        pixels = np.clip(
            a=crop * 255,
            a_min=0,
            a_max=255,
        ).astype(dtype=np.uint8)
        tile = Image.fromarray(obj=pixels).resize(size=tile_size)
        image.paste(
            im=tile,
            box=(left + 4, top + 4),
        )
        drawing.text(
            xy=(left + 4, top + 198),
            text=caption,
            fill="black",
        )

    # Save the completed review sheet
    image.save(fp=output_path)
