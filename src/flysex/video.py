"""Extract sequentially numbered PNG stills from a video."""

from pathlib import Path


def validate_extraction(video_path, output_directory, stride, maximum_frames):
    """Check the video source, output directory, and sampling limits."""
    # Validate the sampling interval and frame limit
    if type(stride) is not int or stride < 1:
        raise ValueError("stride must be a positive integer.")
    if maximum_frames is not None:
        if type(maximum_frames) is not int or maximum_frames < 1:
            raise ValueError("maximum_frames must be a positive integer.")

    # Validate the source file and output folder
    if not video_path.is_file():
        raise ValueError(f"Video file does not exist: {video_path}")
    if output_directory.exists():
        if not output_directory.is_dir() or any(output_directory.iterdir()):
            raise ValueError("Choose an empty still output directory.")


def still_filename(frame_index, frame_rate):
    """Name a still using its decoded frame number and elapsed video time."""
    # Convert the frame position to elapsed time
    elapsed_seconds = int(frame_index / frame_rate)
    hours, remainder = divmod(
        elapsed_seconds,
        3600,
    )
    minutes, seconds = divmod(
        remainder,
        60,
    )

    # Return the numbered PNG filename
    return f"frame_{frame_index:06d}_{hours:02d}-{minutes:02d}-{seconds:02d}.png"


def write_sampled_frames(container, output_directory, stride, maximum_frames):
    """Decode a video stream and write the sampled grayscale frames."""
    # Define the progress interval
    progress_interval = 200

    # Require a video stream with a usable frame rate
    if not container.streams.video:
        raise ValueError("The input file contains no video stream.")
    stream = container.streams.video[0]
    if stream.average_rate is None or float(stream.average_rate) <= 0:
        raise ValueError("Video has no usable frame rate.")

    # Decode frames in video order
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    written = 0
    for frame_index, frame in enumerate(container.decode(stream)):
        if maximum_frames is not None and frame_index >= maximum_frames:
            break
        if frame_index % stride:
            continue

        # Save each sampled frame without renumbering it
        name = still_filename(
            frame_index=frame_index,
            frame_rate=float(stream.average_rate),
        )
        frame.to_image().convert(mode="L").save(fp=output_directory / name)
        written += 1
        if written % progress_interval == 0:
            print(
                f"Wrote {written} stills",
                flush=True,
            )

    # Return the number of saved frames
    return written


def extract_stills(video_path, output_directory, stride=5, maximum_frames=None):
    """Extract numbered PNG stills from a video at the requested interval."""
    # Load the optional video decoder
    try:
        import av
    except ImportError as error:
        raise ValueError(
            "Install video support with: pip install -e '.[video]'"
        ) from error

    # Resolve and validate the extraction inputs
    video_path = Path(video_path).resolve()
    output_directory = Path(output_directory).resolve()
    validate_extraction(
        video_path=video_path,
        output_directory=output_directory,
        stride=stride,
        maximum_frames=maximum_frames,
    )

    # Open the video and write the sampled frames
    with av.open(file=str(video_path)) as container:
        written = write_sampled_frames(
            container=container,
            output_directory=output_directory,
            stride=stride,
            maximum_frames=maximum_frames,
        )
    print(f"Wrote {written} stills to {output_directory}")

    # Return the number of saved frames
    return written
