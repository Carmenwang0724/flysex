"""Extract sequentially numbered PNG stills from a video."""

from pathlib import Path


def extract_stills(video_path, output_directory, stride=5, maximum_frames=None):
    if stride < 1 or (maximum_frames is not None and maximum_frames < 1):
        raise ValueError("stride and maximum_frames must be positive.")
    try:
        import av
    except ImportError as error:
        raise ValueError("Install video support with: pip install -e '.[video]'") from error
    video_path = Path(video_path).resolve()
    output_directory = Path(output_directory).resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError("Choose an empty still output directory.")
    written = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate is None or float(stream.average_rate) <= 0:
            raise ValueError("Video has no usable frame rate.")
        output_directory.mkdir(parents=True, exist_ok=True)
        for frame_index, frame in enumerate(container.decode(stream)):
            if maximum_frames is not None and frame_index >= maximum_frames:
                break
            if frame_index % stride:
                continue
            seconds = int(frame_index / float(stream.average_rate))
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            name = f"frame_{frame_index:06d}_{hours:02d}-{minutes:02d}-{seconds:02d}.png"
            frame.to_image().convert("L").save(output_directory / name)
            written += 1
            if written % 200 == 0:
                print(f"Wrote {written} stills", flush=True)
    print(f"Wrote {written} stills to {output_directory}")
    return written
