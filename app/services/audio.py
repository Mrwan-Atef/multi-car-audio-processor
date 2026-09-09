from __future__ import annotations

import subprocess
from pathlib import Path


def convert_to_pcm_wav(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
) -> Path:
    """Decode arbitrary audio supported by FFmpeg into mono 16 kHz PCM WAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel',
        'error',
        '-y',
        '-i',
        str(input_path),
        '-vn',
        '-ac',
        '1',
        '-ar',
        str(sample_rate),
        '-c:a',
        'pcm_s16le',
        str(output_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            'ffmpeg is not installed in the container.'
        ) from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or '').strip()
        raise RuntimeError(
            f'Audio decoding failed: {message or "unknown FFmpeg error"}'
        ) from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError('FFmpeg produced no audio output.')

    return output_path
