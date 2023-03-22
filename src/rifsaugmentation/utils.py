"""
This module contains utility functions for the rifsaugmentation package.
"""

import librosa
import numpy as np


def load_wav_with_checks(filename: str) -> np.ndarray:
    """
    Load a wav file. Assert that the sample rate is equals to sr, that it is mono and is not empty.

    Parameters
    ----------
    filename : str
        The path to the wav file. Can be relative or absolute.
    sr : int, optional
        The sample rate to load the wav file at, by default 16_000
    mono : bool, optional
        Whether to load the wav file as mono, by default True

    Returns
    -------
    np.ndarray
        The loaded wav file.
    int
        The sample rate of the wav file.
    """
    audio, sample_rate = librosa.load(filename, sr=16_000, mono=True)
    assert sample_rate == 16_000, f"Sample rate must be 16_000, not {sample_rate}"
    assert len(audio.shape) == 1, "Audio file must be mono"
    assert audio.shape[0] > 0, "Audio file is empty"
    return audio, sample_rate
