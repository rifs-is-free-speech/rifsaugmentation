"""Augment all audios in a directory. Main adapter for the rifs CLI

You may use this outside the CLI as well, but it is recommended to use the
augment_all function directly in the CLI.
"""

import os
import soundfile as sf

from glob import glob
from rifsaugmentation.augmentation import (
    NoiseAugmentation,
    RoomSimulationAugmentation,
    ModifySpeedAugmentation,
)
from rifsaugmentation.utils import load_wav_with_checks


# TODO: Extend this to support other augmentations as well.
def augment_all(
    source_path: str,
    target_path: str,
    with_room_simulation: bool = False,
    speed: float = 1.0,
    noise_path: str = None,
    recursive=False,
    move_other_files=True,
    verbose=False,
    quiet=False,
):
    """
    Main function. Augment all files in a given folder and deliver to target folder.

    Parameters
    ----------
    source_path: str
        Path to the source data
    target_path: str
        Path on where to deliver data.
    with_room_simulation: bool
        Whether to perform room simulation.
    speed: float
        Speed factor to apply to the audio. 1.0 is normal speed. Optional. Default: 1.0
    noise_path: str
        Path to the noise data. Optional, no noise augmentation is performed.
    recursive: bool
        Whether to recursively search for files in the data_path.
    move_other_files: bool
        Whether to move other files than wav files to the target folder. Default: True
    verbose: bool
        Whether to print verbose output. Default: False
    quiet: bool
        Whether to print no output. Default: False

    Examples
    --------
    >>> !ls data # doctest: +SKIP
    clean  noise
    >>> !ls data/clean # doctest: +SKIP
    1.wav  2.wav 3.wav 4.wav 5.wav
    >>> augment_all("data/clean", "data/augmented_data", with_room_simulation=True, speed=1.1 noise_path="noise") # doctest: +SKIP # noqa: E501
    >>> !ls augmented_data # doctest: +SKIP
    1.wav 2.wav 3.wav 4.wav 5.wav
    """
    filenames = glob(f"{source_path}/*.wav", recursive=recursive)

    os.makedirs(target_path, exist_ok=True)

    kwargs = {
        "mu": 0.25,
        "sd": 0.1,
        "n": 100,
    }

    augments = []

    if with_room_simulation:
        if verbose and not quiet:
            print("Initializing room simulation augmentation")
        augments.append(RoomSimulationAugmentation(**kwargs))
    if noise_path:
        if verbose and not quiet:
            print("Initializing noise augmentation")
        augments.append(NoiseAugmentation(noise_path, **kwargs))

    if speed != 1.0:
        if verbose and not quiet:
            print(
                f"Initializing speed modification augmentation with speed factor {speed}"
            )
        augments.append(ModifySpeedAugmentation(speed))

    for filename in filenames:
        audio_array, sr = load_wav_with_checks(filename)

        for augmentation in augments:
            audio_array = augmentation(audio_array)

        target = os.path.join(target_path, os.path.relpath(filename, source_path))

        sf.write(
            target,
            audio_array,
            sr,
        )

    if move_other_files:
        other_files = list(
            set(glob(f"{source_path}/**", recursive=recursive)) - set(filenames)
        )
        for filename in other_files:
            target = os.path.join(target_path, os.path.relpath(filename, source_path))
            if verbose and not quiet:
                print(f"Moving {filename} to {target}")
            os.copy(filename, target)
