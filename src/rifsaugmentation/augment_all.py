"""Augment all audios in a directory. Main adapter for the rifs CLI

You may use this outside the CLI as well, but it is recommended to use the
augment_all function directly in the CLI.
"""

from glob import glob
from rifsaugmentation.augmentation import NoiseAugmentation, RoomSimulationAugmentation, ModifySpeedAugmentation
from rifsaugmentation.utils import load_wav_with_checks

import soundfile as sf
import os


# TODO: Extend this to support other augmentations as well.
def augment_all(
    source_path: str,
    target_path: str,
    with_room_simulation: bool = False,
    speed: float = 1.0,
    noise_path: str = None,
    recursive=False,
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

    Examples
    --------
    >>> !ls data
    clean  noise
    >>> !ls data/clean
    1.wav  2.wav 3.wav 4.wav 5.wav
    >>> augment_all("data/clean", "data/augmented_data", with_room_simulation=True, speed=1.1 noise_path="noise")
    >>> !ls augmented_data
    1.wav 2.wav 3.wav 4.wav 5.wav
    """
    # Find list of all files
    # TODO: Consider how to handle the case when the dataset is not segmented
    filenames = glob(f"{source_path}/*.wav", recursive=recursive)

    os.makedirs(target_path, exist_ok=True)

    # TODO: Get from environment variables
    kwargs = {
        "mu": 0.25,
        "sd": 0.1,
        "n": 100,
    }

    augments = []

    # Initialize augmentations
    if with_room_simulation:
        augments.append(RoomSimulationAugmentation(**kwargs))
    if noise_path:
        augments.append(NoiseAugmentation(noise_path, **kwargs))
    # TODO: Add other augmentations down the line

    augments.append(ModifySpeedAugmentation(speed))

    for filename in filenames:
        audio_array, sr = load_wav_with_checks(filename)

        # Augment audio
        for augmentation in augments:
            audio_array = augmentation(audio_array)

        # Save to target destination
        sf.write(
            os.path.join(target_path, os.path.basename(filename)),
            audio_array,
            sr,
        )
