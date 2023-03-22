"""Augment all audios in a directory. Main adapter for the CLI"""

from glob import glob
from rifsaugmentation.augmentation import NoiseAugmentation
from rifsaugmentation.utils import load_wav_with_checks

import soundfile as sf
import os

# TODO: Extend this to support other augmentations as well.
def augment_all(
    source_path: str, target_path: str, noise_path: str = None, recursive=False
):
    """
    Main function. Augment all files in a given folder and deliver to target folder.

    Parameters
    ----------
    source_path: str
        Path to the source data
    target_path: str
        Path on where to deliver data.
    noise_path: str
        Path to the noise data. Optional, no noise augmentation is performed.
    recursive: bool
        Whether to recursively search for files in the data_path.
    """
    # Find list of all files
    # TODO: Consider how to handle the case when the dataset is not segmented
    filenames = glob(f"{source_path}/*.wav", recursive=recursive)

    os.makedirs(target_path, exist_ok=True)

    kwargs = {
        "mu": 0.25,
        "sd": 0.1,
    }

    augments = []

    # Initialize augmentations
    if noise_path:
        augments.append(NoiseAugmentation(noise_path, **kwargs))
    # TODO: Add other augmentations down the line
    # if with_room_simulation:
    #    augments.append(RoomSimulationAugmentation())
    # if with_voice_conversion:
    #    augments.append(VoiceConversionAugmentation()

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
