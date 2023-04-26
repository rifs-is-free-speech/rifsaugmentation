"""
Rifs Augmentation Library
=========================
This library contains functions to augment waveform files.

We provide the following functionality:

- Augmenting with added noise from wav files.
- Augmenting with room characteristics and microphone placements.
- Augmenting with playback speed of the recording.

All of the augmentations can be applied over a provided list of files using the
rifs CLI. See the documentation of rifs for more information.
"""

__version__ = "0.0.2"

from rifsaugmentation.augment_all import augment_all

__all__ = ["augment_all"]
