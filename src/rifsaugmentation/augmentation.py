"""This module contains the augmentation classes.

The augmentation classes are used to augment the data. Right now we support the
following augmentation classes:

- NoiseAugmentation
- RoomSimulationAugmentation
- ModifySpeedAugmentation

"""

import librosa
import random

from rifsaugmentation.base import BaseAugmentation
from rifsaugmentation.utils import load_wav_with_checks

from glob import glob
from os.path import join

import numpy as np
import pyroomacoustics as pra
import scipy.stats as stats


class NoiseAugmentation(BaseAugmentation):
    """Add noise to the audio. Does not take train/dev/test sets into account."""

    def __init__(self, noise_path: str, mu: float = 0.25, sd: float = 0.1):
        """Initialize the augmentation."""
        self.mu = mu
        self.sd = sd

        self.noise_path = noise_path

        # Load the noise files
        self.noise_audios = []
        for filename in glob(join(self.noise_path, "**/*.wav"), recursive=True):
            audio, _ = load_wav_with_checks(filename)
            self.noise_audios.append(audio)
        assert len(self.noise_audios) > 0, "No noise files found"

    def __call__(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Apply the augmentation to a file.

        Parameters
        ----------
        audio : numpy.ndarray
            The waveform to augment

        Returns
        -------
        numpy.ndarray
            The augmented waveform.
        """
        noise = random.choice(self.noise_audios)
        noise = np.roll(noise, np.random.randint(0, len(noise)))
        noise = np.resize(noise, audio_array.shape[0])
        factor = np.abs(np.random.normal(self.mu, self.sd))
        audio_array = audio_array + (noise * factor)
        return audio_array


class RoomSimulationAugmentation(BaseAugmentation):
    """
    Simulate the audio in a room.
    """

    def __init__(self, n: int = 10):
        """
        Initialize the augmentation.

        Parameters
        ----------
        n: int
            Number of rooms to sample from. Automatically generated.
        """
        assert n > 0, "n must be a positive integer."
        np.random.seed(42)

        self.rooms = [self._generate_room() for _ in range(n)]

    def __call__(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Apply the augmentation to a file.

        Parameters
        ----------
        audio : numpy.ndarray
            The waveform to augment

        Returns
        -------
        numpy.ndarray
            The augmented waveform.
        """

        while True:
            try:
                # Simulate the audio in a room
                room = self._generate_room()

                # place the source in the room
                room.add_source(
                    self._get_random_location(room), signal=audio_array, delay=0.5
                )

                # place a microphone in the room
                loc = self._get_random_location(room)

                room.add_microphone(loc)

                break
            except AttributeError:
                continue

        # Run the simulation
        room.simulate()

        audio_array = room.mic_array.signals[0]

        return audio_array

    @staticmethod
    def _generate_room() -> pra.ShoeBox:
        """
        Generate a room.
        This abstraction allows for more room types to be added in the future.

        Returns
        -------
        pra.ShoeBox
            The generated room.
        """
        # TODO: Add more room types
        return RoomSimulationAugmentation._generate_cuboid_room()

    @staticmethod
    def _generate_cuboid_room() -> pra.ShoeBox:
        """
        Generate a room with four walls.
        Minimum room size is 3m x 3m x 5m, maximum size is 10m x 10m x 5m.

        Returns
        -------
        pra.ShoeBox
            The generated room.
        """

        rt60 = RoomSimulationAugmentation._truncated_normal(0.5, 0.2, 0.3, 1)

        # Generate a random room
        length = RoomSimulationAugmentation._truncated_normal(6, 3, 3, 10)
        width = RoomSimulationAugmentation._truncated_normal(5, 2, 3, 10)
        height = (
            RoomSimulationAugmentation.h
        ) = RoomSimulationAugmentation._truncated_normal(2.4, 0.6, 2.2, 5)

        room_dim = np.array([length, width, height])

        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

        room = pra.ShoeBox(
            room_dim,
            fs=16000,
            materials=pra.Material(e_absorption),
            max_order=max_order,
            use_rand_ism=True,
            max_rand_disp=0.05,
        )

        return room

    @staticmethod
    def _get_random_location(room: pra.ShoeBox) -> np.ndarray:
        """
        Get a uniformly random location with the room

        Parameters
        ----------
        room: pra.ShoeBox
            The room to get the location from.

        Returns
        -------
        np.ndarray
            The location.
        """

        # TODO: Consider how to place the source in the room.
        # Currently it is uniformly distributed. Maybe use a Gaussian distribution?

        length = np.random.uniform(0, room.shoebox_dim[0])
        width = np.random.uniform(0, room.shoebox_dim[1])
        height = np.random.uniform(0, room.shoebox_dim[2])

        location = np.array([length, width, height])
        assert room.is_inside(location), "Location is not inside the room."

        return location

    @staticmethod
    def _truncated_normal(mean, sd, min, max):
        """
        Generate a truncated normal distribution.

        Parameters
        ----------
        mean: float
            The mean of the distribution.
        sd: float
            The standard deviation of the distribution.
        min: float
            The minimum value of the distribution.
        max: float
            The maximum value of the distribution.

        Returns
        -------
        float
            The generated value.
        """
        return stats.truncnorm(
            (min - mean) / sd, (max - mean) / sd, loc=mean, scale=sd
        ).rvs(1)[0]


class ModifySpeedAugmentation(BaseAugmentation):
    """
    Modify the speed of the audio.
    """

    def __init__(self, speed: float = 1.0):
        """
        Initialize the augmentation.

        Parameters
        ----------
        speed: float
            The speed to modify the audio with. 1.0 is the original speed. Default is 1.0.
            If rate > 1.0, then the audio is sped up. If rate < 1.0, then the audio is slowed down.
        """
        self.speed = speed

    def __call__(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Apply the augmentation to a file.

        Parameters
        ----------
        audio : numpy.ndarray
            The waveform to augment

        Returns
        -------
        numpy.ndarray
            The augmented waveform.
        """
        audio_array = librosa.effects.time_stretch(audio_array, rate=self.speed)
        return audio_array
