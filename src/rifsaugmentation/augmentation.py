"""This module contains the augmentation classes."""

from rifsaugmentation.base import BaseAugmentation

from rifsaugmentation.utils import load_wav_with_checks
from glob import glob
from os.path import join

import numpy as np
import pyroomacoustics as pra


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

        # Add noise
        noise = np.random.choice(self.noise_audios)
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

        # Simulate the audio in a room
        room = np.random.choice(self.rooms)
        # place the source in the room
        # TODO: Consider how to place the source in the room. Currently it is uniformly distributed. Maybe use a Gaussian distribution?

        room.add_source(self._get_random_location(room), signal=audio_array, delay=0.5)

        # place a microphone in the room
        room.add_microphone(self._get_random_location(room))

        # Run the simulation
        room.simulate()

        audio_array = room.mic_array.signals[0]

        return audio_array


    @staticmethod
    def _generate_room() -> pra.ShoeBox:
        """
        Generate a room.
        Minimum room size is 3m x 3m x 3m, maximum size is 10m x 10m x 10m.

        Returns
        -------
        pra.ShoeBox
            The generated room.
        """

        rt60 = 0.5

        # Generate a random room
        l = np.random.uniform(3, 10)
        w = np.random.uniform(3, 10)
        h = np.abs(np.random.normal(3, 0.6)) # 2.4m is the average height of a room
        room_dim = np.array([l, w, h])

        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

        room = pra.ShoeBox(
            room_dim,
            fs=16000,
            materials=pra.Material(e_absorption),
            max_order=max_order,
            use_rand_ism=True,
            max_rand_disp=0.05
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
        l = np.random.uniform(0, room.shoebox_dim[0])
        w = np.random.uniform(0, room.shoebox_dim[1])
        h = np.random.uniform(0, room.shoebox_dim[2])

        location = np.array([l, w, h])
        assert room.is_inside(location), "Location is not inside the room."

        return location



