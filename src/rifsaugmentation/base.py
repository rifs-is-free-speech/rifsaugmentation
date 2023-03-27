"""Base classes for the rifsaugmentation library."""

from abc import ABC, abstractmethod

import numpy as np


class BaseAugmentation(ABC):
    """Base class for all augmentation classes."""

    @abstractmethod
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply the augmentation to a file.

        Parameters
        ----------
        audio : numpy.ndarray
            The waveform to augment

        Returns
        -------
        numpy.ndarray
            The augmented waveform.
        """
        ...
