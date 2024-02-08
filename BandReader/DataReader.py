from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class DataReader(ABC):
    def __init__(self, dfolder: str, prefix: str, **kwargs):
        self.dfolder = dfolder
        self.prefix = prefix

    @abstractmethod
    def bands_available(self) -> bool:
        pass

    @abstractmethod
    def dos_available(self) -> bool:
        pass

    @abstractmethod
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_band_fatness(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_efermi(self) -> float:
        pass

    @abstractmethod
    def get_kcoords(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_symcoords(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_dos(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_dos_e(self) -> np.ndarray:
        pass