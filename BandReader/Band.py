from typing import List, Type, Dict
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import os
import re

from . import HA2EV
from .DataReader import DataReader
from .AmsReader import AMSReader

class BandData:
    def __init__(self, dfolder: str, prefix: str, reader: Type[DataReader] = AMSReader, reader_kwargs: Dict = None):

        if reader_kwargs is None:
            reader_kwargs = {}

        if not issubclass(reader, DataReader):
            raise TypeError("reader must be a subclass of DataReader")

        self.dfolder = dfolder
        self.prefix = prefix

        self.reader = reader(self.dfolder, self.prefix, **reader_kwargs)

        self.bands_available = self.reader.bands_available()
        if self.bands_available:
            self.bands, self.bands2 = self.reader.get_bands()
            self.fatness, self.fatness2 = self.reader.get_band_fatness()
            self.band_k = self.reader.get_kcoords()
            self.symcoords = self.reader.get_symcoords()
            self._sym_labels = None
            self.is_spinp = self.bands2 is not None
        else:
            warnings.warn(f'Band data not available for prefix "{self.prefix}" in {self.dfolder}')

        self.dos_available = self.reader.dos_available()
        if self.dos_available:
            self.dos, self.dos2 = self.reader.get_dos()
            self.dos_e = self.reader.get_dos_e()
            self.is_spinp = self.dos2 is not None
        else:
            warnings.warn(f'DOS data not available for prefix "{self.prefix}" in {self.dfolder}')

        if not self.bands_available and not self.dos_available:
            raise ValueError('Could not find neither bands nor DOS data.')

        try:
            self.e_fermi = self.reader.get_efermi()
        except:
            warnings.warn("Could not get Fermi energy. Proceeding with E_fermi = 0.")
            self.e_fermi = 0.0

    def _check_bands(self):
        if not self.bands_available:
            raise ValueError("No band data available.")

    def _check_dos(self):
        if not self.bands_available:
            raise ValueError("No DOS data available.")

    @property
    def shifted_bands(self):
        self._check_bands()
        b1 = self.bands - self.e_fermi
        b2 = self.bands2 - self.e_fermi if self.is_spinp else None
        return b1, b2

    @property
    def shifted_dos_e(self):
        self._check_dos()
        return self.dos_e - self.e_fermi

    @property
    def sym_labels(self):
        self._check_bands()
        if self._sym_labels is None:
            return [f'A_{i}' for i in range(len(self.symcoords))]
        else:
            return self._sym_labels

    @sym_labels.setter
    def sym_labels(self, sym_labels: List[str]):
        self._check_bands()
        if len(sym_labels) != len(self.symcoords):
            raise ValueError(f"Number of sym. labels ({len(sym_labels)} does not correspond to number of sym. points ({len(self.symcoords)})")
        self._sym_labels = list(sym_labels)

    def assign_sym_labels(self, sym_labels: List[str]):
        self.sym_labels = sym_labels

    def get_bandgap_info(self, fermi_offset=0):
        self._check_bands()
        iu = find_bg_info(self.bands, self.band_k, self.e_fermi)
        if self.is_spinp:
            id = find_bg_info(self.bands2, self.band_k, self.e_fermi)
            iud = summarize_bg_info(iu, id)

        else:
            id = iud = None

        return iu, id, iud


def find_bg_info(bands, x, e_fermi, fermi_offset=0):
    mins, maxes, xmins, xmaxes = [], [], [], []
    for band in bands:
        i_min = np.argmin(band)
        i_max = np.argmax(band)
        mins.append(band[i_min])
        maxes.append(band[i_max])
        xmins.append(x[i_min])
        xmaxes.append(x[i_max])
    i_max_below = np.searchsorted(maxes, e_fermi+fermi_offset) - 1
    i_min_above = np.searchsorted(mins, e_fermi+fermi_offset)
    if i_min_above != i_max_below + 1:
        metallic_bands = [i for i in range(i_max_below+1, i_min_above)]
        is_metallic = True
    else:
        metallic_bands = []
        is_metallic = False
    max_below = maxes[i_max_below]
    min_above = mins[i_min_above]
    x_max_below = xmaxes[i_max_below]
    x_min_above = xmins[i_min_above]

    res = {
        'i_vb': i_max_below,
        'i_cb': i_min_above,
        'vbm': max_below,
        'cbm': min_above,
        'k_vbm': x_max_below,
        'k_cbm': x_min_above,
        'vb': bands[i_max_below],
        'cb': bands[i_min_above],
        'bandgap': 0 if is_metallic else min_above - max_below,
        'metallic_bands': metallic_bands,
        'is_metallic': is_metallic
    }
    return res


def summarize_bg_info(info_spin_up, info_spin_dwn):

    su, sd = info_spin_up, info_spin_dwn
    s = [su, sd]

    if su['vbm'] >= sd['vbm']:
        vbm = 0
    elif su['vbm'] < sd['vbm']:
        vbm = 1

    if su['cbm'] <= sd['cbm']:
        cbm = 0
    elif su['cbm'] > sd['cbm']:
        cbm = 1


    if not su['is_metallic'] and not sd['is_metallic']:
        is_metallic = False
        metallic_type = ''
    else:
        is_metallic = True
        if su['is_metallic'] and not sd['is_metallic']:
            metallic_type = 'u'
        elif not su['is_metallic'] and sd['is_metallic']:
            metallic_type = 'd'
        else:
            metallic_type = 'ud'

    res = {
        'i_vb': s[vbm]['i_vb'],
        'i_cb': s[cbm]['i_cb'],
        'vbm': s[vbm]['vbm'],
        'cbm': s[cbm]['cbm'],
        'k_vbm': s[vbm]['k_vbm'],
        'k_cbm': s[cbm]['k_cbm'],
        'vb': s[vbm]['vb'],
        'cb': s[cbm]['cb'],
        'bandgap': 0 if is_metallic else s[cbm]['cbm'] - s[vbm]['vbm'],
        'bg_type': '' if is_metallic else 'ud'[vbm] + 'ud'[cbm],
        'metallic_bands': {'u': su['metallic_bands'], 'd': sd['metallic_bands']},
        'is_metallic': is_metallic,
        'metallic_type': metallic_type
    }

    return res
