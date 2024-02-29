import numpy as np
import os

from . import HA2EV
from .DataReader import DataReader
from .debug_utils import *

class AMSReader(DataReader):
    def __init__(self, dfolder, prefix='', **kwargs):
        self.dfolder = dfolder
        self.prefix = prefix
        self._files = get_files(self.dfolder, self.prefix)
        self.__bands_cache = None
        self.__dos_cache = None
        self.band_ev_range = kwargs.pop('band_ev_range', None)
        if kwargs:
            raise TypeError(f'Unknown keyword arguments for AMSReader: {list(kwargs.keys())}')

    @property
    def _bands_data(self):
        if self.__bands_cache is None:
            self.__bands_cache = self._read_band_data()
        return self.__bands_cache

    @property
    def _dos_data(self):
        if self.__dos_cache is None:
            self.__dos_cache = self._read_dos_data()
        return self.__dos_cache

    def _read_dos_data(self):
        dosf = self._files['dos']
        e_dos, dosup, dosdwn = read_dos(dosf)
        return {'E': e_dos, 'up': dosup, 'dw': dosdwn}


    def _read_band_data(self):
        if self._files['band'] is None:
            b1, b2 = self._files['band_up'], self._files['band_dw']
            segment1, knum1, r1, efermi1, bands1, fat1 = read_bandfile(b1, ev_lims=self.band_ev_range)
            segment2, knum2, r2, efermi2, bands2, fat2 = read_bandfile(b2, ev_lims=self.band_ev_range)
            # just to be sure:
            if not np.all([np.all(np.isclose(a, b)) for a, b in ((segment1, segment2), (knum1, knum2), (r1, r2), (efermi1, efermi2))]):
                raise ValueError('The band up and down files do not correspond.')
        else:
            b1 = self._files['band']
            segment1, knum1, r1, efermi1, bands1, fat1 = read_bandfile(b1)
            bands2 = None
            fat2 = None
        symcoords = get_symcoords(segment1, r1)

        return {'E_fermi': efermi1, 'k': r1, 'sym': symcoords, 'band_up': bands1, 'band_dw': bands2, 'fat_up': fat1, 'fat_dw': fat2}

    def bands_available(self):
        return not (self._files['band'] is None and self._files['band_up'] is None)

    def dos_available(self):
        return self._files['dos'] is not None

    def get_bands(self):
        return self._bands_data['band_up'], self._bands_data['band_dw']

    def get_band_fatness(self):    # each ndarray is 2D
        return self._bands_data['fat_up'], self._bands_data['fat_dw']

    def get_efermi(self):
        return self._bands_data['E_fermi']

    def get_kcoords(self):    #1D array
        return self._bands_data['k']

    def get_symcoords(self):  # 1D array
        return self._bands_data['sym']

    def get_dos(self):    # 1D array
        return self._dos_data['up'], self._dos_data['dw']

    def get_dos_e(self):  # 1D array
        return self._dos_data['E']


def get_paths(dfolder, prefix="", postfix=""):
    return {
        'band': os.path.join(dfolder, prefix + "band" + postfix + ".csv"),
        'band_up': os.path.join(dfolder, prefix + "bandalpha" + postfix + ".csv"),
        'band_dw': os.path.join(dfolder, prefix + "bandbeta" + postfix + ".csv"),
        'dos': os.path.join(dfolder, prefix + "dos" + postfix + ".dat")
        }


def get_files(dfolder, prefix='', postfix=''):
    paths = get_paths(dfolder, prefix)  # band, bandalpha, bandbeta, dos
    # MARK(paths, n='path')
    files = {k: p if os.path.exists(p) else None for k, p in paths.items()}
    # MARK(files, n='file')
    return files


def read_bandfile(fpath, ev_lims=None):

    with open(fpath, 'r') as file:
        header = file.readline().strip().split(',')
    has_fatbands = any('fatband' in col for col in header)

    data = np.genfromtxt(fpath, delimiter=',', skip_header=1).transpose()
    data = np.genfromtxt(fpath, delimiter=',', skip_header=1).transpose()
    segment = data[0]
    knum = data[1]
    r = data[2]
    efermi = data[3][0] * HA2EV
    bands_ha = data[4:]
    if has_fatbands:
        nbands = len(bands_ha) // 2
        if nbands % 1:
            print(nbands, len(bands_ha))
            raise ValueError('something is off')
        bands_ev = bands_ha[:nbands] * HA2EV
        fat = bands_ha[nbands:]
    else:
        bands_ev = bands_ha * HA2EV
        fat = None # np.ones_like(bands_ev)
    # print("Number of bands: ", len(bands))

    if ev_lims is not None:
        bandmaxs = np.amax(bands_ev, axis=-1)
        bandmins = np.amin(bands_ev, axis=-1)
        inrange = np.logical_and(bandmaxs-efermi > ev_lims[0], bandmins-efermi < ev_lims[1])
        return segment, knum, r, efermi, bands_ev[inrange], (fat[inrange] if fat is not None else None)
    else:
        return segment, knum, r, efermi, bands_ev, fat


def get_symcoords(segment, r):
    segment_changes, = np.argwhere(segment[1:] != segment[:-1]).transpose()
    segment_changes = np.concatenate(([0,], segment_changes, [len(r)-1,]))
    # print(len(segment_changes))
    sym_r = r[segment_changes]
    return sym_r


def read_dos(dosfile):
    uplines, downlines = [], []
    is_spinp = False
    with open(dosfile, 'r') as f:
        l = f.readline()
        while l:
            l = f.readline()
            if "Curve" in l:    # only spinpolarized case has multiple headers (or if the data is projected to s,p,d,f which is not implemented)
                is_spinp = True
                break
            arr = np.fromstring(l.strip(), dtype='float', sep=' ')
            uplines.append(arr)
        while l: # only True if spinpolarized
            l = f.readline()
            if "Curve" in l:
                raise NotImplementedError("The data file contains projected s,p,d,f DOS. Handling this is not implemented.")
            if l:
                downlines.append(np.fromstring(l.strip(), dtype=float, sep=' '))
    up, e_dos = np.array(uplines[:-1]).T
    if is_spinp:
        dwn, e_dos2 = np.array(downlines[:-1]).T
        if not np.allclose(e_dos, e_dos2):
            raise ValueError('Erroneous DOS file: up and dwn energies differ')
    else:
        dwn = None
    return e_dos, up, dwn
