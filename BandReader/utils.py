"Various useful utilities"

from __future__ import annotations

import collections
import inspect
import json
import os
import sys
from collections import defaultdict
from copy import copy
from dataclasses import asdict, make_dataclass
from enum import Enum
from functools import wraps
from typing import (Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar,
                    Union, overload)

import numpy as np
import termcolor as tc

T = TypeVar("T")
Vectorized = TypeVar("Vectorized", int, float, complex, np.ndarray)

def multiply_along_axis(A, B, axis: int = -1):
    """
    Multiply A with the 1D array B along the specified axis.
    B.size must be the same as A.shape[axis].

    Args:
        A (np.ndarray): array to be multiplied
        B (np.ndarray): 1D array of multipliers
        axis (int): the axis along which A will be multiplied

    Raises:
        ValueError: if B.size != A.shape[axis]

    Returns:
        np.ndarray: A mutiplied with B along the specified axis.
    """

    # just to be sure
    A = np.array(A)
    B = np.array(B)

    # check if B is a single number (not the intended use but there's no need to crash)
    try:
        return A * B.item()
    except ValueError:
        pass

    # check shape
    if not A.shape:
        raise ValueError("Cannot multiply empty array")
    if A.shape[axis] != B.size:
        raise ValueError(
            f"'A' and 'B' must have the same length along the given axis, but "
            f"they have A: {A.shape[axis]} and B: {B.shape[axis]}"
        )

    shape = np.swapaxes(A, A.ndim - 1, axis).shape
    B_brc = np.broadcast_to(B, shape)
    B_brc = np.swapaxes(B_brc, A.ndim - 1, axis)
    return A * B_brc


def obj_array_to_type(arr, typ):
    """
    Convert an object array of same-sized arrays to a normal 3D array with dtype=typ.

    This is a workaround as numpy doesn't deteck if the object arrays are numpy
    arrays of the same legth, so just using array.astype(typ) fails. Technically
    works if the items are numbers and not arrays, but then it's better to just
    use `arr.astype(typ)`.
    """
    full_shape = (*arr.shape, *np.shape(arr.flat[0]))
    return np.vstack(arr.flatten()).astype(typ).reshape(full_shape)


def argfind_width_above(y, thr):
    """
    Find the args of the 1D array y that enclose a region where y > thr.
    If there are multiple such regions, the start of the first and the end
    of the last region is returned.
    """
    imax = np.argmax(y)
    l = len(y)
    if imax == 0:
        ilow = 0
    else:
        ilow = np.searchsorted(y[:imax], thr)
    if imax == l:
        ihigh = l-1
    else:
        ihigh =  l - np.searchsorted(y[imax:][::-1], thr)
        if ihigh == l:
            ihigh = l-1

    return ilow, ihigh


def find_region_count_above(y, thr):
    """
    Find the number of regions in y where  y > thr. Specifically, find the number of
    + -> - crossings through `thr`, and add 1 if the last value is greater than thr.
    """
    if np.ndim(y) != 1:
        raise ValueError('only works with 1D arrays')
    data = np.array(y) - thr
    #count + -> - zero-crossings (credit to @lmjons3: https://stackoverflow.com/a/21468492)
    pos = data > 0
    p_m_crossings = (pos[:-1] & ~pos[1:]).nonzero()[0]
    n = len(p_m_crossings)
    if data[-1] > 0:
        n += 1 # we didn't count the last section
    return n

def sym_lims(low, high, middle=0):
    "symmetrize a lower and upper bound WRT middle (e.g. for plotting)"
    mn, mx = low-middle, high-middle
    if abs(mn) > abs(mx):
        mx = -mn
    else:
        mn = -mx
    return mn+middle, mx+middle

def quickplot(x,y, title, *args, **kwargs):
    "Lazy way to show plt.plot(x,y) and add a title."
    fig, ax = plt.subplots()
    ax.plot(x,y, *args, **kwargs)
    fig.suptitle(title)
    plt.show()

def quickpcolormesh(x, y, z, title, *args, **kwargs):
    "Lazy way to show plt.plot(x,y) and add a title."
    fig, ax = plt.subplots()
    ax.pcolormesh(x,y,z, *args, **kwargs)
    fig.suptitle(title)
    plt.show()


#################################################
# ~~~~~ Terminal related ~~~~~
#################################################
def _terminal_supports_color():
    """
    Return True if the running system's terminal supports color,
    and False otherwise.
    Code cloned from https://github.com/django/django/blob/main/django/core/management/color.py
    """
    def vt_codes_enabled_in_windows_registry():
        """
        Check the Windows Registry to see if VT code handling has been enabled
        by default, see https://superuser.com/a/1300251/447564.
        """
        try:
            # winreg is only available on Windows.
            import winreg
        except ImportError:
            return False
        else:
            reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Console')
            try:
                reg_key_value, _ = winreg.QueryValueEx(reg_key, 'VirtualTerminalLevel')
            except FileNotFoundError:
                return False
            else:
                return reg_key_value == 1

    # isatty is not always implemented, #6223.
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

    return is_a_tty and (
        sys.platform != 'win32' or
        # HAS_COLORAMA or
        'ANSICON' in os.environ or
        # Windows Terminal supports VT codes.
        'WT_SESSION' in os.environ or
        # Microsoft Visual Studio Code's built-in terminal supports colors.
        os.environ.get('TERM_PROGRAM') == 'vscode' or
        vt_codes_enabled_in_windows_registry()
    )
try:
    TERM_SUPPORTS_COLOR = _terminal_supports_color() #: Whether the terminal supports color codes.
except:
    TERM_SUPPORTS_COLOR = False # failsafe, no need to break due to colors

def tc_colored(text, *args, **kwargs):
    """
    termcolor.colored if colors are supported, othervise return the
    untouched string.
    """
    if TERM_SUPPORTS_COLOR:
        return tc.colored(text, *args, **kwargs)
    else:
        return text

def cprint(text, *args, **kwargs):
    """
    termcolor.cprint if colors are supported, othervise regular print.
    """
    if TERM_SUPPORTS_COLOR:
        tc.cprint(text, *args, **kwargs)
    else:
        print(text)


def query_yes_no(question: str, default="yes") -> bool:
    """
    Ask a yes/no question via raw_input() and return their answer.
    Source: https://code.activestate.com/recipes/577058/

    Params:
        question: string that is presented to the user.
        default: the presumed answer if the user just hits <Enter>. It must be
            "yes", "no" or None (meaning an answer is required of the user).

    Returns:
        answer: True for "yes" or False for "no".

    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")



#################################################
# ~~~ Math / geometry
#################################################


def round_relative(a: np.ndarray, decimals: int) -> np.ndarray:
    """Round :param:`a` to the given number of decimals.

    If a.max is x**10^(-y), and decimals is d, then the array is rounded up
    to y + d digits.

    Args:
        a (np.ndarray): array to round
        decimals (int): number of decimals relative to a.max

    Returns:
        np.ndarray: a rounded to int(-log10(a.max) + decimals) digits
    """
    absmax = np.amax(np.abs(a))
    ndigits = int(-np.log10(absmax) + decimals)
    return a.round(ndigits)


def nozeros(v: Vectorized, val: float = 1e-100) -> Vectorized:
    """replace zeros with val (to avoid division by zeros)."""
    if is_array_like(v):
        a = np.array(v)
        a[a == 0] = val
        return a
    else:
        return val if v == 0 else v


def cgradient(f, *varargs, axis=-1, edge_order=1):
    """Derivate the complex data as (Re(f(x)) + Im(f(x)) using np.gradient.
    Axis must be given."""
    dre = np.gradient(np.real(f), *varargs, axis=axis, edge_order=edge_order)
    dim = np.gradient(np.imag(f), *varargs, axis=axis, edge_order=edge_order)
    # res = dabs * np.exp(1j * argf) +
    res = dre + 1j * dim
    return res


def cgradient_exp(f, *varargs, axis=-1, edge_order=1, discont=np.pi):
    """
    Alternative complex derivate calculation avoiding rapid oscillation.
    Instead of taking the gradient of d/dx (Re(f(x)) + Im(f(x)), this calculates it
    as d/dx (abs(f(x))*exp(i*arg(f(x))).

    The phase must be accurately unwrappable.

    Todo:
        - Implement checks on gradient unwrapping
    """
    absf = np.abs(f)
    argf = np.unwrap(np.angle(f), discont=discont, axis=axis)
    dabs = np.gradient(absf, *varargs, axis=axis, edge_order=edge_order)
    darg = np.gradient(argf, *varargs, axis=axis, edge_order=edge_order)
    # res = dabs * np.exp(1j * argf) +
    res = (dabs + 1j * absf * darg) * np.exp(1j * argf)
    return res


def cylind2cart(
        r: Vectorized, phi: Vectorized, z: Vectorized
) -> Tuple[Vectorized, Vectorized, Vectorized]:
    """Convert the cylindrical coordinates to cartesian."""
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, z


def cart2cylind(
        x: Vectorized, y: Vectorized, z: Vectorized
) -> Tuple[Vectorized, Vectorized, Vectorized]:
    """Convert the cartesian coordinates to cylindrical."""
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return r, phi, z


#################################################
# ~~~ Array semantics
#################################################

# Cool hack: nested dictionary that doesn't need an empty dict associated to
# each entry in each sub-lebel. Just use as: a = NestedDict(); a['x']['y']['z'] = 1.0
NestedDict = lambda: defaultdict(NestedDict)


def is_array_like(a: Any) -> bool:
    """Deterimne if a is an array-like object; that is,
    return isinstance(a, (np.ndarray, collections.sequence))

    Args:
        a (any): object to check

    Returns:
        bool: whether a is array-like
    """
    return isinstance(a, (collections.Sequence, np.ndarray))


def multiply_along_axis(A: np.ndarray, B: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Multiply A with the 1D array B along the specified axis.
    B.size must be the same as A.shape[axis].

    Args:
        A (np.ndarray): array to be multiplied
        B (np.ndarray): 1D array of multipliers
        axis (int): the axis along which A will be multiplied

    Raises:
        ValueError: if B.size != A.shape[axis]

    Returns:
        np.ndarray: A mutiplied with B along the specified axis.
    """

    # just to be sure
    A = np.array(A)
    B = np.array(B)

    # check if B is a single number (not the intended use but there's no need to crash)
    try:
        return A * np.array(B).item()
    except ValueError:
        pass

    # check shape
    if not A.shape:
        raise ValueError("Cannot multiply empty array")
    if A.shape[axis] != B.size:
        raise ValueError(
            f"'A' and 'B' must have the same length along the given axis, but "
            f"they have A: {A.shape[axis]} and B: {B.shape[axis]}"
        )

    shape = np.swapaxes(A, A.ndim - 1, axis).shape
    B_brc = np.broadcast_to(B, shape)
    B_brc = np.swapaxes(B_brc, A.ndim - 1, axis)
    return A * B_brc


def argfind_width_above(y, thr):
    """ Find the args of the 1D array x that enclose a specific value above thr.
    If there are multiple such regions, the start of the first and the beginning
    of the last is returned.
    """
    imax = np.argmax(y)
    l = len(y)
    if imax == 0:
        ilow = 0
    else:
        ilow = np.searchsorted(y[:imax], thr)
    if imax == l:
        ihigh = l-1
    else:
        ihigh =  l - np.searchsorted(y[imax:][::-1], thr)
        if ihigh == l:
            ihigh = l-1

    return ilow, ihigh


#################################################
# ~~ Dict <-> dataclass <-> str manipulations
#################################################


@overload
def to_serializable(a: Enum) -> Any:
    ...


@overload
def to_serializable(a: np.array) -> list:
    ...


@overload
def to_serializable(a: T) -> T:
    ...


def to_serializable(a):
    """If (something in) a is an np.ndarray, convert it to list, else return untouched"

    This helps in cases when :param:a nees to be a pickleable data, to e.g. pass
    to jsonstr.

    Args:
        a: item to pass through

    Returns:
        list(a) if a is a Numpy array, else a untouched.
    """

    if isinstance(a, dict):
        return {key: to_serializable(val) for key, val in a.items()}  # yay recursion
    if isinstance(a, collections.Sequence) and not isinstance(a, str):
        return [to_serializable(x) for x in a]
    if isinstance(a, np.ndarray):
        return a.tolist()
    if isinstance(a, np.integer):
        return int(a)
    if isinstance(a, np.floating):
        return float(a)
    if isinstance(a, Enum):
        return to_serializable(a.value)
    return a


def dict_to_dataclass(d: Dict[str, Any]) -> Type:
    """Convert the dictionary as a simple dataclass with the keys as fields and
    values as field values. d["key"] will become dict_to_dataclass(d).key.
    """
    Dclass = make_dataclass("Parameters", list(d.keys()))
    return Dclass(**d)


def dataclass_to_dict(
        dclass: Any,
        blacklist: Optional[List[str]] = None,
        whitelist: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convert a dataclass to a dictionary with whitelisting or blacklisting attributes.

    Args:
        dclass (Any): dataclass to convert
        blacklist (, optional): Keys to exclude. Defaults to None.
        whitelist (, optional): Keys to include, or None to include all. Defaults to None.
    """
    # assert type(dict) is dict
    dct = asdict(dclass)
    if whitelist is not None:
        d = dict((key, dct.pop(key)) for key in whitelist)
        dct = d
    if blacklist is not None:
        for delkey in blacklist:
            dct.pop(delkey, "")
    return dct


def dataclass_to_json_dict(
        dclass: Any,
        blacklist: Optional[List[str]] = None,
        whitelist: Optional[List[str]] = None,
) -> dict:
    """Pickle the dataclass into a dictionary suitable for json parsing
    according to the dataclass's fields.

    Args:
        dclass (type): dataclass to be parsed to dictionary. The dataclass fields
            should be pickleable or np.ndarray.
        blacklist (list of str, optional): Exclude fields with these names.
            Defaults to None.
        whitelist (list of str, optional): Only include fields with these names.
            Defaults to None.

    Returns:
        [type]: [description]
    """
    dct = dataclass_to_dict(dclass, blacklist, whitelist)
    return to_serializable(dct)


def dataclass_to_json_str(
        dclass: Any,
        blacklist: Optional[List[str]] = None,
        whitelist: Optional[List[str]] = None,
        sort_keys: bool = True,
        indent: int = 4,
) -> str:
    """Convert the dataclass to a json string.

    Args:
        dclass (type): dataclass to be parsed to dictionary. The dataclass fields
            should be pickleable or np.ndarray.
        blacklist (list of str, optional): Exclude fields with these names.
            Defaults to None.
        whitelist (list of str, optional): Only include fields with these names.
            Defaults to None.
        sort_keys (bool, optional): Whether to sort the keys in alphabetical
            order instead of the order provided by the dataclass. Defaults to True.
        indent (int, optional): Amoundt of indentation in the string. Defaults to 4.

    Returns:
        str: json string containing the dataclass fields.
    """

    dct = dataclass_to_json_dict(dclass, blacklist, whitelist)
    return json.dumps(dct, sort_keys=sort_keys, indent=indent)


def dataclass_to_json_file(
        dclass: Any,
        fpath=str,
        blacklist: Optional[List[str]] = None,
        whitelist: Optional[List[str]] = None,
        sort_keys: bool = True,
        indent: int = 4,
) -> None:
    """Save the dataclass to a json file.

    Args:
        dclass (type): dataclass to be parsed to dictionary. The dataclass fields
            should be pickleable or np.ndarray.
        fpath (str): Path to the resulting file.
        blacklist (list of str, optional): Exclude fields with these names.
            Defaults to None.
        whitelist (list of str, optional): Only include fields with these names.
            Defaults to None.
        sort_keys (bool, optional): Whether to sort the keys in alphabetical
            order instead of the order provided by the dataclass. Defaults to True.
        indent (int, optional): Amoundt of indentation in the string. Defaults to 4.

    """
    jsonstr = dataclass_to_json_str(dclass, blacklist, whitelist, sort_keys, indent)
    with open(fpath, "w+") as f:
        print(jsonstr, file=f)


def dict_to_pretty_str(
        dct: Dict,
        blacklist: Optional[List] = None,
        whitelist: Optional[List] = None,
        linestart: str = "",
        header: Optional[str] = None,
        colored: bool = False,
        headercolor: Optional[str] = "blue",
        namecolor: Optional[str] = "green",
        valcolor: Optional[str] = None,
        indent: str = "  ",
) -> str:
    """Convert a dictionary to a formatted, and possibly colored, string as

    header (if given)
    [linestart]str(key1): str(value1)
    [linestart]str(key2): str(value2)
    ...

    Args:
        dct (dict): dictionary to prettyprint
        blacklist (, optional): Keys to exclude. Defaults to None.
        whitelist (, optional): Keys to include, or None to include all. Defaults to None.
        linestart (str, optional): Beginning of each line. Defaults to "".
        header: First line of the returned string, if given. Defaults to None.
        colored (bool, optional): Whether to color the output according to
            :param:`namecolor` and :param:`valcolor` If True, use `utils.cprint`
            instead of print to translate . Defaults to False.
        headercolor (str, optional): Color of the header. Defaults to "blue".
        namecolor (str, optional): Color of the keys. Defaults to "green".
        valcolor ([type], optional): Color of the field values. Defaults to None.
        indent: the indent preceeding each level.
    """
    # assert type(dict) is dict
    dct = copy(dct)
    if whitelist is not None:
        d = dict((key, dct.pop(key)) for key in whitelist)
        dct = d
    if blacklist is not None:
        for delkey in blacklist:
            dct.pop(delkey, "")

    separator = ": "
    lineend = "\n"
    printstr = ""
    if not colored:
        headercolor = namecolor = valcolor = None

    if header is not None:
        printstr += tc_colored(header, headercolor) + lineend

    for key, val in dct.items():
        if isinstance(val, dict):
            # because recursion is cool; generate pretty string with added indent
            new_linestart = linestart + indent
            pretty = dict_to_pretty_str(
                dct=val,
                blacklist=None,
                whitelist=None,
                linestart=new_linestart,
                header=None,
                colored=colored,
                headercolor=headercolor,
                namecolor=namecolor,
                valcolor=valcolor,
                indent=indent,
            )
            printstr += tc_colored(str(key) + separator + "{", namecolor) + lineend
            printstr += pretty + lineend
            printstr += tc_colored(indent + "}", namecolor) + lineend

        else:
            printstr += (
                    linestart
                    + tc_colored(str(key) + separator, namecolor)
                    + tc_colored(str(val), valcolor)
                    + lineend
            )
    return printstr[: -len(lineend)]  # remove last newline


def dataclass_to_pretty_str(dataclass: Any, **kwargs) -> str:
    """Convert the dataclass to dictionary and pass to `dict_to_pretty_str` with the supplied arguments."""
    return dict_to_pretty_str(asdict(dataclass), **kwargs)


# TODO are these two neccessary here?
def prettycprint_dict(dct: Dict, **kwargs):
    """Pretty print the dictionary using `dataclass_to_pretty_str`."""
    print(dict_to_pretty_str(dct, **kwargs))


def prettycprint_dataclass(dclass: Any, **kwargs):
    """Pretty print the dataclass's fields using `dataclass_to_pretty_str`."""
    print(dataclass_to_pretty_str(dclass, **kwargs))