"Various useful utilities"

from __future__ import annotations

import inspect
import os
from typing import Callable
from .utils import prettycprint_dict
from . import ENABLE_DEBUG


def debug_only(func: Callable) -> Callable:
    if not ENABLE_DEBUG:
        return lambda *args, **kwargs: None
    else:
        return func

@debug_only
def DB_PRINT(*args, **kwargs):
    print(*args, **kwargs)

@debug_only
def MARK(mark, max_levels=20, pause=False, verbose=True, n=None):
    """Place a mark that gets printed to std with line and stack info."""
    maxlevels = len(inspect.stack()) - 2
    functs, i = [], 0
    stack = inspect.stack()
    while i < max_levels and i < maxlevels:
        funct = stack[i+1].function
        lineno = stack[i+1].lineno
        fname = os.path.basename(stack[i+1].filename)
        functs.append(f'{funct}[{fname}:{lineno}]' if verbose else f'{funct}[{lineno}]')
        i += 1
    # fname = os.path.basename(inspect.stack()[1].filename)
    fname = stack[1].filename
    line = stack[1].lineno
    functstr = ' <- '.join(functs)
    if verbose:
        print(f'MARK{"" if n is None else f"[{n}]"} >>> {mark}\n     >>> Call stack: {functstr} <<< in {fname}, line {line}')
    else:
        print(f'MARK{"" if n is None else f"[{n}]"} >>> {mark} <<< {functstr} || in {os.path.basename(fname)}, line {line}')
    if pause:
        input("Paused; press Enter to continue... ")
        print("\033[A                             \033[A")


_call_count: dict = {}
@debug_only
def COUNT_CALL(print_count=True):
    func_name = inspect.stack()[1].function
    func_fname = inspect.stack()[1].filename
    refname = func_name + '_' + func_fname
    count = _call_count.get(refname, 0)
    count += 1
    _call_count[refname] = count
    if print_count:
        print(f'{func_name} called {count} times so far (in {os.path.basename(func_fname)}')


@debug_only
def PRINT_COUNT_CALL(func_name=None):
    if func_name is None:
        prettycprint_dict(_call_count)
    else:
        n = _call_count.get(func_name, None)
        if n is None:
            n = 'N/A'
        print(func_name + ' called ' + str(n) + ' times so far.')


@debug_only
def generate_pycallgraph(
        func: Callable,
        *args,
        ofile="pycallgraph.png",
        depth=10,
        excluded=["_find_and_load.*"],
) -> None:
    """
    Profile the supplied function and create a Pycallgraph image to visualize
    the result. Depends on the pycallgraph2 external module.

    Args:
        func (function): function to benchmark.
        ofile (str, optional): Path ho output file. Defaults to "pycallgraph.png".
        depth (int, optional): Defaults to 10.
        excluded (list, optional): Defaults to ["_find_and_load.*"].
    """
    try:
        from pycallgraph2 import Config, GlobbingFilter, PyCallGraph
        from pycallgraph2.output import GraphvizOutput
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "generate_pycallgraph depends on the pycallgraph2 module; "
            "please install it with 'pip install pycallgraph2'."
        )

    config = Config(max_depth=depth)
    config.trace_filter = GlobbingFilter(exclude=excluded)
    graphviz = GraphvizOutput(output_file=ofile)
    with PyCallGraph(graphviz):
        func(*args)
