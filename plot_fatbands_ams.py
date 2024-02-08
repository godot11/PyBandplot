
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib import colors
from scipy.interpolate import interp1d
import os
import re

mpl.rcParams['figure.dpi'] = 200
# mpl.use("svg")

def dumb_show(fig):
    plt.show()
    return
    "save a temporary image and open it with system viewer"
    tmpfile = '/tmp/mpl.png'
    fig.savefig(tmpfile)
    plt.close(fig)
    # os.system("xdg-open " + tmpfile)
    os.system("gwenview " + tmpfile)

HA2EV = 27.2114

prefix = 'fetetra'
ev_lims = -3, 4
load_ev_lims = -15, 15
fermi_offset = 0
plotsize = (6, 4)
fig_dpi = 600

# symmetries = (
#     0.0000,
#     0.70707,
#     1.0607,
#     1.5607,
#     2.1730,
#     2.9230,
#     3.1730,
#     3.4230,
#     3.6730)
symm_names = (r"\Gamma", 'H', 'N', r'\Gamma', 'p', 'H,P', r'\Gamma')

# efermi = -3.9942 #eV

def get_fnames(prefix):
    bandupf = f"data-yig-banddos/bandalpha-{prefix}.csv"
    banddwf = f"data-yig-banddos/bandbeta-{prefix}.csv"
    dosf = f"data-yig-banddos/dos-{prefix}.csv"
    return bandupf, banddwf, dosf

plotfile = f"bands_{prefix}.svg"


def read_bandfile(fpath, has_fatbands=False):
    data = np.genfromtxt(fpath, delimiter=',', skip_header=1).transpose()
    segment = data[0]
    knum = data[1]
    r = data[2]
    efermi = data[3][0] * HA2EV
    _bands = data[4:]
    if has_fatbands:
        nbands = len(_bands) // 2
        if nbands % 1:
            print(nbands, len(_bands))
            raise ValueError('something is off')
        bands = _bands[:nbands] * HA2EV
        fat = _bands[nbands:]
    else:
        bands = _bands * HA2EV
        fat = np.ones_like(bands)
    print("Number of bands: ", len(bands))

    bandmaxs = np.amax(bands, axis=-1)
    bandmins = np.amin(bands, axis=-1)
    # return segment, knum, r, efermi, bands, fat

    inrange = np.logical_and(bandmaxs-efermi > load_ev_lims[0], bandmins-efermi < load_ev_lims[1])
    return segment, knum, r, efermi, bands[inrange], fat[inrange]


def get_symcoords(segment, r):
    segment_changes, = np.argwhere(segment[1:] != segment[:-1]).transpose()
    segment_changes = np.concatenate(([0, ], segment_changes, [len(r)-1, ]))
    print(len(segment_changes))
    sym_r = r[segment_changes]
    return sym_r


def draw_arrow(ax, A, B, **kwargs):
    ax.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1], **kwargs)


def plot_bands(ax, bandsup, bandsdw, x, fermi):
    xaxis = [min(x), max(x)]
    for band in bandsup:
        ax.plot(x, np.array(band)-fermi, color='red', lw=0.6)
    for band in bandsdw:
        ax.plot(x, np.array(band)-fermi, color='blue', lw=0.6, linestyle='--')
    ax.axhline(0, color="gray", ls="--", alpha=0.5, lw=1)
    ax.set_ylabel(r'$E - E_{Fermi}$')
    ax.set_xlim(xaxis)


def plot_fatbands_width(ax, bandsup, bandsdw, fatup, fatdwn, x, fermi, ylims=None, fatness_fact=0.1):
    xlims = [min(x), max(x)]
    if ylims is None:
        mn, mx = np.amin([bandsup, bandsdw]), np.amax([bandsup, bandsdw])
        ylims = mn-0.1*(mx-mn), mx+0.1*(mx-mn)
    yrange = ylims[1] - ylims[0]
    fatfact = 1/np.amax([fatup, fatdwn]) * yrange * fatness_fact

    print(fatfact, np.amax(fatup), np.amax(fatdwn))

    # !! I WAS HERE
    upcolor='red'
    dwncolor='blue'
    fatalpha=0.6

    for band, fat in zip(bandsup, fatup):
        y = np.array(band) - fermi
        w = fatfact * fat
        # ax.plot(x, y, color=upcolor, lw=0.6)
        ax.fill_between(x, y, y-w/2, y+w/2, edgecolor=None, facecolor=upcolor, alpha=fatalpha)
    for band, fat in zip(bandsdw, fatdwn):
        y = np.array(band) - fermi
        w = fatfact * fat
        # ax.plot(x, y, color=dwncolor, lw=0.6)
        ax.fill_between(x, y, y-w/2, y+w/2, edgecolor=None, facecolor=dwncolor, alpha=fatalpha)
    ax.axhline(0, color="gray", ls="--", alpha=0.5, lw=1)
    ax.set_ylabel(r'$E - E_{Fermi}$')
    ax.set_xlim(xlims)


def get_transparent_cmap(color=None, cmap=None, alpha_scale='lin', log_strength=1.0):
    # get colormap
    ncolors = 256
    if cmap is not None and color is None:
        color_array = plt.get_cmap('gist_rainbow')(range(ncolors))
    elif cmap is None and color is not None:
        col = colors.to_rgba(color)
        color_array = np.tile(col, (ncolors, 1))
    else:
        raise ValueError('exactly one of "color" and "cmap" should be passed')
    # change alpha values
    if alpha_scale == 'lin':
        alphas = np.linspace(0.0, 1.0, ncolors)
    elif alpha_scale == 'log':
        alphas = np.exp(np.linspace(0.0, 1.0, ncolors)*log_strength)
        alphas -= alphas[0]
        alphas /= alphas[-1]
    else:
        raise ValueError("alpha_scale can be either 'lin' or 'log'")
    color_array[:,-1] = alphas
    print(color_array)
    map_object = colors.LinearSegmentedColormap.from_list(name='_custom_transparent',colors=color_array)
    return map_object

# # show some example data
# f,ax = plt.subplots()
# cmap = get_transparent_cmap(color='darkred') #cmap='jet')
# h = ax.imshow(np.random.rand(100,100),cmap=cmap)
# plt.colorbar(mappable=h)

# plt.show()
# exit()

from matplotlib.collections import LineCollection


# Create a set of line segments so that we can color them individually
# This creates the points as an N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be (numlines) x (points per line) x 2 (for x and y)
def plot_line_with_cmap(ax, x, y, c, cmap, min=None, max=None, **lc_kwargs):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    if min is None:
        min = c.min()
    if max is None:
        max = c.max()
    norm = plt.Normalize(min, max)
    lc = LineCollection(segments, cmap=cmap, norm=norm, **lc_kwargs)
    # Set the values used for colormapping
    lc.set_array(c)
    line = ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.01)
    return line

# x = np.linspace(0, 3 * np.pi, 500)
# y = np.sin(x)
# dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative

# fig, ax = plt.subplots()
# line = plot_line_with_cmap(ax, x, y, dydx, cmap='jet')
# fig.colorbar(line, ax=ax)
# plt.show()

# exit()

def resample(x_old, y_old, x_new, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=False):
    intp = interp1d(x_old, y_old, kind=kind, axis=axis, copy=copy, bounds_error=bounds_error, fill_value=fill_value, assume_sorted=assume_sorted)
    return intp(x_new)


def plot_fatbands_transp(ax, bandsup, bandsdw, fatup, fatdwn, x, fermi, color='red'):
    xlims = [min(x), max(x)]
    # if ylims is None:
    #     mn, mx = np.amin([bandsup, bandsdw]), np.amax([bandsup, bandsdw])
    #     ylims = mn-0.1*(mx-mn), mx+0.1*(mx-mn)
    # yrange = ylims[1] - ylims[0]
    # fatfact = 1/np.amax([fatup, fatdwn]) * yrange * fatness_fact

    # !! I WAS HERE
    # upcolor='red'
    # dwncolor='blue'
    # fatalpha=0.6

    logstr = 2
    minfactor = 0.0
    maxfactor = 1.0#0.8
    lw = 0.7

    upcmap = get_transparent_cmap(color=color, alpha_scale='log', log_strength=logstr)
    dwncmap = get_transparent_cmap(color=color, alpha_scale='log', log_strength=logstr)

    for band, fat in zip(bandsup, fatup):
        y = np.array(band) - fermi
        vmax = np.amax([np.amax(fatup), np.amax(fatdwn)]) * maxfactor
        vmin = 0 # np.amin(fatup) + minfactor*vmax
        # l = plot_line_with_cmap(ax, x, y, fat, cmap=upcmap, min=vmin, max=vmax, linewidths=lw)
    for band, fat in zip(bandsdw, fatdwn):
        print('wtf')
        y = np.array(band) - fermi
        vmax = np.amax([np.amax(fatup), np.amax(fatdwn)]) * maxfactor
        # vmax = np.amax([fatup, fatdwn]) * maxfactor
        vmin = 0# np.amin(fatdwn) + minfactor*vmax
        l = plot_line_with_cmap(ax, x, y, fat, cmap=dwncmap,  min=vmin, max=vmax, linewidths=lw, linestyles='--')
    ax.axhline(0, color="gray", ls="--", alpha=0.5, lw=1)
    ax.set_ylabel(r'$E - E_{Fermi}$')
    ax.set_xlim(xlims)
    return l


def plot_fatbands_delta(ax, x, bandsup, bandsdw, fatup1, fatdwn1, fatup2, fatdwn2, fermi, cmap='cividis'):
    xlims = [min(x), max(x)]
    # if ylims is None:
    #     mn, mx = np.amin([bandsup, bandsdw]), np.amax([bandsup, bandsdw])
    #     ylims = mn-0.1*(mx-mn), mx+0.1*(mx-mn)
    # yrange = ylims[1] - ylims[0]
    # fatfact = 1/np.amax([fatup, fatdwn]) * yrange * fatness_fact

    # !! I WAS HERE
    # upcolor='red'
    # dwncolor='blue'
    # fatalpha=0.6

    logstr = 2
    lw = 0.7

    print('mins:', np.amin(fatup1), np.amin(fatup2), np.amin(fatdwn1), np.amin(fatdwn2))
    print('maxs:', np.amax(fatup1), np.amax(fatup2), np.amax(fatdwn1), np.amax(fatdwn2))

    dfatup = (fatup1 - fatup2) / (np.abs(fatup1) + np.abs(fatup2))
    dfatdwn = (fatdwn1 - fatdwn2) / (np.abs(fatdwn1) + np.abs(fatdwn2))

    vmax = np.amax([np.amax(dfatup), np.amax(dfatdwn)])
    vmin = np.amin([np.amin(dfatup), np.amin(dfatdwn)])
    print(vmax, vmin)
    if np.abs(vmax) < np.abs(vmin):
        vmax = -vmin
    else:
        vmin = -vmax


        # np.amin(fatup) + minfactor*vmax
    # upcmap = get_transparent_cmap(color=color, alpha_scale='log', log_strength=logstr)
    # dwncmap = get_transparent_cmap(color=color, alpha_scale='log', log_strength=logstr)

    for band, fat in zip(bandsup, dfatup):
        y = np.array(band) - fermi
        l = plot_line_with_cmap(ax, x, y, fat, cmap=cmap, min=vmin, max=vmax, linewidths=lw)
    for band, fat in zip(bandsdw, dfatdwn):
        y = np.array(band) - fermi
        l = plot_line_with_cmap(ax, x, y, fat, cmap=cmap,  min=vmin, max=vmax, linewidths=lw, linestyles='--')
    ax.axhline(0, color="gray", ls="--", alpha=0.5, lw=1)
    ax.set_ylabel(r'$E - E_{Fermi}$')
    ax.set_xlim(xlims)
    return l

def plot_symmetries(ax, xcoords, names):
    for x in xcoords:
        ax.axvline(x, linestyle='-', linewidth=0.7, alpha=0.4, color='k')
    ax.set_xticks(xcoords)
    ax.set_xticklabels([r'$'+n+r'$' for n in names])


def fermi_bandgap(ax, bands, x, fermi, fermi_offset):
    mins, maxes, xmins, xmaxes = [], [], [], []
    for band in bands:
        i_min = np.argmin(band)
        i_max = np.argmax(band)
        mins.append(band[i_min])
        maxes.append(band[i_max])
        xmins.append(x[i_min])
        xmaxes.append(x[i_max])
    i_max_below = np.searchsorted(maxes, fermi+fermi_offset) - 1
    i_min_above = np.searchsorted(mins, fermi+fermi_offset)
    if i_min_above != i_max_below + 1:
        print(f"Error: band overlaps Fermi energy", i_max_below, i_min_above)
    max_below = maxes[i_max_below]
    min_above = mins[i_min_above]
    x_max_below = xmaxes[i_max_below]
    x_min_above = xmins[i_min_above]
    print(i_max_below, max_below, x_max_below)
    print(i_min_above, min_above, x_min_above)
    bgap = min_above - max_below

    draw_arrow(
        ax, (x_max_below, max_below - fermi),
        (x_max_below, min_above - fermi),
        length_includes_head=True, head_width=0.025, head_length=bgap * 0.15,
        color='k')
    draw_arrow(
        ax, (x_max_below, min_above - fermi),
        (x_max_below, max_below - fermi),
        length_includes_head=True, head_width=0.025, head_length=bgap * 0.15,
        color='k')

    middle = max_below + bgap/2 - fermi
    if x_max_below <= np.amax(x) / 2:
        ax.annotate(
            rf"$\Delta E={bgap:.2} eV$",
            (x_max_below + 0.01,
             middle),
            verticalalignment='center')
    else:
        ax.annotate(
            rf"$\Delta E={bgap:.2} eV$",
            (x_max_below - 0.01,
             middle),
            verticalalignment='center',
            horizontalalignment='right')

    # ax.scatter(x_max_below, max_below-fermi, c='gray', marker=6, alpha=0.7, label="bandgap low bound")
    # ax.scatter(x_min_above, min_above-fermi, c='gray', marker=7, alpha=0.7, label="bandgap high bound")
    return bgap


def plot_dos(ax, dosfile, efermi):
    dos, energy = np.genfromtxt(dosfile, skip_header=1).transpose()
    y = energy-efermi
    ax.plot(dos, y, color='black', linewidth=0.6)
    #ax.set_xlabel("$E (eV) - E_{Fermi}$")
    # ax.set_xlabel("$E (eV) - E_{Fermi}$")
    ax.fill_betweenx(y, dos, 0*dos, color='black', alpha=0.4, where=(y < 0))
    ax.set_xlim(0, np.amax(dos)*1.1)
    # ax.axhline(0, color='#66ccff', ls="solid", alpha=0.5, lw=1.2)
    # ax.set_xlabel("DOS")


def main():


    fig = plt.figure(figsize=plotsize)
    gs = fig.add_gridspec(1, 5, wspace=0, hspace=0)
    bandsax = fig.add_subplot(gs[0:4])
    dosax = fig.add_subplot(gs[4], sharey=bandsax)

    prefix = 'fetetra'
    bandupf, banddwf, dosf = get_fnames(prefix)
    segment, knum, x, efermi, bandsup, fatup = read_bandfile(bandupf, True)
    segment, knum, x, efermi, bandsdw, fatdw = read_bandfile(banddwf, True)
    symcoords = get_symcoords(segment, x)
    assert len(symcoords) == len(symm_names)

    # fermi = read_fermi(scff)
    fermi = efermi
    print(f"fermi energy: {fermi}")

    # l = plot_fatbands_transp(bandsax, bandsup, bandsdw, fatup, fatdw, x, fermi)
    # plt.show()
    # return
    #    bands, x = read_bnd(bandf)
    prefix = 'feocta'
    bandupf, banddwf, dosf = get_fnames(prefix)
    segment, knum, x, efermi, bandsup, fatup2 = read_bandfile(bandupf, True)
    segment, knum, x, efermi, bandsdw, fatdw2 = read_bandfile(banddwf, True)

    l = plot_fatbands_delta(bandsax, x, bandsup, bandsdw, fatup, fatdw, fatup2, fatdw2, fermi, cmap='cividis')
    # xnew = np.linspace(x[0], x[-1], 200)
    # bup, fup = resample(x, np.array([bandsup, fatup]), xnew, kind='linear', axis=-1, assume_sorted=True)
    # bdwn, fdwn = resample(x, np.array([bandsdw, fatdw]), xnew, kind='linear', axis=-1, assume_sorted=True)
    # l = plot_fatbands_transp(bandsax, bup, bdwn, fup, fdwn, xnew, fermi)
    plt.colorbar(l)
    # plot_fatbands_transp(bandsax, bandsup, bandsdw, fatup, fatdw, x, fermi)
    # bandgap = fermi_bandgap(bandsax, bands, x, fermi, fermi_offset)
    # print(f"Fermi bandgap: {bandgap}")

    symcoords = get_symcoords(segment, x)
    assert len(symcoords) == len(symm_names)

    # fermi = read_fermi(scff)

    #    bands, x = read_bnd(bandf)
    # plot_fatbands_transp(bandsax, bandsup, bandsdw, fatup, fatdw, x, fermi, color='blue')
    plot_symmetries(bandsax, symcoords, symm_names)

    # plot_dos(dosax, dosfile, fermi)
    # dosax.set_ylabel("DOS")
    # dosax.set_axis_off()
    # dosax.axes.get_yaxis().set_visible(False)
    # dosax.xaxis.tick_top()
    # bandsax.set_title(name + " band structure and DOS")

    # fig.text(0.1, 0.9, f"Fermi energy: {fermi:.5f} eV\n"
    #  f"Bandgap: N/A")# {bandgap:.5f} eV")

    bandsax.set_ylim(*ev_lims)

    # plt.savefig(plotfile, dpi=fig_dpi)
    # plt.show()
    dumb_show(fig)
    print(f"figure saved to: " + plotfile)



def main_old_full():
    fig = plt.figure(figsize=plotsize)
    gs = fig.add_gridspec(1, 5, wspace=0, hspace=0)
    bandsax = fig.add_subplot(gs[0:4])
    dosax = fig.add_subplot(gs[4], sharey=bandsax)

    segment, knum, x, efermi, bands = read_bandfile(bandf)
    symcoords = get_symcoords(segment, x)
    assert len(symcoords) == len(symm_names)

    # fermi = read_fermi(scff)
    fermi = efermi
    print(f"fermi energy: {fermi}")

    #    bands, x = read_bnd(bandf)
    plot_bands(bandsax, bands, x, fermi)

    bandgap = fermi_bandgap(bandsax, bands, x, fermi, fermi_offset)
    print(f"Fermi bandgap: {bandgap}")

    plot_symmetries(bandsax, symcoords, symm_names)

    plot_dos(dosax, dosfile, fermi)
    dosax.set_ylabel("DOS")
    dosax.set_axis_off()
    # dosax.axes.get_yaxis().set_visible(False)
    # dosax.xaxis.tick_top()
    # bandsax.set_title(name + " band structure and DOS")

    # fig.text(0.1, 0.9, f"Fermi energy: {fermi:.5f} eV\n"
    #  f"Bandgap: N/A")# {bandgap:.5f} eV")

    bandsax.set_ylim(*ev_lims)

    plt.savefig(plotfile, dpi=fig_dpi)
    plt.show()
    print(f"figure saved to: " + plotfile)


main()
