import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import os
import re

mpl.rcParams['figure.dpi'] = 200
# matplotlib.use("svg")

# name='V4'
# symmetries = (0.0000, 0.70707, 1.0607, 1.5607, 2.1730, 2.9230, 3.1730, 3.4230, 3.6730)
# symm_names = ("X", 'M', 'A','R','X',r'\Gamma', 'R', 'Z',r'\Gamma')
symm_names = (r"\Gamma", "Z", "R", "X", "\Gamma",  "A", "M", "\Gamma")

prefix = 'vi'

vacs = 'pr', # 'vi', 'vpb'
absorbs = None, 'ar', 'co2', 'h2o', 'n2', 'o2'

# efermi = -3.9942 #eV

fermi_offset = 0

dfolder = "./mapbi3_band_dos_data"

# bandf = "data/v4nsp_band.csv"
# dosfile = "data/v4nsp_dos.csv"

lims = -3, 3

HA2EV = 27.2114
plotsize=(3.5,2.5)
fig_dpi=600

def get_paths(dfolder, prefix):
    ba = os.path.join(dfolder, prefix + "_bandalpha.csv")
    bb = os.path.join(dfolder, prefix + "_bandbeta.csv")
    dos = os.path.join(dfolder, prefix + "_dos.dat")
    return ba, bb, dos


def read_bandfile(fpath):
    data = np.genfromtxt(fpath, delimiter=',', skip_header=1).transpose()
    segment = data[0]
    knum = data[1]
    r = data[2]
    efermi = data[3][0] * HA2EV
    bands = data[4:] * HA2EV
    # print("Number of bands: ", len(bands))

    return segment, knum, r, efermi, bands


def get_symcoords(segment, r):
    segment_changes, = np.argwhere(segment[1:] != segment[:-1]).transpose()
    segment_changes = np.concatenate(([0,], segment_changes, [len(r)-1,]))
    # print(len(segment_changes))
    sym_r = r[segment_changes]
    return sym_r


def read_dos(dosfile):
    uplines, downlines = [], []
    with open(dosfile, 'r') as f:
        l = f.readline()
        while l:
            l = f.readline()
            if "Curve" in l:
                break
            arr = np.fromstring(l.strip(), dtype='float', sep=' ')
            uplines.append(arr)
        while l:
            l = f.readline()
            if l:
                downlines.append(np.fromstring(l.strip(), dtype=float, sep=' '))
    e_dos, up = np.array(uplines).T
    e_dos, dwn = np.array(downlines).T
    return e_dos * HA2EV, up, dwn

    # dos, e_dos = np.genfromtxt(dosf, skip_header=1).transpose()


def read_data(prefix):
    ba, bb, dosf = get_paths(dfolder, prefix)
    segment1, knum1, r1, efermi1, bandsup = read_bandfile(ba)
    segment2, knum2, r2, efermi2, bandsdwn = read_bandfile(bb)
    # just to be sure:
    assert np.all([np.all(np.isclose(a, b)) for a, b in ((segment1, segment2), (knum1, knum2), (r1, r2), (efermi1, efermi2))])
    symcoords = get_symcoords(segment1, r1)

    e_dos, dosup, dosdwn = read_dos(dosf)

    return efermi1, (r1, symcoords, bandsup, bandsdwn), (e_dos, dosup, dosdwn)


def draw_arrow(ax, A, B, **kwargs):
    ax.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],**kwargs)


def plot_bands(ax, bandsup, bandsdwn, x, fermi):
    xaxis = [min(x),max(x)]
    color = 'black'
    lw = 0.6
    for i, band in enumerate(bandsdwn):
        ax.plot(x, np.array(band)-fermi, color='red', lw=0.39, linestyle='--', label='spin-dwn' if not i else None)
    for i, band in enumerate(bandsup):
        ax.plot(x, np.array(band)-fermi, color=color, lw=0.4, linestyle='-', label='spin-up' if not i else None)
    ax.axhline(0, color="gray",ls="--", alpha = 0.5,lw = 1)
    ax.set_ylabel(r'$E - E_{Fermi}$')
    ax.set_xlim(xaxis)


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

    draw_arrow(ax, (x_max_below, max_below-fermi), (x_max_below, min_above-fermi),length_includes_head=True, head_width=0.025, head_length=bgap*0.15, color='k')
    draw_arrow(ax, (x_max_below, min_above-fermi), (x_max_below, max_below-fermi),length_includes_head=True, head_width=0.025 ,head_length=bgap*0.15, color='k')

    middle = max_below + bgap/2 - fermi
    if x_max_below <= np.amax(x) / 2:
        ax.annotate(rf"$\Delta E={bgap:.2} eV$", (x_max_below + 0.01, middle), verticalalignment='center')
    else:
        ax.annotate(rf"$\Delta E={bgap:.2} eV$", (x_max_below - 0.01, middle), verticalalignment='center', horizontalalignment='right')

    # ax.scatter(x_max_below, max_below-fermi, c='gray', marker=6, alpha=0.7, label="bandgap low bound")
    # ax.scatter(x_min_above, min_above-fermi, c='gray', marker=7, alpha=0.7, label="bandgap high bound")
    return bgap


def plot_dos(ax, dos_en, dosup, dosdwn, efermi, ylims=None):
    # dos, energy = np.genfromtxt(dosfile, skip_header=1).transpose()
    y = dos_en-efermi

    ax.plot(dosup, y, color='black', linewidth=0.6)
    ax.plot(-dosdwn, y, color='black', linewidth=0.6)
    #ax.set_xlabel("$E (eV) - E_{Fermi}$")
    ax.fill_betweenx(y, dosup, 0*dosup, color='silver', alpha=1.0, where=(y < 0))#ax.set_xlabel("$E (eV) - E_{Fermi}$")
    ax.fill_betweenx(y, -dosdwn, 0*dosdwn, color='silver', alpha=1.0, where=(y < 0))#ax.set_xlabel("$E (eV) - E_{Fermi}$")
    if ylims is not None:
        where = np.logical_and(y > ylims[0], y < ylims[1])
    else:
        where = np.ones_like(y, dtype=bool)
    ax.set_xlim(-np.amax(dosdwn[where])*1.1, np.amax(dosup[where])*1.1)
    ax.axvline(0, color='k', ls="solid", alpha=1.0, lw=0.5)
    #ax.axhline(0, color='#66ccff', ls="solid", alpha=0.5, lw=1.2)
    #ax.set_xlabel("DOS")
    #ax.title("C16 DOS")


def make_plot(prefix):
    fig = plt.figure(figsize=plotsize)
    gs = fig.add_gridspec(1, 10, wspace=0, hspace=0)
    bandsax = fig.add_subplot(gs[0:7])
    dosax = fig.add_subplot(gs[7:], sharey=bandsax)


    efermi, bandsdata, dosdata = read_data(prefix)
    (r, symcoords, bandsup, bandsdwn) = bandsdata
    (e_dos, dosup, dosdwn) = dosdata
    assert len(symcoords) == len(symm_names)

    # segment, knum, x, efermi, bands = read_bandfile(bandf)
    # symcoords = get_symcoords(segment, x)

    # fermi = read_fermi(scff)
    fermi = efermi
    print(f"fermi energy: {fermi}")

    #    bands, x = read_bnd(bandf)
    plot_bands(bandsax, bandsup, bandsdwn, r, fermi)

    # bandgap = fermi_bandgap(bandsax, bands, x, fermi, fermi_offset)
    # print(f"Fermi bandgap: {bandgap}")

    plot_symmetries(bandsax, symcoords, symm_names)

    plot_dos(dosax, e_dos, dosup, dosdwn, fermi, ylims=lims)
    dosax.set_ylabel("DOS")
    dosax.set_axis_off()
    # dosax.axes.get_yaxis().set_visible(False)
    # dosax.xaxis.tick_top()
    # bandsax.set_title(name + " band structure and DOS")

    # fig.text(0.1, 0.9, f"Fermi energy: {fermi:.5f} eV\n"
	    #  f"Bandgap: N/A")# {bandgap:.5f} eV")

    bandsax.set_ylim(-7-efermi, -2-efermi)
    # fig.suptitle(prefix.upper())

    plotfile = f"{prefix}_bands"
    plt.tight_layout()
    plt.savefig(os.path.join('plots', plotfile + '.png'), dpi=fig_dpi)
    plt.savefig(os.path.join('plots', 'svg', plotfile + '.svg'), dpi=fig_dpi)
    # plt.show()
    plt.close()
    print(f"figure saved to: " + plotfile)

def main():
    for vac in vacs:
        for ab in absorbs:
            if ab is None:
                prefix = vac
            else:
                if vac == 'pr':
                    prefix = ab
                else:
                    prefix = vac + '_' + ab
            make_plot(prefix)


main()
