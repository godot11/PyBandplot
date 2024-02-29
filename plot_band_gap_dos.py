from BandReader import HA2EV
from BandReader.debug_utils import MARK
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import os
import re

from BandReader.Band import BandData
from BandReader.utils import prettycprint_dict

mpl.rcParams['figure.dpi'] = 250
# matplotlib.use("svg")


def main():

    fermi_offset = 0.5

    dfolder = "/home/ezio/Work/TMP_FAPB_STUFF"
    prefix = 'pristine_'
    # prefix = 'sn_25c_'
    # prefix = 'sn_25f_'
    # prefix = 'sn_50c_'
    band_data = BandData(dfolder, prefix, fermi_offset=fermi_offset)
    print('!!', prefix, '!!')


    if band_data.bands_available:
        symm_names = (r"\Gamma", 'X', 'S', 'Y', r'\Gamma', 'S', 'R', 'Z', r'\Gamma')
        band_data.assign_sym_labels(symm_names)

    lims = -1, 3.5

    HA2EV = 27.2114
    plotsize = (7, 5)
    plotsize = (6, 5)
    plotsize = (3.5, 3)
    fig_dpi = 300

    _, _, bg_info = band_data.get_bandgap_info()
    prettycprint_dict(bg_info, blacklist=['cb', 'vb'])

    fig, bandax, dosax = plot_band_dos_data(band_data, lims, plotsize, do_dos=False)
    bandax.legend(loc='upper right')

    plotfile = f"{prefix}_bands"
    plt.savefig(os.path.join(dfolder, plotfile + '.png'), dpi=fig_dpi)
    plt.savefig(os.path.join(dfolder, plotfile + '.svg'), dpi=fig_dpi)
    # plt.savefig(os.path.join('plots', plotfile + '.png'), dpi=fig_dpi)
    # plt.savefig(os.path.join('plots', 'svg', plotfile + '.svg'), dpi=fig_dpi)
    print(f"figure saved to: " + plotfile)
    plt.show()
    # # plt.show()
    # plt.close()


def draw_arrow(ax, A, B, **kwargs):
    ax.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],**kwargs)


def plot_bands(ax, bandsup, bandsdwn, x, fermi):
    xaxis = [min(x),max(x)]
    color = 'black'
    lw = 0.6
    for i, band in enumerate(bandsdwn):
        label='spin up' #r'$\downarrow$'
        ax.plot(x, np.array(band)-fermi, color='red', lw=0.39, linestyle='--', label=label if not i else None)
    for i, band in enumerate(bandsup):
        label='spin down'#r'$\uparrow$'
        ax.plot(x, np.array(band)-fermi, color=color, lw=0.4, linestyle='-', label=label if not i else None)
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
    y = dos_en - efermi

    ax.plot(dosup, y, color='black', linewidth=0.6)
    ax.plot(-dosdwn, y, color='black', linewidth=0.6)
    #ax.set_xlabel("$E (eV) - E_{Fermi}$")
    ax.fill_betweenx(y, dosup, 0*dosup, color='silver', alpha=1.0, where=(y < 0))#ax.set_xlabel("$E (eV) - E_{Fermi}$")
    ax.fill_betweenx(y, -dosdwn, 0*dosdwn, color='silver', alpha=1.0, where=(y < 0))#ax.set_xlabel("$E (eV) - E_{Fermi}$")
    if ylims is not None:
        where = np.logical_and(y > ylims[0], y < ylims[1])
    else:
        where = np.ones_like(y, dtype=bool)
    # MARK((efermi, ylims, y, where), n='where')
    ax.set_xlim(-np.amax(dosdwn[where])*1.1, np.amax(dosup[where])*1.1)
    ax.axvline(0, color='k', ls="solid", alpha=1.0, lw=0.5)
    #ax.axhline(0, color='#66ccff', ls="solid", alpha=0.5, lw=1.2)
    #ax.set_xlabel("DOS")
    #ax.title("C16 DOS")


def plot_band_dos_data(data: BandData, ev_lims=None, plotsize=(7,5), do_dos=True):
    fig = plt.figure(figsize=plotsize)
    if do_dos:
        gs = fig.add_gridspec(1, 10, wspace=0, hspace=0)
        bandsax = fig.add_subplot(gs[0:7])
        dosax = fig.add_subplot(gs[7:], sharey=bandsax)
    else:
        bandsax = fig.add_subplot()

    efermi = data.e_fermi
    r = data.band_k
    symcoords = data.symcoords
    bandsup = data.bands
    bandsdwn = data.bands2
    e_dos = data.dos_e
    dosup = data.dos
    dosdwn = data.dos2
    symm_names = data.sym_labels

    # efermi, bandsdata, dosdata = read_data(prefix)
    # (r, symcoords, bandsup, bandsdwn) = bandsdata
    # (e_dos, dosup, dosdwn) = dosdata
    # assert len(symcoords) == len(symm_names)

    # segment, knum, x, efermi, bands = read_bandfile(bandf)
    # symcoords = get_symcoords(segment, x)

    # fermi = read_fermi(scff)
    fermi = efermi
    print(f"fermi energy: {fermi}")

    #    bands, x = read_bnd(bandf)
    plot_bands(bandsax, bandsup, bandsdwn, r, fermi)

    _, _, bg_info = data.get_bandgap_info()
    print(f"Band gap: " + f"{bg_info['bandgap']} eV (type: {bg_info['bg_type']})" if not bg_info['is_metallic'] else '[metallic]')

    if not bg_info['is_metallic']:
        vb, cb = bg_info['vb']-efermi, bg_info['cb']-efermi
        vbm_x, vbm_y = bg_info['k_vbm'], bg_info['vbm'] - efermi
        cbm_x, cbm_y = bg_info['k_cbm'], bg_info['cbm'] - efermi
        bandsax.fill_between(r, vb, cb, color='k', alpha=0.15)
        # bandsax.scatter(vbm_x, vbm_y, c='gray', marker=7, alpha=1.0)#, label="VBM")
        # bandsax.scatter(cbm_x, cbm_y, c='gray', marker=6, alpha=1.0)#, label="CBM")


    plot_symmetries(bandsax, symcoords, symm_names)
    if do_dos:
        plot_dos(dosax, e_dos*HA2EV, dosup, dosdwn, fermi, ylims=ev_lims)
        dosax.set_ylabel("DOS")
        dosax.set_axis_off()
    # dosax.axes.get_yaxis().set_visible(False)
    # dosax.xaxis.tick_top()
    # bandsax.set_title(name + " band structure and DOS")

    # fig.text(0.1, 0.9, f"Fermi energy: {fermi:.5f} eV\n"
	    #  f"Bandgap: N/A")# {bandgap:.5f} eV")

    if ev_lims is not None:
        bandsax.set_ylim(*ev_lims)
    else:
        bandsax.set_ylim(-7-efermi, -2-efermi)
    # fig.suptitle(prefix.upper())

    plt.tight_layout()

    return fig, bandsax, dosax if do_dos else None

main()
