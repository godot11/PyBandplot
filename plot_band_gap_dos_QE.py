import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import os
import re

matplotlib.use("svg")

typee="c14b2a"
scf_folder = "../../scf_hires/outputs"
bands_folder = "../bands"
dos_folder = "."
plotfile = f"{typee}_bands.svg"
bandlines = 4

symmetries = (0.0000, 0.70707, 1.0607, 1.5607, 2.1730, 2.9230, 3.1730, 3.4230, 3.6730)
symm_names = (r'$\Gamma$', '$X$', '$W$', '$L$', '$\Gamma$', '$K$', '$W$', '$U$', '$X$')

fermi_offset = 0

scff = os.path.join(scf_folder, f'{typee}.in.out')
bandf = os.path.join(bands_folder, f'{typee}.bandout')
dosfile=os.path.join(dos_folder, f'{typee}.dosout')

plotsize=(12,8)
fig_dpi=600

def dist(a,b):
    # calculate the distance between vector a and b
    d = ((a[0]-b[0]) ** 2 + (a[1]-b[1]) ** 2 + (a[2]-b[2]) ** 2)**0.5
    return d

def read_fermi(file_name):
    # Read the fermi energy in scf.out
    fermi = 0
    with open(file_name, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "the Fermi energy" in line:
            fermi = float(line.split()[4])
    
    return fermi

def read_bnd(file_name):
    # Read the bands in Band.dat
    coord_regex = r"^\s+(.\d+.\d+)\s+(.\d+.\d+)\s+(.\d+.\d+)$"
    kvecs = []
    bands = [] 

    with open(file_name, "r") as f:
        lines = f.readlines()

    for i in range(len(lines)):
        line = lines[i]
        match = re.match(coord_regex,line)
        if match:
            k = np.asarray((match.group(1), match.group(2), match.group(3)), dtype=np.float, order="C")
            kvecs.append(k)
            #x_coord.append([float(match.group(1)), float(match.group(2)), float(match.group(3)) ])
            bandvals = ""
            for ii in range(1, bandlines+1):
                bandvals += lines[i+ii]
            #bandvals = bandvals.split()
            band = np.asarray(bandvals.split(), dtype=np.float, order='C')
            bands.append(band)
            
            #for j in range(len(bandvals)):
            #    if j not in bands.keys():
            #        bands[j] = []
            #    bands[j].append(float(bandvals[j]))
    bands = np.array(bands).transpose()
    x = [0]
    for i in range(1, len(kvecs)):
        delta_k = kvecs[i] - kvecs[i-1]
        delta = np.sqrt(np.sum(delta_k**2))
        x.append( x[-1] + delta)
    return bands, np.array(x)

def plot_bands(ax, bands, x, fermi):
    xaxis = [min(x),max(x)]
    for band in bands: 
        ax.plot(x, np.array(band)-fermi, color='black', lw=0.2)
    ax.axhline(0, color="#66ccff",ls="solid", alpha = 0.5,lw = 1.2)
    ax.set_ylabel(r'$E - E_{Fermi}$')
    ax.set_xlim(xaxis)

def plot_symmetries(ax, xcoords, names):
    for x in xcoords:
        ax.axvline(x, linestyle='--', linewidth=1, alpha=0.7, color='darkred')
    ax.set_xticks(xcoords)
    ax.set_xticklabels(names)

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
    ax.scatter(x_max_below, max_below-fermi, c='gray', marker=6, alpha=0.7, label="bandgap low bound")
    ax.scatter(x_min_above, min_above-fermi, c='gray', marker=7, alpha=0.7, label="bandgap high bound")
    return min_above - max_below

def plot_dos(ax, dosfile, efermi):
    data = np.loadtxt(dosfile)
    energy, dos, dos_int = data.transpose()
    y = energy-efermi
    ax.plot(dos, y, color='black', linewidth=0.6)
    #ax.set_xlabel("$E (eV) - E_{Fermi}$")
    ax.fill_betweenx(y, dos, 0*dos, color='black', alpha=0.4, where=(y < 0))
    ax.set_xlim(0, np.amax(dos)*1.1)
    #ax.axhline(0, color='#66ccff', ls="solid", alpha=0.5, lw=1.2)
    #ax.set_xlabel("DOS")
    #ax.title("C16 DOS")

def main():
    fig = plt.figure(figsize=plotsize)
    gs = fig.add_gridspec(1, 5, wspace=0, hspace=0)
    bandsax = fig.add_subplot(gs[0:4])
    dosax = fig.add_subplot(gs[4], sharey=bandsax)

    fermi = read_fermi(scff)
    print(f"fermi energy: {fermi}")
     
    bands, x = read_bnd(bandf)
    plot_bands(bandsax, bands, x, fermi)
    
    #bandgap = fermi_bandgap(bandsax, bands, x, fermi, fermi_offset)
    #print(f"Fermi bandgap: {bandgap}")
    
    plot_symmetries(bandsax, symmetries, symm_names)
    
    plot_dos(dosax, dosfile, fermi)
    dosax.set_ylabel("")
    dosax.set_axis_off()
    bandsax.set_title(typee + " band structure and DOS")

    fig.text(0.1, 0.9, f"Fermi energy: {fermi:.5f} eV\n"
	     f"Bandgap: N/A")# {bandgap:.5f} eV")

    plt.savefig(plotfile, dpi=fig_dpi)
    print(f"figure saved to: " + plotfile)

main()
