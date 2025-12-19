"""Sector analysis script

"""

import os, sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA

HYDROPHOBIC = {'A', 'V', 'L', 'I', 'M', 'F', 'W', 'P', 'G', 'C'}
HYDROPHILIC = {'S', 'T', 'N', 'Q', 'Y', 'D', 'E', 'K', 'R', 'H'}

HYDROPHOBICITY = {
    'I':  4.5,   'V':  4.2,   'L':  3.8,   'F':  2.8,   'C':  2.5,
    'M':  1.9,   'A':  1.8,   'G': -0.4,   'T': -0.7,   'S': -0.8,
    'W': -0.9,   'Y': -1.3,   'P': -1.6,   'H': -3.2,   'E': -3.5,
    'Q': -3.5,   'D': -3.5,   'N': -3.5,   'K': -3.9,   'R': -4.5,
}

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--sca_results_dir", type=str, required=True)
    parser.add_argument("-gf", "--groups_fpath", type=str, required=True)
    parser.add_argument("-vf", "--values_fpath", type=str, required=True)
    parser.add_argument("-s", "--sector_dir", type=str, required=True)
    parser.add_argument("-o", "--outdir", type=str, required=True)
    parser.add_argument("-si", "--sector_idxs", type=int, nargs="+", default=0)
    parser.add_argument("--binarize", action="store_true")
    parser.add_argument("--hydro", action="store_true")
    return parser.parse_args(args)


def main(args):    
    # Process command line args
    sca_results_dir = args.sca_results_dir
    groups_fpath = args.groups_fpath
    values_fpath = args.values_fpath
    sector_dir = args.sector_dir
    sector_idxs = args.sector_idxs
    outdir = args.outdir
    binarize = args.binarize
    hydro = args.hydro

    # Load the MSA, retained sequences, retained positions, and sequence IDs
    msa_fname = "msa.npy"
    retained_sequences_fname = "retained_sequences.npy"
    retained_positions_fname = "retained_positions.npy"
    seqids_fname = "retained_sequence_ids.npy"
    sym2int_fname = "sym2int.json"
    msa = np.load(
        os.path.join(sca_results_dir, msa_fname)
    )
    retained_sequences = np.load(
        os.path.join(sca_results_dir, retained_sequences_fname)
    )
    retained_positions = np.load(
        os.path.join(sca_results_dir, retained_positions_fname)
    )
    retained_seqids = np.load(
        os.path.join(sca_results_dir, seqids_fname)
    )
    with open(os.path.join(sca_results_dir, sym2int_fname), "r") as f:
        sym2int = json.load(f)

    print(f"Loaded MSA of shape {msa.shape}")
    print(f"Loaded {len(retained_sequences)} sequence indices")
    print(f"Loaded {len(retained_positions)} position indices")
    print(f"Loaded {len(retained_seqids)} retained sequence IDs")
    assert msa.shape == (len(retained_sequences), len(retained_positions)), \
        f"Mismatch between MSA shape and number of retained sequences/positions!"

    # Load grouping information
    df = pd.read_csv(groups_fpath, header=None, names=["seqid", "group"], sep=" ")
    print(f"Loaded group assignments for {len(df)} sequences.")
    # Load parameter information and append to dataframe
    df["parameter_value"] = np.genfromtxt(values_fpath)

    # Map sequence ID to group assignment and check if the sequence is retained
    id_to_group = {}
    id_to_retainment = {}
    for _, row in df.iterrows():
        seqid = row["seqid"]
        id_to_group[seqid] = int(row["group"])
        id_to_retainment[seqid] = seqid in retained_seqids
        if seqid not in retained_seqids:
            print(f"Sequence {seqid} was not retained in MSA preprocessing!!!")

    # List the contents of the directory defining sectors by MSA positions
    sector_files = os.listdir(sector_dir)
    if sector_idxs[0] == 0:
        sector_idxs = [i for i in range(len(sector_files))]

    # Subset the MSA to the sequences of interest
    retained_sequences_screen = np.isin(retained_seqids, df["seqid"].values)
    msa = msa[retained_sequences_screen,:]
    retained_seqids = retained_seqids[retained_sequences_screen]
    assert msa.shape == (len(df), msa.shape[1]), "MSA shape incorrect!"

    # Reorder the dataframe to match the MSA order
    df = df.set_index("seqid").loc[retained_seqids].reset_index()
    assert np.all(df["seqid"].values == retained_seqids)

    sector_results = {}
    os.makedirs(outdir, exist_ok=True)
    combined_sectors = [[1, 2], [3, 4]]
    sector_idx_sets = [[i] for i in sector_idxs] + combined_sectors
    for sector_idx_set in sector_idx_sets:
        print(f"Processing Sector(s): {sector_idx_set}")
        sector_idxs_str = "".join([str(i) for i in sector_idx_set])
        sector_arrs = []
        for sector_idx in sector_idx_set:
            sector_fname = f"group_{sector_idx}_msapos.npy"
            sector_fpath = os.path.join(sector_dir, sector_fname)
            if not os.path.isfile(sector_fpath):
                raise FileNotFoundError(f"File {sector_fpath} does not exist!")
            # Get MSA positions defining the sector
            sector_arrs.append(np.load(sector_fpath))
        sector_positions = np.concatenate(sector_arrs, axis=0)

        # Subset the MSA to the sector(s) of interest
        sector_msa = msa[:, sector_positions]

        # Binarize the MSA if desired
        if binarize:
            print("Normalizing MSA (binary hydrophobic/philic scale)")
            aa_to_binary = {sym2int[aa]: 0 for aa in HYDROPHOBIC}
            aa_to_binary.update({sym2int[aa]: 1 for aa in HYDROPHILIC})
            aa_to_binary[sym2int['-']] = 0.5
            sector_msa = np.array([
                [aa_to_binary[x] for x in arr] for arr in sector_msa
            ])
        elif hydro:
            print("Normalizing MSA (hydrophobic/philic scale)")
            aa_to_value = {
                sym2int[aa]: HYDROPHOBICITY[aa] for aa in HYDROPHOBICITY
            }
            aa_to_value[sym2int['-']] = 0.5
            sector_msa = np.array([
                [aa_to_value[x] for x in arr] for arr in sector_msa
            ])
        else:
            print("No MSA normalization (raw sequence data)")
        
        nuniq = len(np.unique(sector_msa, axis=0))
        print("{}/{} unique length-{} sequences (after normalization)".format(
            nuniq, sector_msa.shape[0], sector_msa.shape[1]
        ))

        # Apply PCA to the sector
        data_pca, pcs, exp_var_ratio = apply_pca(
            sector_msa, n_components=10
        )
        sector_results[sector_idx] = {}
        sector_results[sector_idx]["msa_pc"] = data_pca
        sector_results[sector_idx]["components"] = pcs
        sector_results[sector_idx]["exp_var_ratio"] = exp_var_ratio

        # Plot PCA explained variance
        fig, ax = plt.subplots(1, 1)
        ax.plot(
            1 + np.arange(len(exp_var_ratio)), np.cumsum(exp_var_ratio),
            ".-"
        )
        ax.set_xlim(0, 1 + len(exp_var_ratio))
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(f"PC")
        ax.set_ylabel(f"exp var")
        ax.set_title(f"Cumulative variance explained")
        saveas = f"expvar_sector{sector_idxs_str}"
        plt.savefig(f"{outdir}/{saveas}.png")
        plt.close()
        
        # Plot MSA in PCA coordinates
        fig, ax = plt.subplots(1, 1)
        ax.plot(data_pca[:,0], data_pca[:,1], "k.", alpha=0.6)
        ax.set_xlabel(f"PC1")
        ax.set_ylabel(f"PC2")
        ax.set_title(f"PCA Sector {"&".join(sector_idxs_str)}")
        saveas = f"pc1pc2_sector{sector_idxs_str}"
        plt.savefig(f"{outdir}/{saveas}.png")
        plt.close()

        # Plot MSA in PCA coordinates colored by group
        fig, ax = plt.subplots(1, 1)
        ax.plot(data_pca[:,0], data_pca[:,1], "k.", alpha=0.6)
        sc = ax.scatter(
            add_noise(data_pca[:,0]), add_noise(data_pca[:,1]),
            c=df["group"],
            alpha=0.5,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig = ax.figure
        cbar = fig.colorbar(sc, cax=cax)
        cbar.ax.set_title("Group")
        ax.set_xlabel(f"PC1")
        ax.set_ylabel(f"PC2")
        ax.set_title(f"PCA Sector {"&".join(sector_idxs_str)}")
        saveas = f"pc1pc2_bygroup_sector{sector_idxs_str}"
        plt.savefig(f"{outdir}/{saveas}.png")
        plt.close()

        # Plot MSA in PCA coordinates colored by parameter value
        fig, ax = plt.subplots(1, 1)
        ax.plot(data_pca[:,0], data_pca[:,1], "k.", alpha=0.6)
        sc = ax.scatter(
            add_noise(data_pca[:,0]), add_noise(data_pca[:,1]),
            c=df["parameter_value"],
            alpha=0.5,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig = ax.figure
        cbar = fig.colorbar(sc, cax=cax)
        cbar.ax.set_title("$r_A$")
        ax.set_xlabel(f"PC1")
        ax.set_ylabel(f"PC2")
        ax.set_title(f"PCA Sector {"&".join(sector_idxs_str)}")
        saveas = f"pc1pc2_byparam_sector{sector_idxs_str}"
        plt.savefig(f"{outdir}/{saveas}.png")
        plt.close()
    return


def apply_pca(data, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    pcs = pca.components_
    exp_var_ratio = pca.explained_variance_ratio_
    data_pca = pca.transform(data)
    return data_pca, pcs, exp_var_ratio


def add_noise(data, scale=0.005):
    r = scale * (data.max() - data.min())
    return data + r * np.random.standard_normal(len(data))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
