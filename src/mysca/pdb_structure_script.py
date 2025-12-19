"""PDB structures analysis script
# TODO: INCOMPLETE
"""

import argparse
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tqdm as tqdm
from mpl_toolkits.mplot3d import Axes3D

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

from mysca.io import load_msa
from mysca.io import get_residue_sequence_from_pdb_structure
from mysca.io import load_pdb_structure
from mysca.mappings import SymMap
from mysca.preprocess import preprocess_msa
from mysca.preprocess import compute_background_freqs
from mysca.core import run_sca, run_ica
from mysca.helpers import get_top_k_conserved_retained_positions
from mysca.helpers import get_conserved_rawseq_positions
from mysca.helpers import get_rawseq_positions_in_groups
from mysca.helpers import get_group_rawseq_positions_by_entry
from mysca.helpers import get_rawseq_indices_of_msa

DEFAULT_BACKGROUND_FREQ = {
        'A': 0.078, 'C': 0.020, 'D': 0.053, 'E': 0.063,
        'F': 0.039, 'G': 0.072, 'H': 0.023, 'I': 0.053,
        'K': 0.059, 'L': 0.091, 'M': 0.022, 'N': 0.043,
        'P': 0.052, 'Q': 0.042, 'R': 0.051, 'S': 0.071,
        'T': 0.058, 'V': 0.066, 'W': 0.014, 'Y': 0.033,
    }


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-msa", "--msa_fpath", type=str, required=True,
                        help="Filepath of input MSA in fasta format.")
    parser.add_argument("-i", "--sca_dir", type=str, required=True,
                        help="Path to directory containing SCA results.")
    parser.add_argument("-s", "--structure_dir", type=str, required=True,
                        help="Path to directory containing structure pdb files.")
    parser.add_argument("-o", "--outdir", type=str, required=True, 
                        help="Output directory.")
    parser.add_argument("-nc", "--n_top_conserved", type=int, required=True, 
                        help="Number of top conserved residues to consider.")
    parser.add_argument("--reference", type=str, default=None, 
                        help="SCA optional reference entry in MSA")
    parser.add_argument("--pbar", action="store_true")
    parser.add_argument("-v", "--verbosity", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    
    return parser.parse_args(args)


def main(args):

    # Process command line args
    sca_dir = args.sca_dir
    struct_dir = args.structure_dir
    MSA_FPATH = args.msa_fpath
    reference_id = args.reference
    OUTDIR = args.outdir
    verbosity = args.verbosity
    n_top_conserved = args.n_top_conserved
    reference_id = args.reference
    N_BOOT = args.n_boot
    PBAR = args.pbar
    SEED = args.seed

    raise NotImplementedError()
    
    # Housekeeping
    if SEED is None or SEED <= 0:
        SEED = np.random.randint(2**32)
    rng = np.random.default_rng(seed=SEED)

    if reference_id is None or reference_id.lower() == "none":
        if verbosity:
            print("No reference entry specified.")
        reference_id = None

    # Load SCA results
    retained_sequences = np.load(f"{sca_dir}/retained_sequences.npy")
    retained_positions = np.load(f"{sca_dir}/retained_positions.npy")
    Di = np.load(f"{sca_dir}/conservation.npy")
    msa = np.load(f"{sca_dir}/msa.npy")

    IMGDIR = os.path.join(OUTDIR, "images")
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(IMGDIR, exist_ok=True)    

    # Load MSA
    msa_obj_orig, msa_orig, _, sym_map = load_msa(
        MSA_FPATH, format="fasta", 
        mapping=None,  # TODO: consider allowing for specified mapping
        verbosity=1
    )
    _, NUM_POS_ORIG = msa_orig.shape
    NSYMS = len(sym_map)
    
    # if verbosity:
    #     print(f"Loaded MSA. shape: {msa_orig.shape} (sequences x positions)")
    #     print(f"Symbols: {sym_map.aa_list}")

    # msa, xmsa, seqids, weights, fi0_pretrunc, \
    # retained_sequences, retained_positions, ref_results = preprocess_msa(
    #     msa_orig, seqids_orig, 
    #     mapping=sym_map,
    #     gap_truncation_thresh=gap_truncation_thresh,
    #     sequence_gap_thresh=sequence_gap_thresh,
    #     reference_id=reference_id,
    #     reference_similarity_thresh=reference_similarity_thresh,
    #     sequence_similarity_thresh=sequence_similarity_thresh,
    #     position_gap_thresh=position_gap_thresh,
    #     verbosity=1,
    # )

    
    # Determine the top conserved positions
    topk_conserved_msa_pos, top_conserved_Di = get_top_k_conserved_retained_positions(
        retained_positions, Di, n_top_conserved
    )

    if verbosity:
        print("top k conserved MSA positions:", topk_conserved_msa_pos)

    # Plot conservation
    fig, ax = plt.subplots(1, 1, figsize=(10,4))
    ax.plot(
        retained_positions, Di, "o",
        color="Blue",
        alpha=0.2
    )
    ax.plot(
        topk_conserved_msa_pos, top_conserved_Di, "o",
        color="Green",
        alpha=0.5
    )
    ax.set_xlim(0, NUM_POS_ORIG)
    ax.set_xlabel(f"Position")
    ax.set_ylabel("Relative Entropy (KL Divergence, $D_i$)")
    ax.set_title(f"Position-wise Conservation")
    plt.savefig(f"{IMGDIR}/positional_conservation.png")
    plt.close()

    # Map MSA positions to raw sequence positions
    rawseq_idxs = get_rawseq_indices_of_msa(msa_obj_orig)
    rawseq_idxs = rawseq_idxs[retained_sequences,:]
    rawseq_idxs = rawseq_idxs[:,retained_positions]
    
    # Determine the conserved positions in each raw sequence
    conserved_aa_idxs = get_conserved_rawseq_positions(
        msa_obj_orig, retained_sequences, topk_conserved_msa_pos
    )

    # Load PDB structures if available

    pdb_mappings = {}
    missing_pdb_entries = []
    nan_filler = np.array([np.nan, np.nan, np.nan])
    for i, seqidx in enumerate(retained_sequences):
        entry = msa_obj_orig[int(seqidx)]
        id = entry.id
        conserved_positions = conserved_aa_idxs[i]
        pdbfpath = f"{struct_dir}/{id}.pdb"
        if not os.path.isfile(pdbfpath):
            missing_pdb_entries.append(id)
            continue
        if -1 in conserved_positions:
            print(f"Entry {id} does not contain all conserved positions.")
            continue
        structure = load_pdb_structure(pdbfpath, id=id, quiet=True)
        residues = get_residue_sequence_from_pdb_structure(structure)
        conserved_residues = [
            residues[i] if i >= 0 else None for i in conserved_positions
        ]
        conserved_residue_positions = np.array(
            [nan_filler if r is None else r['CA'].coord for r in conserved_residues]
        )
        pdb_mappings[id] = conserved_residue_positions

    if len(pdb_mappings) == 0:
        print("No PDB files found!")
    else:
        # Compute pairwise distance matrix for conserved positions.
        ncombs = n_top_conserved * (n_top_conserved - 1) // 2
        all_pdists = np.nan * np.ones([len(pdb_mappings), ncombs])

        for i, id in enumerate(sorted(list(pdb_mappings.keys()))):
            x = pdb_mappings[id]
            dists = pdist(x, metric="euclidean")
            all_pdists[i] = dists

        # Plot pairwise distances between conserved residues
        fig, ax = plt.subplots(1, 1)
        sc = ax.imshow(all_pdists, cmap="plasma")
        ax.set_xlabel("pairwise distance")
        ax.set_ylabel("variant")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig = ax.figure
        cbar = fig.colorbar(sc, cax=cax)
        cbar.ax.set_ylabel("Distance (Angstroms)")
        plt.savefig(f"{IMGDIR}/conserved_residues_pdists.png")
        plt.close()

        # Normalize the pairwise distance data and plot
        all_pdists_centered = (all_pdists - all_pdists.mean(0)) / all_pdists.std(0)

        fig, ax = plt.subplots(1, 1)
        sc = ax.imshow(all_pdists_centered, cmap="plasma")
        ax.set_xlabel("pairwise distance")
        ax.set_ylabel("variant")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig = ax.figure
        cbar = fig.colorbar(sc, cax=cax)
        cbar.ax.set_ylabel("Distance (Normalized)")
        plt.savefig(f"{IMGDIR}/conserved_residues_pdists_normalized.png")
        plt.close()

        # Apply pca to pairwise distance data
        pca = PCA(n_components=min(20, ncombs, all_pdists_centered.shape[0]))
        pca.fit(all_pdists_centered)
        data_pca = pca.transform(all_pdists_centered)

        # Plot PCs 1 and 2
        fig, ax = plt.subplots(1, 1)
        ax.plot(
            data_pca[:,0], data_pca[:,1], "."
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Conserved residues pairwise distance PCA")
        plt.savefig(f"{IMGDIR}/conserved_residues_pdists_pc1_pc2.png")
        plt.close()

        fig, ax = plt.subplots(1, 1)
        ax.plot(
            1 + np.arange(len(pca.explained_variance_ratio_)), 
            np.cumsum(pca.explained_variance_ratio_)
        )
        ax.set_xlabel("PC")
        ax.set_ylabel("explained variance")
        ax.set_title("Explained variance (cumulative proportion)")
        plt.savefig(f"{IMGDIR}/conserved_residues_pdists_exp_var.png")
        plt.close()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
