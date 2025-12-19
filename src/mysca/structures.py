"""Protein structures

"""

import numpy as np
import math
import warnings
from collections import defaultdict
from Bio.PDB import PDBParser, PPBuilder, Superimposer, is_aa
from Bio.PDB.Structure import Structure
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Residue import Residue
from Bio.Data.PDBData import protein_letters_3to1


def rgyrate(
        structure: Structure, 
        chain_id=None, 
        atom_selector=lambda a: True
):
    """Compute the radius of gyration from a Biopython Structure object.

    Parameters:
        structure (Structure): Protein structure.
        chain_id (str or None): If specified, restrict to a single chain.
        atom_selector (function): Function to filter atoms (default: all atoms).

    Returns:
        (float) Radius of gyration in angstroms.
    """
    x = []
    mass = []
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.id != chain_id:
                continue
            for residue in chain:
                for atom in residue:
                    if atom_selector(atom):
                        x.append(atom.coord)
                        mass.append(atom.mass)
    x = np.array(x)
    mass = np.array(mass)

    if len(x) == 0:
        raise ValueError("No atoms selected.")

    xm = [(m*i,m*j,m*k) for (i,j,k),m in zip(x, mass)]
    tmass = sum(mass)
    rr = sum(mi*i+mj*j+mk*k for (i,j,k),(mi,mj,mk) in zip(x, xm))
    mm = sum((sum(i)/tmass)**2 for i in zip(*xm))
    rg = np.sqrt(rr/tmass - mm)
    return rg


def classify_secondary_structure(phi: float, psi: float) -> str:
    """Classify residue as helix, sheet, or coil based on dihedral angles
    
    Args:
        phi (float): angle in radians.
        psi (float): angle in radians.

    Returns:
        (str) Classification. Either helix, sheet, or coil.
    """
    if phi is None or psi is None:
        return 'coil'
    
    phi_deg = math.degrees(phi)
    psi_deg = math.degrees(psi)
    
    # Alpha-helix range (-57,-47)
    if (-90 < phi_deg < -30) and (-77 < psi_deg < -17):
        return 'helix'
    # Beta-sheet range (-119,113)
    elif (-180 < phi_deg < -40) and (90 < psi_deg < 180) or (-180 < phi_deg < -40) and (-180 < psi_deg < -100):
        return 'sheet'
    else:
        return 'coil'
    

def get_secondary_structure_proportions(
        structure: Structure
) -> dict[str: float]:
    """Calculate secondary structure proportions using dihedral angles.
    
    Args:
        structure (Structure): Protein structure.
    
    Returns:
        (dict[str: float]) Map from key (helix, sheet, coil) to proportion.
    """
    ppb = PPBuilder()
    
    secondary_structure = defaultdict(int)
    total_residues = 0
    
    for pp in ppb.build_peptides(structure):
        # Get phi/psi angles for each residue
        angles = pp.get_phi_psi_list()
        for i, residue in enumerate(pp):
            total_residues += 1
            phi, psi = angles[i]
            ss_type = classify_secondary_structure(phi, psi)
            secondary_structure[ss_type] += 1
    
    if total_residues > 0:
        return {
            'helix': secondary_structure['helix'] / total_residues,
            'sheet': secondary_structure['sheet'] / total_residues,
            'coil': secondary_structure['coil'] / total_residues
        }
    return {'helix': 0., 'sheet': 0., 'coil': 0.}


def compute_sasa(structure: Structure, level: str = "S") -> float:
    """Compute surface accessibility surface area (SASA) using Shrake-Rupley.
    
    Args:
        structure (Structure): Protein structure.
        level (str): level at which to compute the SASA. Defaults to "S".

    Returns:
        (float) Average SASA value at the indicated level.
    """
    sr = ShrakeRupley()
    sr.compute(structure, level=level)
    return structure.sasa


def analyze_sequence(structure: Structure) -> ProteinAnalysis:
    """Wrapper for ProteinAnalysis to perform analysis given structure.

    Args:
        structure (Structure): Protein structure.

    Returns:
        (ProteinAnalysis) Analyzed sequence object.
    """
    analyzed_seq = ProteinAnalysis(struct2seq(structure))
    return analyzed_seq


def struct2seq(structure: Structure) -> str:
    """Convert a protein structure into its amino acid sequence.
    
    Args:
        structure (Structure): Protein structure.

    Returns:
        (str) amino acid sequence of the protein structure.
    """
    ppb = PPBuilder()
    for pp in ppb.build_peptides(structure):
        protein_seq = pp.get_sequence()  # returns a Bio.Seq.Seq object
        break
    return str(protein_seq)


def get_ca_atoms(structure):
    """Return the C-alpha atoms in the given protein structure.

    TODO: Test

    Args:
        structure (Structure): Protein structure.

    Returns:
        (list[Residue]) List of C-alpha residues of the given structure.
    """
    warnings.warn("This function isn't tested and may be buggy!")
    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue) and "CA" in residue:
                    ca_atoms.append(residue["CA"])
        break  # only use first model
    
    return ca_atoms


def align_atoms_in_structures(
        structure1, 
        structure2, 
        atoms1,
        atoms2,
):
    """Align structure1 to structure2.

    TODO: Test

    Args:
        structure1 (Structure): Protein structure 1.
        structure2 (Structure): Protein structure 2.

    Returns:
        (float): rmsd
    """
    warnings.warn("This function isn't tested and may be buggy!")
    sup = Superimposer()
    sup.set_atoms(atoms2, atoms1)
    sup.apply(structure1.get_atoms())
    return sup.rms
    

def extract_residue_sequence(structure: Structure) -> tuple[list[Residue], str]:
    """Get amino acid sequence and corresponding residues from a PDB structure.

    This function iterates over the first model and all chains in a Biopython 
    structure, collects all standard amino acid residues that contain a C-alpha 
    atom, converts their three-letter residue names to one-letter codes, and 
    returns both the residue list and the concatenated sequence string.

    TODO: Test

    Args:
        structure (Bio.PDB.Structure.Structure): A Biopython Structure object
            parsed from a PDB file.

    Returns:
        residues (list[Residue]): List of amino acid residues with C-alpha atoms.
        seq (str): The corresponding one-letter amino acid sequence.

    Notes:
        - Non-standard residues are skipped.
        - Only the first model of the structure is used.
        - Residues without a C-alpha atom are ignored.
    """
    warnings.warn("This function isn't tested and may be buggy!")
    seq = ""
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue) and "CA" in residue:
                    try:
                        seq += protein_letters_3to1[residue.get_resname()]
                        residues.append(residue)
                    except KeyError:
                        pass  # Skip non-standard residues
        break
    return residues, seq
