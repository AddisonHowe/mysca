"""Helper function tests

"""

import pytest
from contextlib import nullcontext as does_not_raise
from tests.conftest import DATDIR, TMPDIR, remove_dir

import numpy as np

from mysca.io import msa_from_aligned_seqs
from mysca.mappings import SymMap
from mysca.helpers import get_top_k_conserved_retained_positions
from mysca.helpers import map_msa_positions_to_sequence
from mysca.helpers import get_rawseq_indices_of_msa
from mysca.helpers import get_conserved_rawseq_positions
from mysca.helpers import get_rawseq_positions_in_groups
from mysca.helpers import get_group_rawseq_positions_by_entry



#####################
##  Configuration  ##
#####################

TEST_MSA5 = f"{DATDIR}/msas/msa05.faa"
TEST_MSA6 = f"{DATDIR}/msas/msa06.faa"
TEST_MSA7 = f"{DATDIR}/msas/msa07.faa"

SYMMAP5 = SymMap("ABCDEFGH", '-')
BACKGROUND_MAP5 = {
    "A": 1/8,
    "B": 1/8,
    "C": 1/8,
    "D": 1/8,
    "E": 1/8,
    "F": 1/8,
    "G": 1/8,
    "H": 1/8,
}

SYMMAP6 = SymMap("ABCD", '-')
BACKGROUND_MAP6 = {
    "A": 0.25,
    "B": 0.25,
    "C": 0.25,
    "D": 0.25,
}

SYMMAP7 = SymMap("ACDE", '-')
BACKGROUND_MAP7 = {
    "A": 0.25,
    "C": 0.25,
    "D": 0.25,
    "E": 0.25,
}


LAMBDA1 = 0.03

###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################


@pytest.mark.parametrize(
        "retained_positions, Di, k, topidxs_exp, topvals_exp", [
    [
        [0,1,3,4,5], [0.1,0.2,0.2,0.3,0.0], 3, 
        [4, 3, 1], [0.3, 0.2, 0.2],
    ],
])
def test_get_top_k_conserved_retained_positions(
        retained_positions, Di, k, topidxs_exp, topvals_exp
):
    topidxs, topvals = get_top_k_conserved_retained_positions(
        retained_positions, Di, k,
    )
    errors = []
    if np.any(np.array(topidxs) != np.array(topidxs_exp)):
        msg = f"topidxs do not match.\n"
        msg += f"Expected: {topidxs_exp}\n"
        msg += f"     Got: {topidxs}"
        errors.append(msg)
    if np.any(np.array(topvals) != np.array(topvals_exp)):
        msg = f"topvals do not match.\n"
        msg += f"Expected: {topvals_exp}\n"
        msg += f"     Got: {topvals}"
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
    

@pytest.mark.parametrize('msa_seq, raw_seq, raise_context, msa2seq_exp', [
    ["--ABC--DEF", "ABCDEF", does_not_raise(),
     {2:0,3:1,4:2,7:3,8:4,9:5,}
    ],
])
def test_map_msa_positions_to_sequence(
    msa_seq, raw_seq, msa2seq_exp, raise_context
):    
    with raise_context:
        msa2seq, seq2msa = map_msa_positions_to_sequence(msa_seq, raw_seq)
        seq2msa_exp = {v: k for k, v in msa2seq_exp.items()}
        errors = []
        if msa2seq_exp != msa2seq:
            msg = f"Incorrect map msa2seq:\n"
            msg += f"Expected: {msa2seq_exp}\n"
            msg += f"Expected: {msa2seq}"
            errors.append(msg)
        if seq2msa_exp != seq2msa:
            msg = f"Incorrect map seq2msa:\n"
            msg += f"Expected: {seq2msa_exp}\n"
            msg += f"Expected: {seq2msa}"
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.parametrize('seqs_aligned, arr_exp', [
    [
        ["--ABC--DEF",
         "ABCDEFGHIJ",
         "ABCDE-----",],
        np.array([
            [-1,-1,0,1,2,-1,-1,3,4,5],
            [0,1,2,3,4,5,6,7,8,9],
            [0,1,2,3,4,-1,-1,-1,-1,-1],
        ])
    ],
    [
        ["-A-B---D",
         "--CCCC--",
         "DD-----D",],
        np.array([
            [-1,0,-1,1,-1,-1,-1,2],
            [-1,-1,0,1,2,3,-1,-1],
            [0,1,-1,-1,-1,-1,-1,2],
        ])
    ]
])
def test_get_rawseq_indices_of_msa(
        seqs_aligned,
        arr_exp
):
    msa_obj = msa_from_aligned_seqs(seqs_aligned)

    arr = get_rawseq_indices_of_msa(
        msa_obj,
    )
    errors = []
    if not np.all(arr_exp == arr):
        msg = f"message"
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.parametrize(
        "seqs_aligned, retained_sequences, " \
        "conserved_msa_positions, conserved_rawseq_idxs_exp", [
    [
        ["A--DEF",
         "-ABC-D",
         "ABCDEF",],
        [0, 1, 2],
        [0, 3, 4],
        [[0, 1, 2],
         [-1, 2, -1],
         [0, 3, 4]]
    ],
    [
        ["A--DEF",
         "-ABC-D",
         "ABCDEF",],
        [0, 2],
        [0, 3, 4],
        [[0, 1, 2],
         [0, 3, 4]]
    ],
])
def test_get_conserved_rawseq_positions(
        seqs_aligned, retained_sequences,
        conserved_msa_positions, conserved_rawseq_idxs_exp,
):
    conserved_rawseq_idxs_exp = np.array(conserved_rawseq_idxs_exp)
    msa_obj = msa_from_aligned_seqs(seqs_aligned)
    conserved_rawseq_idxs = get_conserved_rawseq_positions(
        msa_obj, 
        np.array(retained_sequences),
        np.array(conserved_msa_positions),
    )

    errors = []
    if not np.all(conserved_rawseq_idxs_exp == conserved_rawseq_idxs):
        msg = f"Mismatch in conserved_rawseq_idxs.\n"
        msg += f"Expected:\n{conserved_rawseq_idxs_exp}\n"
        msg += f"Got:\n{conserved_rawseq_idxs}\n"
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.parametrize(
        "seqs_aligned, retained_sequences, retained_positions, " \
        "groups, positions_exp", [
    [
        ["A--DEF",
         "-ABC-D",
         "ABCDEF",],
        [0, 1, 2],
        [0, 1, 3, 4],
        [[0, 2],[3],[1, 3]],  # groups
        [[[0, 1], [2], [2]], 
         [[2], [], [0]], 
         [[0, 3], [4], [1, 4]]],
    ],
    [
        ["A--DEF",
         "-ABC-D",
         "ABCDEF",],
        [0, 2],
        [0, 1, 3, 4],
        [[0, 2],[3],[1, 3]],  # groups
        [[[0, 1], [2], [2]], 
         [[0, 3], [4], [1, 4]]],
    ],
])
def test_get_rawseq_positions_in_groups(
        seqs_aligned, retained_sequences, retained_positions, 
        groups, positions_exp
):
    msa_obj = msa_from_aligned_seqs(seqs_aligned)
    rawseq_idxs = get_rawseq_indices_of_msa(msa_obj)
    rawseq_idxs = rawseq_idxs[retained_sequences,:]
    rawseq_idxs = rawseq_idxs[:, retained_positions]
    nseqs = len(retained_sequences)

    positions = get_rawseq_positions_in_groups(rawseq_idxs, groups)
    
    errors = []
    for groupidx in range(len(groups)):
        for seqidx in range(nseqs):
            g = np.array(positions[seqidx][groupidx])
            g_exp = np.array(positions_exp[seqidx][groupidx])
            if not np.all(g == g_exp):
                msg = f"Mismatch in positions for sequence {seqidx}, group {groupidx}.\n"
                msg += f"Expected:\n{g_exp}\n"
                msg += f"Got:\n{g}\n"
                errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
    

@pytest.mark.parametrize(
        "seqs_aligned, retained_sequences, retained_positions, " \
        "groups, groups_by_entry_exp", [
    [
        ["A--DEF",
         "-ABC-D",
         "ABCDEF",],
        [0, 1, 2],
        [0, 1, 3, 4],
        [[0, 2],[3],[1, 3]],  # groups
        {
            "sequence0": [
                np.array([0, 1]),
                np.array([2]),
                np.array([2]),
            ],
            "sequence1": [
                np.array([2]),
                np.array([]),
                np.array([0]),
            ],
            "sequence2": [
                np.array([0, 3]),
                np.array([4]),
                np.array([1, 4]),
            ],
        }
    ],
    [
        ["A--DEF",
         "-ABC-D",
         "ABCDEF",],
        [0, 2],
        [0, 1, 3, 4],
        [[0, 2],[3],[1, 3]],  # groups
        {
            "sequence0": [
                np.array([0, 1]),
                np.array([2]),
                np.array([2]),
            ],
            "sequence2": [
                np.array([0, 3]),
                np.array([4]),
                np.array([1, 4]),
            ],
        }
    ],
])
def test_get_group_rawseq_positions_by_entry(
        seqs_aligned, retained_sequences, retained_positions,
        groups, groups_by_entry_exp
):
    msa_obj = msa_from_aligned_seqs(seqs_aligned)
    rawseq_idxs = get_rawseq_indices_of_msa(msa_obj)
    rawseq_idxs = rawseq_idxs[retained_sequences,:]
    rawseq_idxs = rawseq_idxs[:, retained_positions]
    
    group_rawseq_positions = get_rawseq_positions_in_groups(rawseq_idxs, groups)

    groups_by_entry = get_group_rawseq_positions_by_entry(
        msa_obj, retained_sequences, groups, group_rawseq_positions
    )
    print(groups_by_entry_exp)
    print(groups_by_entry)

    errors = []
    for key in groups_by_entry_exp:
        gexp = groups_by_entry_exp[key]
        g = groups_by_entry[key]
        print(f"Expected:\n{gexp}")
        print(f"Got:\n{g}")
        for gidx in range(max(len(gexp), len(g))):
            if not np.allclose(g[gidx], gexp[gidx]):
                msg = f"Mismatch in entry {key}, gidx {gidx}.\n"
                msg += f"Expected:\n{gexp[gidx]}\n"
                msg += f"Got:\n{g[gidx]}\n"
                errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
    