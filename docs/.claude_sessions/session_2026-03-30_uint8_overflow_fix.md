# Session: Fix uint8 Overflow in Sparse One-Hot Dot Product

**Date:** 2026-03-30

## Bug

`get_onehotmsa_sparse` in `src/mysca/preprocess.py` created the sparse CSR
matrix with `dtype=np.uint8`. When scipy computes `msa_sparse @ msa_sparse.T`,
it accumulates into the same dtype. Since `uint8` overflows at 255, any MSA
with more than 255 positions produces silently wrong similarity counts.

Example: two identical 300-position sequences yield a dot product of 44
(300 % 256) instead of 300.

## Impact

Affects weight computation methods **v4, v5, and v6**, all of which use the
sparse dot product path. These are the default and recommended methods.

**Not affected:**
- v3 — uses dense `int16` arrays with direct integer comparison
- torch — uses `==` comparison on `int16` tensors, not a sparse dot product

Wrong similarity counts cause the sequence similarity threshold comparison
(`counts >= thresh`) to fail, producing incorrect neighbor counts and therefore
incorrect sequence weights. For npos > 255, most or all neighbors can fall
below the threshold when they shouldn't, giving every sequence weight 1.0
instead of the correct 1/cluster_size.

## Fix

Changed one line in `src/mysca/preprocess.py:564`:

```python
# Before (buggy)
data = np.ones(cols.shape[0], dtype=np.uint8)

# After (fixed)
data = np.ones(cols.shape[0], dtype=np.int16)
```

`int16` supports values up to 32,767, which is more than sufficient for any
protein MSA length.

## Regression Tests

New file: `tests/test_sparse_overflow.py` (8 tests)

Direct dot product tests:
- `test_dot_product_no_overflow` — parametrized at npos = 100, 256, 300, 500
- `test_dot_product_with_gaps` — verifies gaps excluded from counts (npos=300)
- `test_dot_product_partial_match` — two partially different sequences (npos=300)

End-to-end weight computation tests:
- `test_weights_correct_for_long_alignment` — parametrized for v4 and v5;
  5 identical 300-position sequences should all get weight 1/5

All 8 tests fail with `uint8`, all pass with `int16`.

## Verification

All 351 tests pass (343 existing + 8 new).
