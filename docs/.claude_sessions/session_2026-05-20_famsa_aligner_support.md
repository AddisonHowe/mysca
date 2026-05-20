# 2026-05-20 — Add FAMSA aligner support to `sca-prealign`

## Summary

`sca-prealign` already supported two alignment backends (MAFFT, Clustal Omega)
selected via `--align`, dispatched through `ALIGNERS` / `ALIGNER_BINARIES`
registries in [src/mysca/prealign.py](../../src/mysca/prealign.py). User asked
to add a third backend, FAMSA, useful for large/divergent protein families
because of FAMSA's fast guide-tree (single-linkage / MedoidTree) construction.

The registry-based dispatch made the code change small: one new
`_align_famsa(...)` wrapper, two registry entries. The bulk of the work was
the three documentation surfaces required by CLAUDE.md (module docstring,
argparse `help=`, CLI reference), the install-instruction surfaces (README,
environment.yml), and a test block mirroring the existing clustalo coverage.

## Behavioural decisions (user-confirmed mid-session)

- **--align_args parity with Clustal Omega**, not minimal-mafft style. Exposed
  keys: `guidetree_out=true`, `gt={sl,upgma,nj}` (default `sl`),
  `medoidtree=true`. Anything else can be passed via `--align_extra`.
- **Always pass `-keep-duplicates`** (the wrapper-level decision, not user-
  exposed). FAMSA's default behaviour deduplicates internally and restores
  duplicates after alignment, so `n_in == n_out` would hold either way, but
  passing the flag makes the wrapper's record-preservation invariant
  explicit and version-independent. Parity with MAFFT/Clustal Omega behaviour
  and with the pipeline's downstream assumptions.

## FAMSA CLI gotchas discovered during verification

FAMSA 2.4.1 (the version available on the lab cluster via
`module load famsa`) has two non-obvious CLI quirks that bit the first
implementation and forced revisions:

1. **Flag is `-keep-duplicates` (hyphen), not `-keep_duplicates` (underscore).**
   FAMSA's flag naming is inconsistent — `-gt_export`, `-medoid_threshold`,
   `-refine_mode` use underscores; `-keep-duplicates`, `-gz-lev`,
   `-remove-rare-columns` use hyphens. The first wrapper revision used the
   underscore spelling and FAMSA silently treated `-keep_duplicates` as a
   positional input filename, failing with `Unable to open input file
   -keep_duplicates`.
2. **`-gt_export` is a no-argument flag that repurposes the positional output
   slot**, not a flag that takes a filename. So you cannot get both an
   alignment and a guide tree from a single FAMSA invocation. To produce
   both files (parity with Clustal Omega's `--guidetree-out`) the wrapper
   runs FAMSA **twice** when `guidetree_out=true`: once for the alignment,
   once with `-gt_export` to write `<outdir>/guidetree.dnd`. The two runs
   share `-t`, `-gt`, `-medoidtree` so the exported tree matches the one
   used for the alignment. User accepted the doubled invocation as the
   correct trade-off for feature parity.

Both quirks are reflected in inline comments in `_align_famsa` so they don't
get re-introduced by a future cleanup.

## Changes

1. [src/mysca/prealign.py](../../src/mysca/prealign.py) — new
   `_align_famsa(in_fasta, out_path, *, threads, bin_path, extra_args,
   output_format, aligner_kwargs)` wrapper plus `_FAMSA_GT_CHOICES =
   ("sl", "upgma", "nj")`. Registered as `"famsa"` in both `ALIGNERS` and
   `ALIGNER_BINARIES`. Stockholm output uses the temp-FASTA → `AlignIO.convert`
   pattern already used by `_align_mafft`. Subprocess invocation goes through
   the existing `_run_cmd` helper.
2. [src/mysca/run_prealign.py](../../src/mysca/run_prealign.py) — updated
   docstring (external-binaries line, EXAMPLE USAGE, COMMAND LINE ARGUMENTS,
   per-aligner --align_args block, OUTPUTS) and the argparse `help=` for
   `--align` and `--align_args`. No code change — the `--align` choices are
   populated dynamically from `sorted(ALIGNERS)`.
3. [docs/cli_reference.md](../../docs/cli_reference.md) — added `famsa` to
   `--align` choices, a `famsa:` bullet list in the per-aligner-keys section
   (with a note about the always-on `-keep-duplicates`), updated the guide-
   tree output description and the External Binaries paragraph.
4. [README.md](../../README.md) — added `conda install -c conda-forge
   -c bioconda famsa  # sca-prealign --align famsa` to the install block.
5. [environment.yml](../../environment.yml) — added commented
   `#   - famsa    # sca-prealign --align famsa` next to the other optional
   aligners.
6. [tests/conftest.py](../../tests/conftest.py) — added `("famsa",
   "sca-prealign --align famsa")` to `OPTIONAL_TOOLS` so the missing-tool
   warning is accurate.
7. [tests/test_entrypoint_prealign.py](../../tests/test_entrypoint_prealign.py)
   — new `_FAMSA` / `needs_famsa` skip marker and seven tests mirroring the
   clustalo block:
   - `test_align_only_famsa` — basic run; uniform aligned lengths;
     `record_count(aligned) == record_count(INPUT_FASTA)` (verifies
     `-keep-duplicates` and the n_in==n_out invariant);
     `prealign_args.json["align"] == "famsa"`.
   - `test_align_famsa_stockholm_output` — `--output_format stockholm`
     writes `aligned.sto`, no `aligned.fasta`, AlignIO-readable.
   - `test_align_famsa_guidetree_out` — `--align_args guidetree_out=true`
     writes a non-empty `guidetree.dnd` (exercises the two-invocation path).
   - `test_align_famsa_gt_upgma` — `--align_args gt=upgma` succeeds.
   - `test_famsa_chain_to_preprocess` — chains `aligned.fasta` into
     `sca-preprocess`; asserts `preprocessing_results.npz` exists.
   - `test_align_famsa_missing_binary_fails_fast` — `--align_bin
     /nonexistent/famsa` raises `FileNotFoundError` without needing famsa on
     PATH (verifies aligner-aware `_resolve_bin` looks up famsa, not mafft).
   - `test_align_famsa_unknown_align_args_key_rejected` — wrapper rejects
     an unknown `--align_args` key for famsa.

## Verification

- `module load famsa` → `famsa --version` → FAMSA 2.4.1-45c9b2b (2025-05-09).
- `conda activate ./env && pytest tests/test_entrypoint_prealign.py -k famsa
  -v` → **7/7 passed**.
- Full suite (`pytest tests`) → **1190 passed**, 131 unrelated warnings,
  3m36s. No regressions in mafft, clustalo, or preprocess paths.

## Out of scope (deliberately deferred)

- **`sca-project` / `sca-structure` out-of-sample aligner registry**
  ([src/mysca/project/alignment.py](../../src/mysca/project/alignment.py))
  is a *separate* registry (`mafft_add`, `hmmalign`) for projecting new
  sequences onto an existing MSA. FAMSA has no `--add`-equivalent (no
  profile-to-existing-MSA mode that preserves columns), so it cannot be
  added there without conceptual changes. Not requested.
- **Refinement / distance-export / square-matrix FAMSA flags** — reachable
  via `--align_extra` if anyone needs them. Not promoted to structured
  `--align_args` keys to keep the surface tight.

## Pre-session state

Pre-session HEAD was `335e90a` ("Bump version to 0.1.4"). The working tree
also had unrelated in-progress edits to `src/mysca/pl/plotting.py`,
`src/mysca/run_sca.py`, and the conftest/prealign files (which this session
also touches for FAMSA). The FAMSA commit only stages the FAMSA-related
hunks; the unrelated pending edits are left untouched.

```bash
git checkout 335e90a  # "Bump version to 0.1.4"
```

## Commit

See the commit that follows this session note.
