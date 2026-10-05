# IJIE reproducibility package — 2026-09-27

This archive contains the executable analysis specification, aggregate CSV results,
and all revised figures for the IJIE manuscript. The source workbooks and
individual plotting intermediates are not distributed. It is an analysis package,
not evidence that author declarations or a journal submission have been completed.

## Reproduce the analyses and figures

Use Python 3.12.14. In a clean Python environment, from the extracted
archive directory:

```bash
python -m pip install -r requirements.txt
python code/run_all.py --data-root /absolute/path/to/authorized/raw --output-dir reproduced
```

The data directory must contain these exact relative paths:

- `Exp1/exp1_blur_merged.xlsx`
- `Exp1/exp1_contrast_merged.xlsx`
- `Exp2/exp2_blur_merged.xlsx`
- `Exp2/exp2_contrast_merged.xlsx`

The source SHA-256 checks run before analysis. A changed workbook, including a
resaved workbook with different bytes, fails the check; use the corresponding
archived version rather than disabling verification. The workbook fingerprints
are supplied in `results/IJIE/source_verification.csv` and in the code.

The command creates `reproduced/results/IJIE/` and
`reproduced/figures/IJIE/`, plus a source-verification record. `environment.json`
records versions for that run. The supplied requirements pin direct dependencies,
not every transitive dependency. The recorded analysis environment is preserved
in the supplied results; Pillow 12.3.0 was additionally used to verify figure files.
Small platform-dependent floating-point differences can occur.

The complete computation was run for the revision. This archive's wrapper is an
orchestrator for those same analysis and plotting entry points; its CLI, included
imports/files, Python syntax and archive checksums were checked at packaging.
No prospective registration or independent human-author verification is implied.

## Read the results

`OUTPUT_INVENTORY.md` maps each CSV to its purpose. `DATA_DICTIONARY.md` defines
samples, variables, units, uncertainty methods and random seeds. `output_schema.json`
lists every supplied CSV column and row count. `figures/IJIE/` supplies 600 dpi
PNG and vector PDF versions. IJIE Figure 1 is drawn directly from the validation
architecture; all other figures use computed results.

To redraw only this study after running its analyses:

```bash
python code/make_figures.py --study IJIE --results-root reproduced/results --figures-root redrawn
```

The Age and IJIE plotting steps require their locally generated `_private`
intermediates. Those files remain individual-level calculation intermediates and
are excluded from this archive. A supplied final image is not a substitute for
the underlying source data when reproducing a plot.

## Interpretation and distribution

Recorded identifiers are not a verified participant registry. Fixed presentation
order, missing calibration and cross-file identity uncertainty are retained
limitations. The three manuscripts share source observations and are not
independent replications. This revision supplies no new participants, experiment,
external validation or retrospective consent/ethics documentation.

The package does not include permission to release the original workbooks or
individual-level records. Authors must determine data-use and sharing permissions
from their actual ethics and consent records. No data license is invented here.
`SHA256SUMS.txt` fingerprints every distributed file except itself.
