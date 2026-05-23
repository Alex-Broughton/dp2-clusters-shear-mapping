# Python pipeline: Schirmer aperture E/B maps

Scripts in this directory build **Schirmer-filtered aperture mass** maps in E- and B-mode, with a per-cell signal-to-noise estimate, from DP2 metadetect shear. They support two input paths:

1. **FITS catalogs** produced under `notebooks/` (tract-native `x`, `y`) — primary path for production runs on SDF.
2. **Butler `object_shear_all`** via `map.py` (RA/Dec → virtual pixels) — for direct repo access without an intermediate FITS file.

The parent project overview is in [../README.md](../README.md). SLURM wrappers live in [../sh/](../sh/).

## Workflow (typical)

```
notebooks/metadetect_object_crossmatch.ipynb
        │
        ▼
_data/shear_table_xmatch_gold.fits
        │
        ├── schirmer_snr_weight.py     (one Rs, bin — maps + diagnostics)
        └── schirmer_snr_sweep.py      (grid of Rs, bin — CSV + summary plots)
```

## Module reference

| File | Role |
|------|------|
| [`catalog_dedupe.py`](catalog_dedupe.py) | Load shear FITS; quality mask; optional position dedupe before binning |
| [`schirmer_snr_weight.py`](schirmer_snr_weight.py) | Main FITS pipeline: bin → \(M_{\rm ap}\) → S/N maps and figures |
| [`schirmer_snr_sweep.py`](schirmer_snr_sweep.py) | Parameter sweep over \((R_{\rm ap}, {\rm bin})\); summary CSV and per-case E/B PNG/PDF |
| [`map.py`](map.py) | Same physics on Butler tables; CONFIG dict (no CLI); virtual pixel grid |
| [`schirmer_snr_weight_shenming.py`](schirmer_snr_weight_shenming.py) | Upstream reference copy (Shenming Fu); prefer `schirmer_snr_weight.py` for active development |

Checkpoint copies under `.ipynb_checkpoints/` are editor artifacts and are not part of the supported pipeline.

## Physics and units

- **Shear:** catalog columns `gauss_g1`, `gauss_g2` (metadetect `ns` step).
- **Weight:** inverse sum of Gaussian shear covariances, floored at `WEIGHT_FLOOR = 2×10⁵` (DP2-tuned in `schirmer_snr_weight.py`).
- **Coordinates (FITS path):** `x`, `y` in **tract-native pixels** at `PIX_SCALE_ARCSEC = 0.2` arcsec/pix (must match how the FITS table was built).
- **Schirmer scale:** `Rs_input` is \(R_{\rm ap}\) in those same pixels. The filter used in the map is **`Rs = Rs_input / bin_size`** (in units of binned grid cells).
- **S/N:** `SNR_E = M_ap_E / n_M_ap` (and similarly for B), with `n_M_ap` from the Schirmer et al. noise formula on squared weights and \(e^2\).

E-mode uses tangential shear \(g_t\); B-mode uses \(g_\times\) with the same filter \(Q(r/R_s)\).

## `catalog_dedupe.py`

Shared loader for FITS-based scripts.

**Quality cut (always):** finite `x`, `y`, `gauss_g1`, `gauss_g2`, and weight; invalid covariance rows are excluded (not given a floor weight).

**Optional dedupe (`--dedupe`):** greedy merge of sources within `dedupe_sep_pix` (default **1.0** tract pix ≈ 0.2″), keeping the **highest weight**. Intended for patch-overlap duplicates in `object_shear_all`, not `objectId` deduplication.

```python
from catalog_dedupe import load_shear_arrays, parse_dedupe_flags

x, y, e1, e2, weight, info = load_shear_arrays(
    "path/to/shear_table_xmatch_gold.fits",
    weight_floor=2.0e5,
    dedupe=True,
    dedupe_sep_pix=1.0,
)
```

CLI flags (must appear **before** positional arguments): `--dedupe`, `--no-dedupe`, `--dedupe-sep-pix SEP`.

## `schirmer_snr_weight.py`

Single-tract run from a gold FITS table.

### Usage

```bash
# Inside LSST stack env (matplotlib, astropy, scipy, numpy)
python schirmer_snr_weight.py [--dedupe] [--dedupe-sep-pix SEP] \
  <fits_file> <Rs_input_pix> <n_cpus> <threshold> [bin_size_pix]
```

| Argument | Meaning |
|----------|---------|
| `Rs_input_pix` | Schirmer \(R_{\rm ap}\) in tract pixels |
| `n_cpus` | Worker processes for per-cell aperture loop |
| `threshold` | Fallback color-scale limit if percentile stretch is zero |
| `bin_size_pix` | Cell size for 2D binning (default **90**) |

`ENABLE_DEDUPE = True` in the module turns dedupe on by default; override with `--no-dedupe`.

### Example (cluster-scale, ~3′ aperture, 18″ cells)

```bash
python schirmer_snr_weight.py --dedupe \
  ../_data/shear_table_xmatch_gold.fits \
  900 32 3 90
```

`Rs_input=900`, `bin_size=90` → `Rs/bin = 10` (~300×300 cells on tract 9813).

### Outputs (written to the FITS parent directory, e.g. `_data/`)

| Pattern | Description |
|---------|-------------|
| `{tag}_M_ap_b{bin}_Rs{Rs}.png/.pdf` | E/B \(M_{\rm ap}\) |
| `{tag}_M_ap_SNR_b{bin}_Rs{Rs}.png/.pdf` | E/B S/N |
| `{tag}_n_gal_b{bin}_Rs{Rs}.png/.pdf` | Galaxies per cell |
| `{tag}_shape_noise_std_b{bin}_Rs{Rs}.png/.pdf` | Per-cell std(\|e\|) |
| `{tag}_M_ap_E_vs_B_*.png/.pdf` | Cell-by-cell E vs B scatter / hexbin |
| `{tag}_M_ap_SNR_*_max.txt` | Peak E-mode S/N position (tract pixels) |

`tag` is the first token of the FITS basename (e.g. `shear` from `shear_table_xmatch_gold.fits`).

### Presets (comments in source)

| Name | `Rs_input` | `bin_size` | `Rs/bin` | ~scale |
|------|------------|------------|----------|--------|
| Tight | 600 | 60 | 10 | ~2′ / 12″ |
| Default | 900 | 90 | 10 | ~3′ / 18″ |
| Wide | 1500 | 150 | 10 | ~5′ / 30″ |

Avoid `Rs/bin` much larger than ~15 on a grid with only tens of cells (over-smooths the tract).

## `schirmer_snr_sweep.py`

Explores many \((R_{\rm ap}, {\rm bin})\) pairs with fixed **`Rs / bin_size`** ratios.

### Grid configuration

Edit at the top of the file:

```python
RS_INPUT_LIST = (2000, 2500, 3000)   # tract pixels (~6.7–10 arcmin)
RS_CELLS_LIST = (1.0, 2.0, 3.0, 4.0, 5.0)  # Rs_input / bin_size
```

`build_sweep_grid()` sets `bin_size = Rs_input / rs_cells` → **15 cases** for the lists above.

### Usage

```bash
python schirmer_snr_sweep.py [--dedupe] [--no-dedupe] [--dedupe-sep-pix SEP] \
  <fits_file> [n_cpus]
```

`ENABLE_DEDUPE = True` by default.

### Outputs

Written under `<fits_parent>/sweep_output/`:

| Path | Description |
|------|-------------|
| `{tag}_sweep_summary.csv` | Per-case metrics (`snr_e_max`, `snr_e_p90`, `n_gal_*`, grid size, …) |
| `{tag}_sweep_summary.png/.pdf` | Heatmaps vs \(R_{\rm ap}\) and \(R_{\rm ap}/{\rm bin}\) |
| `maps/{tag}_M_ap_Rs{Rs}_rc{rc}_b{bin}.png/.pdf` | E/B \(M_{\rm ap}\) per case |
| `maps/{tag}_M_ap_SNR_Rs{Rs}_rc{rc}_b{bin}.png/.pdf` | E/B S/N per case |

On SDF, submit via [../sh/test_tract_sweep.sh](../sh/test_tract_sweep.sh).

### Interpreting sweep CSV columns

| Column | Notes |
|--------|--------|
| `rs_input_pix`, `bin_size_pix`, `rs_cells` | Filter and grid definition |
| `nrow`, `ncol`, `n_cells` | Binned map size |
| `n_gal_mean`, `n_gal_median` | Sources per cell (check dedupe / patch seams) |
| `snr_e_max` | Max \|S/N_E\| over the tract — use to compare smoothing |
| `snr_e_p90` | 90th percentile of \|S/N_E\| — usually much lower than max |
| `pearson_snr_e_b` | Cell correlation of E and B S/N (near 0 → noise-like) |

**Expectation:** full-tract maps are shape-noise dominated; **\|S/N\| ≪ 3** on most of the tract is normal unless you are centered on a massive cluster. Use `snr_e_max` and the map PNGs, not `snr_e_p90` alone, when hunting peaks. See [../notebooks/README.md](../notebooks/README.md) for catalog provenance.

## `map.py`

Butler-driven variant: loads `object_shear_all` per tract, builds **virtual** `(x, y)` from RA/Dec using `PIX_SCALE_ARCSEC`, bins, and runs the same Schirmer loop. All settings are in the module **`CONFIG`** dict (repo path, collection, tracts, `RS_INPUT_PIX`, `BIN_SIZE_NATIVE_PIX`, `FIGURE_DIR`, …).

```bash
python map.py   # after editing CONFIG; requires lsst_distrib + Butler access
```

Outputs PDFs under `CONFIG["FIGURE_DIR"]` (default in file points at SDF test paths).

## Environment

- **SDF / Rubin:** `source loadLSST.sh` and `setup lsst_sitcom` (or matching tag). See [../sh/test_tract.sh](../sh/test_tract.sh).
- **Local:** Python 3 with `numpy`, `scipy`, `matplotlib`, `astropy` (FITS path only).
- **SLURM:** use `set -eo pipefail` before stack setup; do not enable `set -u` until after `setup` (conda activate scripts use unset variables). See `test_tract_sweep.sh`.

Run scripts from this directory or put it on `PYTHONPATH` so `import schirmer_snr_weight` resolves when invoking `schirmer_snr_sweep.py`.

## Related notebooks

| Notebook | Relation to Python |
|----------|-------------------|
| [../notebooks/metadetect_object_crossmatch.ipynb](../notebooks/metadetect_object_crossmatch.ipynb) | Builds `shear_table_xmatch_gold.fits` |
| [../notebooks/dp2_tract9813_metadetect_massmap.ipynb](../notebooks/dp2_tract9813_metadetect_massmap.ipynb) | HEALPix geometry; independent of FITS scripts |