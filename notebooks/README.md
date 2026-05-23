# Notebooks: DP2 shear catalogs and mass maps

Jupyter notebooks here access **DP2 Butler** data, build crossmatched shear tables, and prototype **Schirmer aperture E/B** maps on alternative sky pixelizations. Outputs feed the FITS-based scripts in [../python/](../python/).

Project context: [../README.md](../README.md).

## Environment

Run on **SDF** (or any host with DP2 Butler access) inside the Rubin stack:

```bash
source /sdf/group/rubin/sw/tag/v30_0_5_rc1/loadLSST.sh   # or your pinned tag
setup lsst_sitcom -t v30_0_5_rc1
jupyter lab   # or notebook
```

Typical extra packages: `astropy`, `healpy` (HEALPix notebook), `pyarrow`, `h5py`, `matplotlib`, `tqdm`.

Butler repo and collection strings are set in each notebook’s `CONFIG` / setup cells; defaults target **`/sdf/data/rubin/repo/dp2_prep`** and DP2 v30 stage-3 collections. Update paths if your install differs.

## Notebook index

| Notebook | Purpose | Main output |
|----------|---------|-------------|
| [`metadetect_object_crossmatch.ipynb`](metadetect_object_crossmatch.ipynb) | Crossmatch `object` ↔ `object_shear_all` on tract 9813; quality cuts; write gold FITS | `../_data/shear_table_xmatch_gold.fits` |
| [`dp2_tract9813_metadetect_massmap.ipynb`](dp2_tract9813_metadetect_massmap.ipynb) | Schirmer \(M_{\rm ap}\) E/B on **HEALPix** (RING, KD-tree neighbors) | In-notebook maps / figures (no gold FITS required) |

Editor checkpoints under `.ipynb_checkpoints/` are not maintained copies of the analysis.

---

## `metadetect_object_crossmatch.ipynb`

**Title:** DP2 object and metadetect shear crossmatch (tract 9813)

### What it does

1. Queries Butler for all **`object`** datasets and **`object_shear_all`** on a chosen tract (default **9813**).
2. Filters shear to `metaStep == "ns"`.
3. Crossmatches shear positions to the nearest **`object`** on the sky (`astropy.coordinates`), producing matched shear and object subsets.
4. Applies **gold** quality selections on the matched shear table.
5. Writes a **FITS** table used by [../python/schirmer_snr_weight.py](../python/schirmer_snr_weight.py) and [../python/schirmer_snr_sweep.py](../python/schirmer_snr_sweep.py).

Butler usage follows the same patterns as LSST DESC tutorials for metadetect catalogs (e.g. `access-dp2-metadetect-catalogs`).

### Key columns in the gold FITS

| Column | Use in Python pipeline |
|--------|-------------------------|
| `x`, `y` | Tract-native positions for 2D binning |
| `ra`, `dec` | Sky position (diagnostics; `map.py` uses RA/Dec directly from Butler) |
| `gauss_g1`, `gauss_g2` | Ellipticity components |
| `gauss_g1_g1_Cov`, `gauss_g2_g2_Cov` | Diagonal covariances → inverse-variance weights |

Units in FITS headers are normalized for compatibility (`prepare_table_for_fits`).

### Outputs

| Variable / file | Description |
|-----------------|-------------|
| `shear_xmatch`, `object_xmatch` | Boolean masks / indices from crossmatch |
| `shear_table_xmatch`, `object_table_xmatch` | Matched tables before gold cuts |
| `shear_table_gold` | After quality cuts |
| **`../_data/shear_table_xmatch_gold.fits`** | Primary input to the Python Schirmer pipeline |

Set `OUT_FITS` in the notebook to your checkout path (default in notebook points at SDF home).

### Downstream

```bash
# Single run (see ../python/README.md)
python ../python/schirmer_snr_weight.py --dedupe \
  ../_data/shear_table_xmatch_gold.fits 900 32 3 90

# Parameter sweep
sbatch ../sh/test_tract_sweep.sh
```

The notebook may also reference auxiliary data under `../_data/` (e.g. color libraries for validation plots); those are not required for the Schirmer FITS pipeline itself.

---

## `dp2_tract9813_metadetect_massmap.ipynb`

**Title:** Filtered aperture E/B shear maps on HEALPix (DP2, tract 9813)

### What it does

- Loads metadetect shear from Butler for tract **9813**.
- Evaluates Schirmer-filtered **aperture mass** at each **HEALPix** cell center (`healpy`, `NSIDE` configurable, default 2048).
- Uses a **cKDTree** on the sphere to find galaxies within a search radius (multiple of \(R_s\), not \(R_s\) alone).
- Builds E- and B-mode maps and optional peak finding with `PEAK_SN_MIN` (default 3).

This is a **different pixelization** from the tract-binned FITS pipeline: full-sky HEALPix cells vs rectangular tract `x`, `y` bins. Results are useful for comparison and prototyping; production tract maps for this repo currently use the FITS + `python/` path.

### Configuration (in notebook)

| Key | Typical value | Meaning |
|-----|---------------|---------|
| `TRACT` | 9813 | Skymap tract |
| `RS_INPUT_PIX` | 10000 | Schirmer scale in catalog pixel units |
| `PIX_SCALE_ARCSEC` | 0.2 | Converts pixel scale to sky angle |
| `SEARCH_RADIUS_RS_MULT` | 3.0 | Tree search radius = this × \(R_s\) |
| `NSIDE` | 2048 | HEALPix resolution |
| `PEAK_SN_MIN` | 3.0 | Peak detection threshold |

### Relation to `python/map.py`

Both read Butler `object_shear_all`, but:

| | HEALPix notebook | `map.py` |
|--|------------------|----------|
| Pixelization | HEALPix RING | Virtual tract grid from RA/Dec |
| Config | Notebook `CONFIG` | `map.py` `CONFIG` dict |
| Output | Notebook plots | PDF under `FIGURE_DIR` |

---

## Recommended order

```
1. metadetect_object_crossmatch.ipynb  →  shear_table_xmatch_gold.fits
2. python/schirmer_snr_sweep.py        →  choose Rs, bin from CSV + maps/
3. python/schirmer_snr_weight.py       →  full diagnostics for chosen parameters
```

Use the HEALPix notebook when you need all-sky pixelization or want to cross-check tract-bin results.

## Data layout

```
dp2-clusters-shear-mapping/
├── _data/
│   ├── shear_table_xmatch_gold.fits    # from crossmatch notebook
│   └── sweep_output/                   # from schirmer_snr_sweep.py
│       ├── shear_sweep_summary.csv
│       └── maps/
└── notebooks/
    ├── metadetect_object_crossmatch.ipynb
    └── dp2_tract9813_metadetect_massmap.ipynb
```

Keep large FITS and sweep products under `_data/` (gitignored or LFS as per repo policy).

## Caveats

- **`object_shear_all` patch overlap:** duplicate detections at patch boundaries can bias `n_gal` maps; the Python pipeline supports **`--dedupe`** (1 tract pixel separation). Run the crossmatch notebook first, then dedupe at map time.
- **Tract pixels vs arcsec:** Python assumes **0.2 arcsec/tract pix** when converting \(R_{\rm ap}\) to arcmin in log messages; keep this consistent with how `x`, `y` were defined when exporting the FITS.
- **S/N on full tracts:** peak \|S/N\| in summary CSVs is often **≪ 3** for tract-wide maps; that reflects shape noise and cosmic variance, not necessarily a failed run. Interpret peaks on the PNG maps or in cluster-centered regions.

## Further reading

- [../python/README.md](../python/README.md) — CLI, sweep grid, outputs, troubleshooting
- [Arun's DP2 shear table tutorial](https://github.com/lsst-so/sciunit_wlshear/tree/main/notebooks/dp2)
- [Project PUB-DB entry](https://lsstdesc.slac.stanford.edu/DESCPub/app/PB/show_project?pid=492) (linked from root README)
