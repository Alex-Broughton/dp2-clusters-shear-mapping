# Notebooks

Analysis notebooks for the DP2 shear and mass-map project. Project context is in the
[root README](../README.md).

## Environment

All notebooks read DP2 data through the Butler, so they need the Rubin stack on SDF (or any
host with access to /sdf/data/rubin/repo/dp2_prep):

```bash
source /sdf/group/rubin/sw/tag/v30_0_8_rc4/loadLSST.sh
setup lsst_sitcom -t v30_0_8_rc4
```

Extra packages used: astropy, healpy, scipy, h5py, pyarrow, matplotlib, tqdm.

## Directory layout

Each notebook resolves the repo root from its working directory (notebooks/ or the repo root)
and uses three directories under it:

| Directory | Contents |
|-----------|----------|
| _data/ | External inputs not produced here (e.g. CWWSB_LSST_Roman_colors_YJH.hdf5) |
| output/ | Generated tables and catalogs (.ecsv, .fits, .npz) and slurm logs |
| plots/ | Figures, saved as PDF |

The contents of all three are gitignored.

## Notebooks

| Notebook | Purpose | Writes to output/ |
|----------|---------|-------------------|
| dp2_shearmap.ipynb | Schirmer-filtered aperture E/B maps on HEALPix for tract 9813, using HSM regauss shapes from the object table, plus peak finding | {tract}_shapeHSM_massmap.fits |
| dp2-shear-map-tract.ipynb | Schirmer-filtered aperture E/B maps and S/N peaks for any tract (set TRACT in the parameters cell; papermill-compatible) | {tract}_shapeHSM_massmap.fits, {tract}_shapeHSM_peaks.fits |
| dp2_shearmap_PSZ2_G309.43-72.86.ipynb | The same maps on tract 2877 around PSZ2 G309.43-72.86, with the tangential shear profile and the filter/cell-size justification | 2877_shapeHSM_massmap.fits |
| dp2_tract_stats.ipynb | Per-tract source counts and survey property statistics (PSF size and ellipticity, depth) over the DP2 footprint | dp2_shapehsm_tract_stats.ecsv, dp2_shapehsm_tract_stats_failures.txt |
| sky-coverage.ipynb | Source density, retained fraction, and n(z) under the shear selection cuts. python/sky_coverage.py (sh/run_sky_coverage.sh) is the batch version | dp2_coverage_summary.ecsv, dp2_coverage_nz.npz |
| investigation_source_selection.ipynb | Object vs metadetect shear crossmatch on tract 9813 and color-color red-sequence cuts | nothing (figures only) |

Editor checkpoints under .ipynb_checkpoints/ are not maintained.
