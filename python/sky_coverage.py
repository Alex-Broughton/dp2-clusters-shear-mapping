#!/usr/bin/env python
"""DP2 sky coverage under the shapeHSM shear selection cuts.

Batch version of the loop that used to live in notebooks/sky-coverage.ipynb. One
tract per worker process, so the wall time is set by the slowest tract rather
than by their sum. The notebook now only reads the products written here and
makes the figures.

Everything is controlled by the CONFIG block below; the script takes no command
line arguments. Run it after setting up the Rubin (LSST) stack so that the
Butler is importable. See sh/run_sky_coverage.sh for the slurm launcher.

Pipeline:
  1. Enumerate tracts with an i-band coadd, an object table, and a photo-z.
  2. In parallel, per tract: read the object table, apply the shapeHSM source
     selection, and record counts, occupied HEALPix cells, and n(z) before and
     after the cuts.
  3. Accumulate the per-tract results and write:
       <PREFIX>_summary.ecsv    per-tract counts, area, and density
       <PREFIX>_cells.fits      per-cell object counts before and after the cuts
       <PREFIX>_nz.npz          per-tract photo-z histograms
       <PREFIX>_failures.txt    tracts that could not be read (only if any)

Effective area is the number of HEALPix cells holding at least one object before
the cuts, so it follows the coadd footprint and does not shrink with the
selection. The density ratio before/after is then a clean measure of the cost of
the cuts.

Author: Alex Broughton.
"""

import os
import time
from multiprocessing import Pool

import numpy as np
import healpy as hp
import astropy.units as u
from astropy.table import Table

from lsst.daf.butler import Butler


# ============================================================================
# CONFIG -- all manual controls live here. No command line arguments.
# ============================================================================
# Resolve the repository root (two levels up from this file: python/ -> repo/)
# so the default output directory is repo/output regardless of where slurm runs.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    # --- Butler / datasets -----------------------------------------------
    "REPO": "/sdf/data/rubin/repo/dp2_prep",  # Butler repository root
    "SKYMAP": "lsst_cells_v2",  # Skymap name
    "INSTRUMENT": "LSSTCam",  # Instrument dimension
    "COADD_COLLECTION": "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage3",  # Source of the tract list
    "OBJECT_COLLECTION": "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage3",  # Object tables
    "PZ_COLLECTION": "u/echarles/pz_market/dp2_v3p1/dp2_6band/v0",  # Photo-z ensembles
    "COADD_DATASET": "deep_coadd",  # Coadd dataset type used to enumerate tracts
    "OBJECT_DATASET": "object",  # Tract-level object table
    "PZ_DATASET": "pz_estimate_fzboost",  # qp Ensemble of p(z), one per tract
    "BAND": "i",  # Band whose coadd defines the footprint
    # --- Sky cells -------------------------------------------------------
    "CELL_NSIDE": 1024,  # HEALPix RING grid (the mass map uses 2048); sets map resolution and area unit
    # --- Source selection, as in shapehsm_massmap.py ---------------------
    "SNR_MIN": 5.0,  # Minimum per-band griz Sersic flux S/N
    "RESO_FACTOR_MIN": 0.125,  # Minimum PSF-resolution factor
    "COLOR_ABS_MAX": 5.0,  # Maximum |g-r|, |r-i|, |i-z| in AB magnitudes
    "EXTENDEDNESS_MIN": 0.2,  # Minimum griz model extendedness
    "E_SQ_MAX": 4.0,  # Maximum e1^2 + e2^2 for the HSM regauss distortion
    "T_MAX_ARCSEC2": None,  # Optional size cut on i_ixx + i_iyy; None disables it
    "PIX_SCALE_ARCSEC": 0.2,  # Arcsec per pixel, for the size cut
    "RED_SEQ_SLOPE": 2.0484,  # Red-sequence line: g-i = slope * (r-z) + intercept
    "RED_SEQ_INTERCEPT": -0.10,  # Objects above that line are rejected (dereddened colors when DEREDDEN)
    # Milky Way dust correction, following Rubin tutorial 311.1: m_0 = m - R_band * E(B-V), with E(B-V)
    # from the Object-table ebv column (SFD98). All color cuts use the corrected magnitudes.
    "DEREDDEN": True,
    "EBV_COLUMN": "ebv",
    # R_band = A_band / E(B-V) from rubin_sim DustValues().r_x: CCM89, R_V = 3.1, flat SED, LSST baseline
    # throughputs (rubin_sim 2.6.2, throughputs_2025_08_25).
    "EXTINCTION_R": {"u": 4.757, "g": 3.661, "r": 2.701, "i": 2.054, "z": 1.590, "y": 1.308},
    # --- Photo-z ---------------------------------------------------------
    "Z_MIN": 0.0,  # Photo-z histogram range
    "Z_MAX": 3.0,
    "Z_NBINS": 60,
    # --- Runtime / output ------------------------------------------------
    # Each worker holds one tract object table in memory, so N_PROC trades
    # throughput against peak memory. 16 x a few hundred MB is comfortable on a
    # roma node; raise it if the tracts turn out to be small.
    "N_PROC": 16,  # Worker processes (keep <= slurm --cpus-per-task)
    "MAX_TRACTS": None,  # Set to an int for a quick partial run
    "OUTPUT_DIR": os.path.join(_REPO_ROOT, "output"),  # All products land here
    "OUTPUT_PREFIX": "coverage",  # Shared stem so files are clearly grouped
}

# Derived quantities (kept out of the manual block to avoid inconsistencies).
CONFIG["CELL_AREA_ARCMIN2"] = float(hp.nside2pixarea(CONFIG["CELL_NSIDE"], degrees=True) * 3600.0)
CONFIG["NPIX"] = int(hp.nside2npix(CONFIG["CELL_NSIDE"]))
Z_BINS = np.linspace(CONFIG["Z_MIN"], CONFIG["Z_MAX"], int(CONFIG["Z_NBINS"]) + 1)


# Flags that must all be clean. Resolved against the table schema at startup, so
# a column the object table does not carry is dropped rather than raising.
FLAG_COLUMNS = [
    "i_i_flag",
    "i_iPSF_flag",
    "i_hsmShapeRegauss_flag",
    "i_gaapFlux_flag",
    "sersic_no_data_flag",
    "sersic_unknown_flag",
    "shape_flag",
    "i_pixelFlags_bad",
    "i_pixelFlags_nodata",
    "i_pixelFlags_edge",
    "i_pixelFlags_saturatedCenter",
    "i_pixelFlags_crCenter",
    "i_pixelFlags_interpolatedCenter",
    "i_pixelFlags_suspectCenter",
]

WANTED_COLUMNS = [
    "objectId",
    "coord_ra",
    "coord_dec",
    "ebv",  # SFD98 E(B-V) at the object position, for the dust correction
    "detect_isDeblendedModelSource",
    "griz_model_extendedness",
    "i_hsmShapeRegauss_e1",
    "i_hsmShapeRegauss_e2",
    "i_ixx",
    "i_iyy",
    "i_ixxPSF",
    "i_iyyPSF",
    "g_sersicFlux",
    "g_sersicFluxErr",
    "r_sersicFlux",
    "r_sersicFluxErr",
    "i_sersicFlux",
    "i_sersicFluxErr",
    "z_sersicFlux",
    "z_sersicFluxErr",
] + FLAG_COLUMNS

# Point-estimate and id keys looked for in the qp Ensemble ancil table.
PZ_POINT_KEYS = ("zmode", "zmean", "zmedian", "z_mode", "z_mean", "z_median")
PZ_ID_KEYS = ("objectId", "object_id", "id")


# ============================================================================
# Table helpers
# ============================================================================
def _col(tbl, name):
    """Column as a float64 numpy array."""
    return np.asarray(tbl[name], dtype=np.float64)


def flux_to_abmag(flux_njy):
    """Convert nJy fluxes to AB magnitudes; non-positive or non-finite fluxes become NaN."""
    flux = np.asarray(flux_njy, dtype=np.float64)
    mag = np.full(flux.shape, np.nan, dtype=np.float64)
    good = np.isfinite(flux) & (flux > 0.0)
    mag[good] = (flux[good] * u.nJy).to(u.ABmag).value
    return mag


def griz_mags(tbl):
    """griz AB magnitudes from the Sersic fluxes, dust-corrected when CONFIG["DEREDDEN"] is set.

    m_0 = m - R_band * E(B-V). Objects with a non-finite E(B-V) get NaN magnitudes and fail the color cuts.
    """
    mags = {b: flux_to_abmag(_col(tbl, f"{b}_sersicFlux")) for b in "griz"}
    if not CONFIG["DEREDDEN"]:
        return mags
    ebv_col = CONFIG["EBV_COLUMN"]
    if ebv_col not in tbl.colnames:
        raise KeyError(
            f"CONFIG['DEREDDEN'] is set but the table has no '{ebv_col}' column. "
            "Add it to WANTED_COLUMNS or set CONFIG['DEREDDEN'] = False."
        )
    ebv = _col(tbl, ebv_col)
    return {b: m - CONFIG["EXTINCTION_R"][b] * ebv for b, m in mags.items()}


# ============================================================================
# Source selection (carried over from the notebook / shapehsm_massmap.py)
# ============================================================================
def selection_masks(data, flag_columns):
    """Combined selection mask and the per-group masks that build it."""
    n = len(data)

    flags_ok = np.ones(n, dtype=bool)
    for name in flag_columns:
        flags_ok &= ~np.asarray(data[name], dtype=bool)

    primary_ok = np.asarray(data["detect_isDeblendedModelSource"], dtype=bool)

    e_sq = _col(data, "i_hsmShapeRegauss_e1") ** 2 + _col(data, "i_hsmShapeRegauss_e2") ** 2
    shape_ok = e_sq <= CONFIG["E_SQ_MAX"]

    # Resolution factor 1 - T_PSF / T_obj; values near 0 are PSF-like, i.e. stars.
    t_obj = _col(data, "i_ixx") + _col(data, "i_iyy")
    with np.errstate(divide="ignore", invalid="ignore"):
        reso = 1.0 - (_col(data, "i_ixxPSF") + _col(data, "i_iyyPSF")) / t_obj
    reso_ok = reso > CONFIG["RESO_FACTOR_MIN"]

    # Dust-corrected when CONFIG["DEREDDEN"]; the color cuts and red-sequence veto use these.
    mag = griz_mags(data)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = {b: _col(data, f"{b}_sersicFlux") / _col(data, f"{b}_sersicFluxErr") for b in "griz"}

    snr_min = CONFIG["SNR_MIN"]
    detect_ok = _col(data, "griz_model_extendedness") > CONFIG["EXTENDEDNESS_MIN"]
    for b in "griz":
        detect_ok &= snr[b] > snr_min

    # NaN comparisons are False, so objects with a non-positive griz flux drop out here.
    color_max = CONFIG["COLOR_ABS_MAX"]
    gi = mag["g"] - mag["i"]
    rz = mag["r"] - mag["z"]
    color_ok = np.abs(mag["g"] - mag["r"]) < color_max
    color_ok &= np.abs(mag["r"] - mag["i"]) < color_max
    color_ok &= np.abs(mag["i"] - mag["z"]) < color_max
    color_ok &= np.isfinite(gi) & np.isfinite(rz)

    # Red-sequence veto: likely cluster members, which would dilute the shear signal.
    color_ok &= ~(gi > CONFIG["RED_SEQ_SLOPE"] * rz + CONFIG["RED_SEQ_INTERCEPT"])

    # Optional size cut; the threshold is in arcsec^2 while the moments are in pixels^2.
    size_ok = np.ones(n, dtype=bool)
    if CONFIG["T_MAX_ARCSEC2"] is not None:
        size_ok = t_obj < CONFIG["T_MAX_ARCSEC2"] / CONFIG["PIX_SCALE_ARCSEC"] ** 2

    groups = {
        "measurement + pixel flags": flags_ok,
        "deblended model source": primary_ok,
        "|e| <= 2": shape_ok,
        f"reso_factor > {CONFIG['RESO_FACTOR_MIN']}": reso_ok,
        f"extendedness + S/N > {snr_min:g}": detect_ok,
        "colors + red sequence": color_ok,
        "size": size_ok,
    }

    mask = np.ones(n, dtype=bool)
    for cut in groups.values():
        mask &= cut
    return mask, groups


# ============================================================================
# Photo-z
# ============================================================================
def photoz_point_estimates(ens):
    """Point-estimate redshifts and object ids from a qp Ensemble; ids are None if absent."""
    ancil = getattr(ens, "ancil", None) or {}

    z = None
    for key in PZ_POINT_KEYS:
        if key in ancil:
            z = np.asarray(ancil[key], dtype=np.float64).ravel()
            break
    if z is None:
        # No stored point estimate, so evaluate the PDF median instead.
        z = np.asarray(ens.median(), dtype=np.float64).ravel()

    ids = None
    for key in PZ_ID_KEYS:
        if key in ancil:
            ids = np.asarray(ancil[key]).ravel()
            break
    return z, ids


def photoz_for_rows(data, ens):
    """Photo-z aligned to the object-table rows, NaN where the object has no p(z)."""
    z_pz, ids_pz = photoz_point_estimates(ens)
    out = np.full(len(data), np.nan, dtype=np.float64)

    if ids_pz is not None and "objectId" in data.colnames:
        obj_ids = np.asarray(data["objectId"])
        order = np.argsort(ids_pz)
        sorted_ids = ids_pz[order]
        pos = np.clip(np.searchsorted(sorted_ids, obj_ids), 0, sorted_ids.size - 1)
        hit = sorted_ids[pos] == obj_ids
        out[hit] = z_pz[order][pos[hit]]
    elif z_pz.size == len(data):
        # No ids to join on, so the ensemble is assumed row-matched to the object table.
        out = z_pz
    return out


# ============================================================================
# Parallel per-tract evaluation
# ============================================================================
# Globals populated in each worker by the Pool initializer. The Butler is built
# per worker rather than pickled from the parent, so each process gets its own
# registry connection.
_W = {}


def _init_worker(object_columns, flag_columns, pz_tracts):
    """Pool initializer: build this worker's Butler and stash the resolved columns."""
    _W["butler"] = Butler(
        CONFIG["REPO"],
        skymap=CONFIG["SKYMAP"],
        instrument=CONFIG["INSTRUMENT"],
    )
    _W["object_columns"] = object_columns
    _W["flag_columns"] = flag_columns
    _W["pz_tracts"] = pz_tracts


def _load_object_table(butler, tract, columns):
    """Object table for one tract, restricted to the selection columns."""
    return butler.get(
        CONFIG["OBJECT_DATASET"],
        dataId={"tract": int(tract), "skymap": CONFIG["SKYMAP"]},
        collections=CONFIG["OBJECT_COLLECTION"],
        parameters={"columns": columns},
    )


def _load_photoz(butler, tract):
    """qp Ensemble of p(z) for one tract."""
    return butler.get(
        CONFIG["PZ_DATASET"],
        dataId={"tract": int(tract), "skymap": CONFIG["SKYMAP"]},
        collections=CONFIG["PZ_COLLECTION"],
    )


def _process_tract(tract):
    """Worker task: counts, effective area, occupied cells, and n(z) for one tract.

    Returns a dict, or a dict carrying only 'tract' and 'error' if the tract
    could not be processed. Exceptions are caught here so one bad tract cannot
    take down the pool.
    """
    tract = int(tract)
    try:
        butler = _W["butler"]
        data = _load_object_table(butler, tract, _W["object_columns"])
        if len(data) == 0:
            return {"tract": tract, "error": "empty object table"}

        mask, _ = selection_masks(data, _W["flag_columns"])
        n_before = int(len(data))
        n_after = int(mask.sum())

        ra = _col(data, "coord_ra")
        dec = _col(data, "coord_dec")
        ipix = hp.ang2pix(CONFIG["CELL_NSIDE"], ra, dec, nest=False, lonlat=True)

        cells_before, counts_before = np.unique(ipix, return_counts=True)
        cells_after, counts_after = np.unique(ipix[mask], return_counts=True)
        area = cells_before.size * CONFIG["CELL_AREA_ARCMIN2"]

        # Circular mean RA, so a tract sitting on RA = 0 is not averaged across the seam.
        ra_rad = np.deg2rad(ra)
        ra_center = float(
            np.rad2deg(np.arctan2(np.mean(np.sin(ra_rad)), np.mean(np.cos(ra_rad)))) % 360.0
        )

        if tract in _W["pz_tracts"]:
            z = photoz_for_rows(data, _load_photoz(butler, tract))
        else:
            z = np.full(n_before, np.nan)
        finite = np.isfinite(z)

        return {
            "tract": tract,
            "ra_center": ra_center,
            "dec_center": float(np.mean(dec)),
            "n_cells": int(cells_before.size),
            "area_arcmin2": float(area),
            "n_before": n_before,
            "n_after": n_after,
            "n_photoz": int(finite.sum()),
            "density_before": n_before / area,
            "density_after": n_after / area,
            "cells_before": cells_before.astype(np.int64),
            "counts_before": counts_before.astype(np.int32),
            "cells_after": cells_after.astype(np.int64),
            "counts_after": counts_after.astype(np.int32),
            "h_before": np.histogram(z[finite], bins=Z_BINS)[0].astype(np.int64),
            "h_after": np.histogram(z[finite & mask], bins=Z_BINS)[0].astype(np.int64),
        }
    except Exception as exc:
        return {"tract": tract, "error": repr(exc)}


# ============================================================================
# Driver
# ============================================================================
def available_tracts(butler, dataset_type, collections, **data_id):
    """Sorted tract ids holding at least one dataset of this type in these collections."""
    data_ids = butler.registry.queryDataIds(
        ["tract"],
        datasets=dataset_type,
        collections=collections,
        skymap=CONFIG["SKYMAP"],
        **data_id,
    )
    return sorted({int(d["tract"]) for d in data_ids})


def resolve_columns(butler, tract):
    """Intersect the wanted columns with the object table schema.

    Reads the schema alone when the datastore supports it, so this costs nothing
    on a table with hundreds of columns.
    """
    data_id = {"tract": int(tract), "skymap": CONFIG["SKYMAP"]}
    try:
        names = set(
            butler.get(
                CONFIG["OBJECT_DATASET"],
                dataId=data_id,
                collections=CONFIG["OBJECT_COLLECTION"],
                storageClass="ArrowSchema",
            ).names
        )
    except Exception as exc:
        print("ArrowSchema unavailable, reading one full table instead:", exc, flush=True)
        names = set(
            butler.get(
                CONFIG["OBJECT_DATASET"],
                dataId=data_id,
                collections=CONFIG["OBJECT_COLLECTION"],
            ).colnames
        )

    object_columns = [c for c in WANTED_COLUMNS if c in names]
    flag_columns = [c for c in FLAG_COLUMNS if c in names]
    missing = [c for c in WANTED_COLUMNS if c not in names]
    print("Missing columns:", missing or "none", flush=True)
    if CONFIG["DEREDDEN"] and CONFIG["EBV_COLUMN"] not in names:
        # Fail once here rather than once per tract inside the workers.
        raise KeyError(f"CONFIG['DEREDDEN'] is set but the object table has no '{CONFIG['EBV_COLUMN']}' column")
    print(f"Reading {len(object_columns)} of {len(names)} columns per tract", flush=True)
    return object_columns, flag_columns


def run(tracts, object_columns, flag_columns, pz_tracts):
    """Process every tract in parallel and accumulate the results.

    Returns the per-tract summary table, the sparse per-cell counts, the n(z)
    histograms keyed by tract, and the per-tract failures.
    """
    map_before = np.zeros(CONFIG["NPIX"], dtype=np.int32)
    map_after = np.zeros(CONFIG["NPIX"], dtype=np.int32)
    rows = []
    hists = {}
    failures = {}

    n_total = len(tracts)
    print(f"Processing {n_total} tracts on {CONFIG['N_PROC']} processes...", flush=True)
    t_start = time.time()

    init_args = (object_columns, flag_columns, pz_tracts)
    with Pool(CONFIG["N_PROC"], initializer=_init_worker, initargs=init_args) as pool:
        # chunksize=1: each task is a whole tract, so there is no IPC overhead to
        # amortize and dynamic scheduling keeps the workers balanced.
        for k, res in enumerate(pool.imap_unordered(_process_tract, tracts, chunksize=1), start=1):
            if "error" in res:
                failures[res["tract"]] = res["error"]
                status = "FAILED"
            else:
                map_before[res["cells_before"]] += res["counts_before"]
                map_after[res["cells_after"]] += res["counts_after"]
                hists[res["tract"]] = (res["h_before"], res["h_after"])
                rows.append({key: v for key, v in res.items() if not isinstance(v, np.ndarray)})
                status = f"{res['n_after']:,} of {res['n_before']:,} kept"

            elapsed = time.time() - t_start
            rate = k / elapsed if elapsed > 0 else 0.0
            eta = (n_total - k) / rate if rate > 0 else float("nan")
            print(
                f"[{k}/{n_total}] tract {res['tract']}: {status}"
                f" | {elapsed / 60.0:.1f} min elapsed, ~{eta / 60.0:.1f} min left",
                flush=True,
            )

    if not rows:
        raise RuntimeError("No tract could be processed; see the failures above.")

    summary = Table(rows=rows)
    summary["retained"] = summary["n_after"] / summary["n_before"]
    summary.sort("tract")

    # Only occupied cells are kept: a full-sky array at NSIDE 1024 is mostly zeros.
    occupied = np.nonzero(map_before)[0]
    cells = Table(
        data=[occupied.astype(np.int64), map_before[occupied], map_after[occupied]],
        names=["IPIX", "N_BEFORE", "N_AFTER"],
    )

    print(f"Processed {len(summary)} tracts, {len(failures)} failed", flush=True)
    return summary, cells, hists, failures


# ============================================================================
# Output writers
# ============================================================================
def _out_path(suffix):
    """Build an output path of the form output/<PREFIX>_<suffix>."""
    return os.path.join(CONFIG["OUTPUT_DIR"], f"{CONFIG['OUTPUT_PREFIX']}_{suffix}")


def _scalar_config():
    """CONFIG entries that survive a round trip through table metadata."""
    return {k: v for k, v in CONFIG.items() if v is None or isinstance(v, (bool, int, float, str))}


def write_products(summary, cells, hists, failures):
    """Write the summary table, the per-cell counts, and the n(z) histograms."""
    summary.meta.update(_scalar_config())
    path = _out_path("summary.ecsv")
    summary.write(path, format="ascii.ecsv", overwrite=True)
    print("Wrote per-tract summary:", path, flush=True)

    cells.meta.update({"NSIDE": CONFIG["CELL_NSIDE"], "ORDERING": "RING"})
    path = _out_path("cells.fits")
    cells.write(path, format="fits", overwrite=True)
    print("Wrote per-cell counts:", path, flush=True)

    tract_ids = np.array(sorted(hists), dtype=np.int64)
    path = _out_path("nz.npz")
    np.savez_compressed(
        path,
        z_bins=Z_BINS,
        tracts=tract_ids,
        h_before=np.array([hists[t][0] for t in tract_ids]),
        h_after=np.array([hists[t][1] for t in tract_ids]),
    )
    print("Wrote photo-z histograms:", path, flush=True)

    if failures:
        path = _out_path("failures.txt")
        with open(path, "w") as fh:
            for tract, msg in sorted(failures.items()):
                fh.write(f"{tract}\t{msg}\n")
        print(f"Wrote {len(failures)} failures:", path, flush=True)


def print_totals(summary):
    """Log the survey-level numbers the notebook also reports."""
    area = float(np.sum(summary["area_arcmin2"]))
    n_before = int(np.sum(summary["n_before"]))
    n_after = int(np.sum(summary["n_after"]))
    print(f"Tracts processed: {len(summary)}")
    print(f"Effective area:   {area / 3600.0:.1f} deg^2")
    print(f"Objects:          {n_before:,} -> {n_after:,} ({100.0 * n_after / n_before:.1f}% retained)")
    print(f"Density:          {n_before / area:.2f} -> {n_after / area:.2f} gal/arcmin^2")
    print(f"With a photo-z:   {int(np.sum(summary['n_photoz'])):,} objects")


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    print(
        f"CELL_NSIDE = {CONFIG['CELL_NSIDE']} -> {CONFIG['NPIX']:,} cells,",
        f"{CONFIG['CELL_AREA_ARCMIN2']:.2f} arcmin^2 each",
        flush=True,
    )

    # 1. Enumerate tracts. The photo-z collection comes from a different DRP run,
    #    so its coverage is tracked separately rather than required.
    butler = Butler(CONFIG["REPO"], skymap=CONFIG["SKYMAP"], instrument=CONFIG["INSTRUMENT"])
    tracts_coadd = available_tracts(
        butler, CONFIG["COADD_DATASET"], CONFIG["COADD_COLLECTION"], band=CONFIG["BAND"]
    )
    tracts_object = available_tracts(butler, CONFIG["OBJECT_DATASET"], CONFIG["OBJECT_COLLECTION"])
    tracts_pz = available_tracts(butler, CONFIG["PZ_DATASET"], CONFIG["PZ_COLLECTION"])

    object_set = set(tracts_object)
    pz_tracts = frozenset(tracts_pz)
    tracts = [t for t in tracts_coadd if t in object_set]
    if CONFIG["MAX_TRACTS"] is not None:
        tracts = tracts[: int(CONFIG["MAX_TRACTS"])]

    print(f"{CONFIG['BAND']}-band coadd tracts: {len(tracts_coadd)}", flush=True)
    print(f"object tracts:          {len(tracts_object)}", flush=True)
    print(f"photo-z tracts:         {len(tracts_pz)}", flush=True)
    print(
        f"tracts in this run:     {len(tracts)}"
        f" ({sum(t in pz_tracts for t in tracts)} with photo-z)",
        flush=True,
    )
    if not tracts:
        raise RuntimeError("No tract has both an i-band coadd and an object table.")

    # 2. Resolve the column list once, then fan the tracts out over the pool.
    object_columns, flag_columns = resolve_columns(butler, tracts[0])
    summary, cells, hists, failures = run(tracts, object_columns, flag_columns, pz_tracts)

    # 3. Write everything the plotting notebook needs.
    write_products(summary, cells, hists, failures)
    print_totals(summary)
    print("Done. Outputs written to", CONFIG["OUTPUT_DIR"])


if __name__ == "__main__":
    main()
