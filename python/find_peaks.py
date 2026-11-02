"""Find E-mode shear peaks in the DP2 deep-field HealSparse S/N maps.

Applies the peak definition in peak_definition.md (same cuts as notebooks/dp2-shear-map-tract.ipynb) to each
field's E/B S/N maps in MAP_DIR. Aperture completeness is rebuilt from the per-tract source-count maps in
MAP_DIR/shapeHSM. For each field it writes, next to the input maps:

    <FIELD>_peaks.fits           peak catalog, highest E-mode S/N first
    <FIELD>_completeness.hs      aperture completeness on the E-mode map's cells

Usage:
    python python/find_peaks.py [--map-dir DIR] [--out-dir DIR]
"""
import argparse
import warnings
from pathlib import Path

import healpy as hp
import healsparse as hsp
import numpy as np
from astropy.io.fits.verify import VerifyWarning
from astropy.table import Table

MAP_DIR = "/global/cfs/cdirs/lsst/groups/CL/DP2/dp2_massmaps/hsp_maps"

RS_ARCMIN = 15.0
THETA_MAX_ARCMIN = 1.2 * RS_ARCMIN  # aperture search radius used for completeness
EXCL_ARCMIN = 0.4 * RS_ARCMIN
B_MATCH_ARCMIN = 0.4 * RS_ARCMIN
SN_MIN = 3.0
COMPLETENESS_MIN = 0.6
B_SN_FLAG = 3.0

# Header keys match the notebook catalogs, several of which need FITS HIERARCH cards.
warnings.simplefilter("ignore", VerifyWarning)

# The two naming schemes in hsp_maps: COSMOS_emap.hs / COSMOS_bmap.hs and elais_emode.hs / elais_bmode.hs.
E_SUFFIXES = {"_emap.hs": "_bmap.hs", "_emode.hs": "_bmode.hs"}


def read_map(path):
    """HealSparse map as (sorted nest pixels, float values) with sentinel cells dropped."""
    m = hsp.HealSparseMap.read(str(path))
    pix = np.sort(m.valid_pixels)
    return m, pix, m[pix].astype(np.float64)


def lookup(sorted_pix, values, query, fill):
    """values at the query pixels, `fill` where a pixel is not in sorted_pix."""
    pos = np.clip(np.searchsorted(sorted_pix, query), 0, sorted_pix.size - 1)
    hit = sorted_pix[pos] == query
    out = np.full(np.shape(query), fill, dtype=np.float64)
    out[hit] = values[pos[hit]]
    return out


def load_counts(shape_dir):
    """Source counts per nest pixel, summed over every tract's ngals map (a cell on a tract boundary appears in both)."""
    pix, counts = [], []
    for path in sorted(shape_dir.glob("shapeHSM_*_ngals.hs")):
        m = hsp.HealSparseMap.read(str(path))
        p = m.valid_pixels
        pix.append(p)
        counts.append(m[p].astype(np.int64))
    if not pix:
        raise FileNotFoundError(f"No shapeHSM_*_ngals.hs maps in {shape_dir}")
    uniq, inv = np.unique(np.concatenate(pix), return_inverse=True)
    return uniq, np.bincount(inv, weights=np.concatenate(counts))


def completeness(pix, nside, count_pix, counts):
    """Sources within THETA_MAX of each cell over the number expected at the field's mean density.

    The field's mean density is the count on its own cells over their area, the field-level version of the
    notebook's N_target / A_tract.
    """
    pix_area = hp.nside2pixarea(nside, degrees=True) * 3600.0
    n_bar = lookup(count_pix, counts, pix, 0.0).sum() / (pix.size * pix_area)
    radius = np.deg2rad(THETA_MAX_ARCMIN / 60.0)
    n_ap = np.empty(pix.size)
    for k, p in enumerate(pix):
        disc = hp.query_disc(nside, hp.pix2vec(nside, int(p), nest=True), radius, nest=True)
        n_ap[k] = lookup(count_pix, counts, disc, 0.0).sum()
    return n_ap / (n_bar * np.pi * THETA_MAX_ARCMIN**2), n_bar


def find_peaks(pix, sn, comp, nside, sn_min):
    """Indices into pix of cells that are S/N maxima within EXCL_ARCMIN and pass the completeness and S/N cuts."""
    sn = np.where(np.isfinite(sn), sn, -np.inf)
    cand = np.flatnonzero((comp >= COMPLETENESS_MIN) & (sn >= sn_min))
    cand = cand[np.argsort(-sn[cand])]
    radius = np.deg2rad(EXCL_ARCMIN / 60.0)
    peaks = []
    for k in cand:
        disc = hp.query_disc(nside, hp.pix2vec(nside, int(pix[k]), nest=True), radius, nest=True)
        if np.all(lookup(pix, sn, disc, -np.inf) <= sn[k]):
            peaks.append(k)
    return np.asarray(peaks, dtype=np.int64)


def nearest_bmode_peaks(e_idx, pix, sn_b, comp, nside):
    """Nearest B-mode peak (either sign, |S/N_B| >= B_SN_FLAG) to each E-mode peak. Never a cut."""
    b_idx = np.concatenate([
        find_peaks(pix, sn_b, comp, nside, B_SN_FLAG),
        find_peaks(pix, -sn_b, comp, nside, B_SN_FLAG),
    ])
    out = {key: np.full(e_idx.size, np.nan) for key in ("B_PEAK_SN", "B_PEAK_RA", "B_PEAK_DEC", "B_PEAK_SEP_ARCMIN")}
    if e_idx.size and b_idx.size:
        vec_e = np.array(hp.pix2vec(nside, pix[e_idx], nest=True)).T
        vec_b = np.array(hp.pix2vec(nside, pix[b_idx], nest=True)).T
        sep = np.rad2deg(np.arccos(np.clip(vec_e @ vec_b.T, -1.0, 1.0))) * 60.0
        j = np.argmin(sep, axis=1)
        ra_b, dec_b = hp.pix2ang(nside, pix[b_idx[j]], nest=True, lonlat=True)
        out.update(B_PEAK_SN=sn_b[b_idx[j]], B_PEAK_RA=ra_b, B_PEAK_DEC=dec_b, B_PEAK_SEP_ARCMIN=sep[np.arange(j.size), j])
    out["B_PEAK_MATCH"] = out["B_PEAK_SEP_ARCMIN"] <= B_MATCH_ARCMIN
    return out, b_idx.size


def run_field(field, e_path, b_path, count_pix, counts, out_dir):
    e_map, pix, sn_e = read_map(e_path)
    _, b_pix, b_vals = read_map(b_path)
    sn_b = lookup(b_pix, b_vals, pix, np.nan)
    nside = e_map.nside_sparse

    comp, n_bar = completeness(pix, nside, count_pix, counts)
    comp_map = hsp.HealSparseMap.make_empty(e_map.nside_coverage, nside, np.float64)
    comp_map[pix] = comp
    comp_map.write(str(out_dir / f"{field}_completeness.hs"), clobber=True)

    e_idx = find_peaks(pix, sn_e, comp, nside, SN_MIN)
    bmatch, n_b = nearest_bmode_peaks(e_idx, pix, sn_b, comp, nside)

    # Chance that a random position has a B-mode peak within the match radius (Poisson).
    area_good = np.sum(comp >= COMPLETENESS_MIN) * hp.nside2pixarea(nside, degrees=True)
    b_density = n_b / (area_good * 3600.0) if area_good else np.nan  # per arcmin^2
    b_chance = 1.0 - np.exp(-b_density * np.pi * B_MATCH_ARCMIN**2)

    ra, dec = hp.pix2ang(nside, pix[e_idx], nest=True, lonlat=True)
    peaks = Table({
        "IPIX": pix[e_idx],
        "RA": ra,
        "DEC": dec,
        "E_SN": sn_e[e_idx],
        "B_SN": sn_b[e_idx],
        "COMPLETENESS": comp[e_idx],
        "B_FLAG": np.abs(sn_b[e_idx]) >= B_SN_FLAG,
        **bmatch,
    })
    peaks.meta.update({
        "FIELD": field, "E_MAP": e_path.name, "B_MAP": b_path.name, "NEST": True,
        "RS_ARCMIN": RS_ARCMIN, "NSIDE": nside, "SN_MIN": SN_MIN, "COMP_MIN": COMPLETENESS_MIN,
        "EXCL_ARCMIN": EXCL_ARCMIN, "THMAX_ARCMIN": THETA_MAX_ARCMIN, "NBAR_ARCMIN2": n_bar,
        "B_SN_FLAG": B_SN_FLAG, "B_MATCH_ARCMIN": B_MATCH_ARCMIN,
        "B_DENSITY_DEG2": b_density * 3600.0, "B_CHANCE": b_chance,
    })
    peaks.write(out_dir / f"{field}_peaks.fits", overwrite=True)
    print(f"{field}: {len(peaks)} peaks, {n_b} B-mode peaks, n_bar = {n_bar:.2f}/arcmin^2, "
          f"{area_good:.2f} deg^2 with completeness >= {COMPLETENESS_MIN}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map-dir", type=Path, default=Path(MAP_DIR), help="directory with the field E/B maps")
    parser.add_argument("--shape-dir", type=Path, help="per-tract shapeHSM maps (default: MAP_DIR/shapeHSM)")
    parser.add_argument("--out-dir", type=Path, help="where to write the outputs (default: MAP_DIR)")
    args = parser.parse_args()
    shape_dir = args.shape_dir or args.map_dir / "shapeHSM"
    out_dir = args.out_dir or args.map_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    count_pix, counts = load_counts(shape_dir)
    for e_suffix, b_suffix in E_SUFFIXES.items():
        for e_path in sorted(args.map_dir.glob(f"*{e_suffix}")):
            field = e_path.name[: -len(e_suffix)]
            b_path = e_path.with_name(field + b_suffix)
            if not b_path.exists():
                print(f"{field}: no {b_path.name}, skipped")
                continue
            run_field(field, e_path, b_path, count_pix, counts, out_dir)


if __name__ == "__main__":
    main()
