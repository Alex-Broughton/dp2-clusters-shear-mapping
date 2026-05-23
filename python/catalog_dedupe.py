"""
Deduplicate shear catalog rows before binning (patch overlap / repeat detections).

--dedupe merges sources within sep_pix on tract (x, y), keeping the highest weight.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree

# Tract pixels: ~0.2 arcsec/pix at 0.2"/pix. Patch duplicates are typically < 1 pix apart.
DEFAULT_DEDUPE_SEP_PIX = 1.0


def _sort_weight(weight, n):
    w = np.asarray(weight, dtype=np.float64) if weight is not None else np.ones(n)
    return np.where(np.isfinite(w), w, -np.inf)


def _dedupe_position_mask(x, y, w_sort, sep_pix):
    """Greedy position dedupe; keeps highest weight first."""
    n = w_sort.size
    keep = np.zeros(n, dtype=bool)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pos_ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(pos_ok):
        return keep

    coords = np.column_stack([x, y])
    tree = cKDTree(coords[pos_ok])
    pos_idx = np.flatnonzero(pos_ok)

    for ii in np.argsort(-w_sort):
        if not pos_ok[ii]:
            continue
        nbr_local = tree.query_ball_point(coords[ii], r=float(sep_pix))
        if not any(keep[pos_idx[j]] for j in nbr_local if pos_idx[j] != ii):
            keep[ii] = True
    return keep


def apply_position_dedupe(x, y, weight, *, sep_pix=DEFAULT_DEDUPE_SEP_PIX):
    """Dedupe on quality-selected rows. Returns (keep_mask, n_removed)."""
    n = int(x.size)
    if n == 0:
        return np.ones(0, dtype=bool), 0
    w_sort = _sort_weight(weight, n)
    keep = _dedupe_position_mask(x, y, w_sort, sep_pix)
    return keep, int(n - keep.sum())


def load_shear_arrays(
    filename,
    *,
    weight_floor,
    dedupe=False,
    dedupe_sep_pix=DEFAULT_DEDUPE_SEP_PIX,
):
    """
    Load FITS table arrays for schirmer pipeline with optional position dedupe.

    Dedupe runs on quality-selected rows only (finite x,y,g1,g2,weight).

    Returns (x, y, e1, e2, weight, info_dict).
    """
    data = Table.read(filename, format="fits")
    n_total = len(data)

    x = np.asarray(data["x"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    e1 = np.asarray(data["gauss_g1"], dtype=np.float64)
    e2 = np.asarray(data["gauss_g2"], dtype=np.float64)
    cov = np.asarray(data["gauss_g1_g1_Cov"], dtype=np.float64) + np.asarray(
        data["gauss_g2_g2_Cov"], dtype=np.float64
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(np.isfinite(cov) & (cov > 0.0), 1.0 / cov, np.nan)
    weight[weight < weight_floor] = weight_floor

    ok = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(e1)
        & np.isfinite(e2)
        & np.isfinite(weight)
    )
    n_quality = int(ok.sum())
    x, y, e1, e2, weight = x[ok], y[ok], e1[ok], e2[ok], weight[ok]

    n_removed = 0
    if dedupe:
        keep, n_removed = apply_position_dedupe(x, y, weight, sep_pix=dedupe_sep_pix)
        x, y, e1, e2, weight = x[keep], y[keep], e1[keep], e2[keep], weight[keep]

    n_binning = int(x.size)
    info = {
        "n_table": n_total,
        "n_quality": n_quality,
        "n_binning": n_binning,
        # Legacy keys (older load_shear_arrays on SDF before quality-first dedupe)
        "n_after_dedupe": n_binning,
        "n_removed_dedupe": n_removed,
        "dedupe": bool(dedupe),
        "dedupe_sep_pix": float(dedupe_sep_pix),
    }
    return x, y, e1, e2, weight, info


def format_catalog_summary(cat_info):
    """One-line catalog stats; tolerates older info dicts missing n_quality."""
    n_table = cat_info.get("n_table", 0)
    n_quality = cat_info.get("n_quality", cat_info.get("n_after_dedupe", cat_info.get("n_binning", 0)))
    n_binning = cat_info.get("n_binning", n_quality)
    parts = [f"table {n_table:,}", f"quality {n_quality:,}", f"binning {n_binning:,}"]
    if cat_info.get("dedupe"):
        parts.append(f"dedupe removed {cat_info.get('n_removed_dedupe', 0):,}")
    return " | ".join(parts)


def parse_dedupe_flags(argv):
    """
    Parse dedupe CLI flags.

    Returns (dedupe_enabled, dedupe_sep_pix, remaining_argv).

    --dedupe           merge nearby (x, y) within sep [tract pixels]
    --dedupe-sep-pix   separation for position dedupe (default 1.0)
    """
    dedupe = False
    sep = DEFAULT_DEDUPE_SEP_PIX
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--dedupe":
            dedupe = True
            i += 1
        elif a == "--no-dedupe":
            dedupe = False
            i += 1
        elif a == "--dedupe-sep-pix" and i + 1 < len(argv):
            sep = float(argv[i + 1])
            i += 2
        else:
            rest.append(a)
            i += 1
    return dedupe, sep, rest
