"""Tract-level metadetect shear → gridded shear → Kaiser–Squires κ maps.

Designed for import from notebooks and, later, cluster batch scripts. All
state is passed in explicitly (no global CONFIG here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


@dataclass(frozen=True)
class ShearColumnMap:
    """Resolved Arrow column names for one object_shear_patch schema."""

    ra: str
    dec: str
    g1: str
    g2: str
    weight: str
    noshear_mode: str  # "bool_ns" | "string_ns" | "string_noshear" | "none"
    noshear_col: str | None


def infer_shear_columns(schema: pa.Schema) -> ShearColumnMap:
    """Pick RA/Dec/shear/weight columns and noshear branch from Arrow schema."""

    names = set(schema.names)

    def pick(candidates: Sequence[str]) -> str | None:
        for c in candidates:
            if c in names:
                return c
        return None

    ra = pick(("coord_ra", "ra", "objectRa"))
    dec = pick(("coord_dec", "dec", "objectDec"))
    if ra is None or dec is None:
        raise ValueError(
            f"Could not infer RA/Dec columns from schema names: {sorted(names)[:40]}…"
        )

    g1 = pick(("shear_1", "g1", "mdet_g_1", "i_mdet_g_1", "r_mdet_g_1"))
    g2 = pick(("shear_2", "g2", "mdet_g_2", "i_mdet_g_2", "r_mdet_g_2"))
    if g1 is None or g2 is None:
        raise ValueError(
            f"Could not infer shear columns from schema names: {sorted(names)[:40]}…"
        )

    weight = pick(
        (
            "weight",
            "shear_weight",
            "mdet_weight",
            "i_weight",
            "r_weight",
            "object_weight",
        )
    )
    if weight is None:
        raise ValueError(
            f"Could not infer weight column from schema names: {sorted(names)[:40]}…"
        )

    noshear_mode = "none"
    noshear_col: str | None = None
    if "ns" in names:
        noshear_mode = "bool_ns"
        noshear_col = "ns"
    elif "shear_type" in names:
        noshear_col = "shear_type"
        # Prefer short metadetect code "ns" if present in data; else "noshear".
        noshear_mode = "string_ns"
    elif "mdet_step" in names:
        noshear_col = "mdet_step"
        noshear_mode = "string_ns"

    return ShearColumnMap(
        ra=ra,
        dec=dec,
        g1=g1,
        g2=g2,
        weight=weight,
        noshear_mode=noshear_mode,
        noshear_col=noshear_col,
    )


def columns_for_loading(cmap: ShearColumnMap) -> list[str]:
    """Minimal Arrow columns to read from Butler for this analysis."""

    cols = [cmap.ra, cmap.dec, cmap.g1, cmap.g2, cmap.weight]
    if cmap.noshear_col is not None:
        cols.append(cmap.noshear_col)
    return list(dict.fromkeys(cols))


def filter_noshear_table(table: pa.Table, cmap: ShearColumnMap) -> pa.Table:
    """Keep only metadetect noshear rows (schema column ``ns`` or shear_type)."""

    if cmap.noshear_mode == "none":
        raise ValueError(
            "No noshear discriminator found (expected column 'ns' or 'shear_type' / "
            "'mdet_step'). Inspect object_shear_patch.schema in Butler."
        )

    if cmap.noshear_mode == "bool_ns":
        col = table[cmap.noshear_col]  # type: ignore[index]
        t = col.type
        if pa.types.is_boolean(t):
            mask = pc.equal(col, True)
        elif pa.types.is_string(t) or pa.types.is_large_string(t):
            mask = pc.equal(col, pa.scalar("ns"))
        elif pa.types.is_integer(t):
            # Unlikely, but allow integer flag.
            mask = pc.equal(col, pa.scalar(1, type=t))
        else:
            raise TypeError(f"Unexpected type for 'ns' column: {t}")
        return table.filter(mask)

    assert cmap.noshear_col is not None
    col = table[cmap.noshear_col]
    # Try common string labels for noshear branch.
    m1 = pc.equal(col, pa.scalar("ns"))
    m2 = pc.equal(col, pa.scalar("noshear"))
    mask = pc.or_(m1, m2)
    return table.filter(mask)


def list_patches_for_tract(butler: Any, tract: int, collection: str) -> list[int]:
    """Return sorted patch indices that have ``object_shear_patch`` for ``tract``."""

    patches: set[int] = set()
    last_err: Exception | None = None

    # Some stack versions expose query_datasets on Butler instead of Registry.
    for meth_name in ("query_datasets", "queryDatasets"):
        fn = getattr(butler, meth_name, None)
        if fn is None:
            continue
        try:
            refs = fn(
                "object_shear_patch",
                where=f"tract = {int(tract)}",
                collections=[collection],
            )
            for ref in refs:
                patches.add(int(ref.dataId["patch"]))
            if patches:
                return sorted(patches)
        except Exception as exc:  # noqa: BLE001 — stack API differences
            last_err = exc

    registry = butler.registry
    query_specs: list[dict[str, Any]] = [
        {"collections": [collection], "where": f"tract = {int(tract)}"},
        {
            "collections": [collection],
            "bind": {"tract": int(tract)},
            "where": "tract = tract",
        },
    ]
    for spec in query_specs:
        try:
            refs = registry.queryDatasets("object_shear_patch", **spec)
            for ref in refs:
                patches.add(int(ref.dataId["patch"]))
            if patches:
                return sorted(patches)
        except Exception as exc:  # noqa: BLE001 — stack API differences
            last_err = exc
            continue
    if last_err is not None:
        raise RuntimeError(
            f"Could not query object_shear_patch for tract={tract} "
            f"in collection={collection!r}: {last_err!r}"
        ) from last_err
    return sorted(patches)


def load_patches_concat(
    butler: Any,
    tract: int,
    patches: Iterable[int],
    *,
    column_subset: Sequence[str] | None = None,
    max_patches: int | None = None,
) -> pa.Table:
    """Load ``object_shear_patch`` per patch and concatenate (column subset optional)."""

    tables: list[pa.Table] = []
    for i, patch in enumerate(patches):
        if max_patches is not None and i >= max_patches:
            break
        data_id = {"tract": int(tract), "patch": int(patch)}
        if not butler.exists("object_shear_patch", dataId=data_id):
            continue
        tbl = butler.get("object_shear_patch", dataId=data_id)
        if column_subset is not None:
            tbl = tbl.select(column_subset)
        tables.append(tbl)
    if not tables:
        return pa.table({})
    return pa.concat_tables(tables, promote_options="default")


def arrow_to_numpy_shear(
    table: pa.Table, cmap: ShearColumnMap
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract RA/Dec (deg), g1, g2, weight as float64 numpy arrays."""

    ra = table[cmap.ra].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    dec = table[cmap.dec].combine_chunks().to_numpy(zero_copy_only=False).astype(
        np.float64
    )
    g1 = table[cmap.g1].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    g2 = table[cmap.g2].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    w = table[cmap.weight].combine_chunks().to_numpy(zero_copy_only=False).astype(
        np.float64
    )
    return ra, dec, g1, g2, w


def select_sources_basic(
    ra: np.ndarray,
    dec: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    w: np.ndarray,
    *,
    min_weight: float = 0.0,
    max_abs_g: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Conservative quality cuts (tune CONFIG in notebook)."""

    finite = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(g1) & np.isfinite(g2) & np.isfinite(w)
    pos_w = w > min_weight
    g_ok = (np.abs(g1) < max_abs_g) & (np.abs(g2) < max_abs_g)
    m = finite & pos_w & g_ok
    return ra[m], dec[m], g1[m], g2[m], w[m]


def bin_shear_weighted(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    w: np.ndarray,
    *,
    ra0_deg: float,
    dec0_deg: float,
    pixel_scale_arcsec: float,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """Tangent TAN projection at (ra0, dec0); nearest-pixel binning of weighted shear.

    Returns (g1_map, g2_map, w_map, hit_count, wcs) where w_map is sum of weights
    per cell and g* maps are sum(w*g*)/sum(w) with 0 where empty.
    """

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS

    pixel_deg = pixel_scale_arcsec / 3600.0
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [float(ra0_deg), float(dec0_deg)]
    wcs.wcs.crpix = [float(nx) / 2.0 + 0.5, float(ny) / 2.0 + 0.5]
    wcs.wcs.cd = np.array([[-pixel_deg, 0.0], [0.0, pixel_deg]])

    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    xpix, ypix = wcs.world_to_pixel(coord)

    ix = np.floor(xpix + 0.5).astype(np.int64)
    iy = np.floor(ypix + 0.5).astype(np.int64)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy, g1, g2, w = ix[inside], iy[inside], g1[inside], g2[inside], w[inside]

    idx = iy * nx + ix
    w_flat = np.bincount(idx, weights=w, minlength=nx * ny).reshape(ny, nx)
    wg1 = np.bincount(idx, weights=w * g1, minlength=nx * ny).reshape(ny, nx)
    wg2 = np.bincount(idx, weights=w * g2, minlength=nx * ny).reshape(ny, nx)
    hits = np.bincount(idx, minlength=nx * ny).reshape(ny, nx)

    with np.errstate(invalid="ignore", divide="ignore"):
        g1_map = np.where(w_flat > 0, wg1 / w_flat, 0.0)
        g2_map = np.where(w_flat > 0, wg2 / w_flat, 0.0)
    return g1_map, g2_map, w_flat, hits, wcs


def kaiser_squires(
    g1: np.ndarray,
    g2: np.ndarray,
    *,
    pixel_scale_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """FFT Kaiser–Squires E/B κ from gridded shear (Bartelmann & Schneider convention).

    Parameters
    ----------
    g1, g2 : (ny, nx)
        Reduced shear components on a regular grid (mean γ per cell is fine).
    pixel_scale_rad : float
        Pixel width in radians (square pixels).

    Returns
    -------
    kappa_e, kappa_b : (ny, nx)
        Real-space E-mode and B-mode convergence maps (B should be noise-like for
        a true mass map; non-zero B indicates PSF leakage, binning artifacts, etc.).
    """

    ny, nx = g1.shape
    g1f = np.fft.fft2(g1)
    g2f = np.fft.fft2(g2)

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=pixel_scale_rad)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=pixel_scale_rad)
    kx2d, ky2d = np.meshgrid(kx, ky)
    k2 = kx2d**2 + ky2d**2
    k2_safe = np.where(k2 > 0, k2, 1.0)

    q = (kx2d**2 - ky2d**2) / k2_safe
    u = (2.0 * kx2d * ky2d) / k2_safe

    kappa_e_f = q * g1f + u * g2f
    kappa_b_f = u * g1f - q * g2f
    kappa_e_f[0, 0] = 0.0
    kappa_b_f[0, 0] = 0.0

    kappa_e = np.real(np.fft.ifft2(kappa_e_f))
    kappa_b = np.real(np.fft.ifft2(kappa_b_f))
    return kappa_e, kappa_b
