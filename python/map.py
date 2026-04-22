"""
Binned Schirmer-filtered aperture E/B S/N maps from Butler `object_shear_all`.

Re-implements the core pipeline of `schirmer_snr_weight.py` with:
  - Virtual tract pixels from RA/Dec (no catalog x/y columns).
  - Butler access matching `dp2_tract9813_metadetect_massmap.ipynb`.
  - All settings in CONFIG below (no command-line arguments).

Run inside an LSST stack environment with access to the configured Butler repo.
"""

import multiprocessing as mp
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from lsst.daf.butler import Butler
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration — edit here only
# ---------------------------------------------------------------------------
#
# Virtual x,y from RA/Dec use PIX_SCALE_ARCSEC (arcsec per virtual pixel).
# RS_INPUT_PIX and BIN_SIZE_NATIVE_PIX use that same virtual-pixel length unit.
# ---------------------------------------------------------------------------

CONFIG = {
    # Butler: `object_shear_all` for each tract in TRACTS
    "REPO": "/sdf/data/rubin/repo/dp2_prep",  # Gen3 Butler repository root (path)
    "COLLECTION": "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage3",  # run collection
    "SKYMAP": "lsst_cells_v2",  # skymap dataset type registered in the repo
    "INSTRUMENT": "LSSTCam",  # instrument name for Butler dataId
    "TRACTS": [9813],  # LSST skymap tract ID(s) (unitless)
    # Schirmer M_ap on a binned grid: Rs and bin width in native virtual pixels
    "RS_INPUT_PIX": 10_000.0,  # Schirmer aperture scale R_ap [virtual pixels]
    "PIX_SCALE_ARCSEC": 0.2,  # plate scale for RA/Dec → x,y [arcsec / virtual pixel]
    "BIN_SIZE_NATIVE_PIX": 32.0,  # 2D bin size [virtual pixels], same length unit as x,y
    # Parallelism: worker processes for the per-cell aperture loop (count)
    "N_WORKERS": max(1, mp.cpu_count() - 1),
    # Figures: PDF paths and colormap stretch from |S/N| (both E and B panels)
    "FIGURE_DIR": "/sdf/home/a/abrought/dp2/dp2-clusters-shear-mapping/sh/test_tract_output",  # output directory for PDFs (and optional peak .txt)
    "SYM_PERCENTILE": 95.0,  # percentile of |S/N| for symmetric vmin/vmax [0–100, %]
    "WRITE_PEAK_SUMMARY": False,  # if True, write one-line E-mode peak (x,y,Rs,S/N) .txt
}

# ---------------------------------------------------------------------------
# Schirmer filter and binned statistics (match schirmer_binned notebook)
# ---------------------------------------------------------------------------


def schirmer_weight(r, rs):
    """Schirmer Q(r/Rs); r and rs in the same units (here: bin-index separation)."""
    r = np.asarray(r, dtype=np.float64)
    rs = float(rs)
    x = r / rs
    a, b, c, d, xc = 6.0, 150.0, 47.0, 50.0, 0.15
    q = 1.0 / (1.0 + np.exp(a - b * x) + np.exp(d * x - c))
    ratio = x / xc
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = np.tanh(ratio) / ratio
    inner = np.where(np.abs(ratio) < 1e-14, 1.0, inner)
    q = q * inner
    return np.where(np.isfinite(q), q, 0.0)


def weighted_mean_2d(x, y, values, weights, x_edges, y_edges):
    """Weighted mean per 2D bin; transpose so rows follow y (Fu script convention)."""
    num, _, _, _ = stats.binned_statistic_2d(
        x,
        y,
        values * weights,
        statistic="sum",
        bins=[x_edges, y_edges],
    )
    den, _, _, _ = stats.binned_statistic_2d(
        x,
        y,
        weights,
        statistic="sum",
        bins=[x_edges, y_edges],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return out.T


# Pool workers: initializer so child processes get binned grids (spawn-safe).

_AP_XV = None
_AP_YV = None
_AP_E1 = None
_AP_E2 = None
_AP_ESQ = None
_AP_RS = None


def _init_ap_worker(xv, yv, e1_binned, e2_binned, e_sq_binned, rs):
    global _AP_XV, _AP_YV, _AP_E1, _AP_E2, _AP_ESQ, _AP_RS
    _AP_XV = xv
    _AP_YV = yv
    _AP_E1 = e1_binned
    _AP_E2 = e2_binned
    _AP_ESQ = e_sq_binned
    _AP_RS = rs


def _compute_m_ap_at_pixel(ind):
    """One grid cell: M_ap_E, M_ap_B, n_M_ap (noise denominator)."""
    assert _AP_XV is not None and _AP_RS is not None
    row, col = int(ind[0]), int(ind[1])
    xv, yv = _AP_XV, _AP_YV
    e1_b, e2_b, esq = _AP_E1, _AP_E2, _AP_ESQ
    rs = _AP_RS

    qw = schirmer_weight(np.sqrt((xv - col) ** 2 + (yv - row) ** 2), rs)
    dx, dy = xv - col, yv - row
    ang = np.arctan2(dy, dx)
    et = -e1_b * np.cos(2.0 * ang) - e2_b * np.sin(2.0 * ang)
    ex = +e1_b * np.sin(2.0 * ang) - e2_b * np.cos(2.0 * ang)
    m_e = np.nansum(qw * et)
    m_b = np.nansum(qw * ex)
    n_map = np.sqrt(np.nansum((qw**2) * esq)) / np.sqrt(2.0)
    return float(m_e), float(m_b), float(n_map)


# ---------------------------------------------------------------------------
# Data: Butler load, cuts, virtual pixels
# ---------------------------------------------------------------------------


def load_shear_table(butler, tract):
    return butler.get("object_shear_all", dataId={"tract": tract})


def apply_quality_mask(data):
    mask = data["metaStep"] == "ns"
    mask &= data["image_flags"] == 0
    mask &= data["psfOriginal_flags"] == 0
    mask &= data["bmask_flags"] == 0
    mask &= data["ormask_flags"] == 0
    mask &= data["mfrac"] < 0.1
    return data[mask]


def table_to_xy_g_weights(tbl, pix_scale_arcsec):
    """Virtual pixels from RA/Dec at tract median; weights = 1 / (Var g1 + Var g2)."""
    ra = np.asarray(tbl["ra"], dtype=np.float64)
    dec = np.asarray(tbl["dec"], dtype=np.float64)
    e1 = np.asarray(tbl["gauss_g1"], dtype=np.float64)
    e2 = np.asarray(tbl["gauss_g2"], dtype=np.float64)
    vsum = np.asarray(tbl["gauss_g1_g1_Cov"], dtype=np.float64) + np.asarray(
        tbl["gauss_g2_g2_Cov"], dtype=np.float64
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.isfinite(vsum) & (vsum > 0.0), 1.0 / vsum, 0.0)

    ra0 = float(np.median(ra))
    dec0 = float(np.median(dec))
    cos_dec0 = np.cos(np.deg2rad(dec0))
    ps = float(pix_scale_arcsec)
    x = (ra - ra0) * cos_dec0 * (3600.0 / ps)
    y = (dec - dec0) * (3600.0 / ps)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e1) & np.isfinite(e2) & (w > 0)
    return x[ok], y[ok], e1[ok], e2[ok], w[ok]


# ---------------------------------------------------------------------------
# Maps and figures
# ---------------------------------------------------------------------------


def symmetric_color_limit(snr_e, snr_b, percentile):
    """Max abs S/N at nanpercentile, for diverging colormap (massmap / binned-notebook spirit)."""
    z = np.concatenate([np.abs(snr_e.ravel()), np.abs(snr_b.ravel())])
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 1.0
    t = float(np.nanpercentile(z, percentile))
    return t if t > 0 else 1.0


def run_aperture_maps(x, y, e1, e2, weight, bin_size, rs_input_pix, n_workers):
    """Returns x_edges, y_edges, SNR_E, SNR_B, n_M_ap."""
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    ncol = int(np.ceil((x_max - x_min) / bin_size))
    nrow = int(np.ceil((y_max - y_min) / bin_size))
    x_edges = x_min + np.arange(ncol + 1) * bin_size
    y_edges = y_min + np.arange(nrow + 1) * bin_size

    xv, yv = np.meshgrid(np.arange(ncol), np.arange(nrow))
    coord_list = list(zip(yv.ravel().tolist(), xv.ravel().tolist()))
    rs_bins = rs_input_pix / bin_size

    e1_b = weighted_mean_2d(x, y, e1, weight, x_edges, y_edges)
    e2_b = weighted_mean_2d(x, y, e2, weight, x_edges, y_edges)
    e_sq_b = weighted_mean_2d(x, y, e1**2 + e2**2, weight**2, x_edges, y_edges)

    n_workers = max(1, int(n_workers))
    with mp.Pool(
        processes=n_workers,
        initializer=_init_ap_worker,
        initargs=(xv, yv, e1_b, e2_b, e_sq_b, rs_bins),
    ) as pool:
        result = pool.map(_compute_m_ap_at_pixel, coord_list)

    arr = np.asarray(result, dtype=np.float64)
    m_e = arr[:, 0].reshape((nrow, ncol))
    m_b = arr[:, 1].reshape((nrow, ncol))
    n_m = arr[:, 2].reshape((nrow, ncol))
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_e = m_e / n_m
        snr_b = m_b / n_m
    return x_edges, y_edges, snr_e, snr_b, n_m


def save_eb_snr_figure(x_edges, y_edges, snr_e, snr_b, out_pdf, *, tract, rs_pix, bin_size, sym_percentile):
    """Massmap-style two-panel imshow; PDF only."""
    thr = symmetric_color_limit(snr_e, snr_b, sym_percentile)
    extent = (float(np.min(x_edges)), float(np.max(x_edges)), float(np.min(y_edges)), float(np.max(y_edges)))

    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    im0 = axes[0].imshow(
        np.flipud(snr_e),
        vmin=-thr,
        vmax=thr,
        cmap="RdBu_r",
        extent=extent,
        aspect="auto",
    )
    axes[0].set_xlabel("x [virtual pix]")
    axes[0].set_ylabel("y [virtual pix]")
    axes[0].set_title(r"E-mode $M_{\rm ap}$ S/N")

    im1 = axes[1].imshow(
        np.flipud(snr_b),
        vmin=-thr,
        vmax=thr,
        cmap="RdBu_r",
        extent=extent,
        aspect="auto",
    )
    axes[1].set_xlabel("x [virtual pix]")
    axes[1].set_ylabel("y [virtual pix]")
    axes[1].set_title(r"B-mode $M_{\rm ap}$ S/N")

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02, label="S/N")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02, label="S/N")

    fig.suptitle(
        rf"Tract {tract} | Schirmer $R_{{\rm ap}}$ = {rs_pix:.0f} pix | bin = {bin_size:.0f} pix | "
        rf"$|S/N|_{{{sym_percentile:.0f}\%}}$ = {thr:.2f}",
        y=1.02,
    )
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def peak_e_snr_virtual_xy(snr_e, x_min, y_min, bin_size, rs_input_pix):
    """Global max of E-mode S/N map; virtual (x,y) at bin center."""
    mat = snr_e
    fin = mat[np.isfinite(mat)]
    if fin.size == 0:
        return None
    max_val = float(np.nanmax(mat))
    idx = np.where(mat == max_val)
    if idx[0].size == 0:
        return None
    row_peak, col_peak = int(idx[0][0]), int(idx[1][0])
    x_out = x_min + col_peak * bin_size + 0.5 * bin_size
    y_out = y_min + row_peak * bin_size + 0.5 * bin_size
    return x_out, y_out, float(rs_input_pix), max_val


def main():
    fig_dir = Path(CONFIG["FIGURE_DIR"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    butler = Butler(
        CONFIG["REPO"],
        collections=[CONFIG["COLLECTION"]],
        skymap=CONFIG["SKYMAP"],
        instrument=CONFIG["INSTRUMENT"],
    )

    bin_size = float(CONFIG["BIN_SIZE_NATIVE_PIX"])
    rs_pix = float(CONFIG["RS_INPUT_PIX"])
    ps = float(CONFIG["PIX_SCALE_ARCSEC"])
    sym_pct = float(CONFIG["SYM_PERCENTILE"])
    n_workers = int(CONFIG["N_WORKERS"])

    for tract in CONFIG["TRACTS"]:
        tract = int(tract)
        print(f"Tract {tract}: loading object_shear_all …")
        data = load_shear_table(butler, tract)
        shear = apply_quality_mask(data)
        print(f"  rows after cuts: {len(shear):,}")

        x, y, e1, e2, w = table_to_xy_g_weights(shear, ps)
        print(f"  galaxies with positive weight: {x.size:,}")

        x_edges, y_edges, snr_e, snr_b, _n_m = run_aperture_maps(
            x, y, e1, e2, w, bin_size, rs_pix, n_workers
        )

        pdf_name = f"schirmer_snr_tract{tract}_b{int(bin_size)}_Rs{int(rs_pix)}.pdf"
        out_pdf = fig_dir / pdf_name
        save_eb_snr_figure(
            x_edges,
            y_edges,
            snr_e,
            snr_b,
            out_pdf,
            tract=tract,
            rs_pix=rs_pix,
            bin_size=bin_size,
            sym_percentile=sym_pct,
        )
        print(f"  saved {out_pdf}")

        if CONFIG.get("WRITE_PEAK_SUMMARY"):
            x_min = float(np.min(x_edges))
            y_min = float(np.min(y_edges))
            pk = peak_e_snr_virtual_xy(snr_e, x_min, y_min, bin_size, rs_pix)
            peak_path = fig_dir / f"schirmer_snr_tract{tract}_b{int(bin_size)}_Rs{int(rs_pix)}_peak.txt"
            if pk is not None:
                x_out, y_out, rs_o, snr_max = pk
                peak_path.write_text(
                    f"# x y Rs SNR_E_max (virtual pixels)\n{x_out} {y_out} {rs_o} {snr_max}\n"
                )
                print(f"  peak summary {peak_path}")


if __name__ == "__main__":
    main()
