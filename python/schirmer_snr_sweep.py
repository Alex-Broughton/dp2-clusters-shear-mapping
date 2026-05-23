"""
Parameter sweep for Schirmer aperture E/B maps (tract-native pixels).

Loads the shear FITS once, then runs many (Rs_input, bin_size) combinations,
writes a summary CSV, heatmaps, and per-case E/B aperture-mass maps under
sweep_output/maps/.

Example (SLURM / local):
  python schirmer_snr_sweep.py /path/to/shear_table_xmatch_gold.fits 32

Patch overlap: object_shear_all can place patch-local x,y in one tract frame, so
n_gal may show rectangular seams and duplicate detections. Use --dedupe for a
positional merge before interpreting peak counts.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import schirmer_snr_weight as ssw
from catalog_dedupe import format_catalog_summary, load_shear_arrays, parse_dedupe_flags

# ---------------------------------------------------------------------------
# Sweep grid — edit here
# ---------------------------------------------------------------------------
# Native tract pixels @ 0.2 arcsec/pix (see ssw.PIX_SCALE_ARCSEC).
# Rs_input: Schirmer R_ap in tract pixels (~6.7–10 arcmin for 2000–3000).
# rs_cells: R_ap / bin_size; 1 = filter spans ~one cell, 5 = wider smoothing.

RS_INPUT_LIST = (2000, 2500, 3000)
RS_CELLS_LIST = (1.0, 2.0, 3.0, 4.0, 5.0)


def build_sweep_grid():
    """(Rs_input, bin_size) pairs with bin_size = Rs / rs_cells."""
    return [(float(rs), float(rs) / float(rc)) for rs in RS_INPUT_LIST for rc in RS_CELLS_LIST]

WEIGHT_FLOOR = ssw.WEIGHT_FLOOR
PIX_SCALE_ARCSEC = ssw.PIX_SCALE_ARCSEC
ENABLE_DEDUPE = True  # set False or pass --no-dedupe on the command line


def _set_pool_globals(x_bin, y_bin, xv, yv, e1_b, e2_b, esq_b, rs_cells):
    ssw.x_bin = x_bin
    ssw.y_bin = y_bin
    ssw.xv = xv
    ssw.yv = yv
    ssw.e1_binned = e1_b
    ssw.e2_binned = e2_b
    ssw.e_sq_binned = esq_b
    ssw.Rs = rs_cells


def run_one_case(x, y, e1, e2, weight, rs_input, bin_size, cpu_num):
    """Bin, aperture mass, return maps + scalar metrics (no file plots)."""
    t0 = time.perf_counter()
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_bin = np.arange(x_min, x_max + bin_size, bin_size)
    y_bin = np.arange(y_min, y_max + bin_size, bin_size)
    ncol = int(np.ceil((x_max - x_min) / bin_size))
    nrow = int(np.ceil((y_max - y_min) / bin_size))
    xv, yv = np.meshgrid(np.arange(ncol), np.arange(nrow))
    coord_list = list(zip(yv.ravel().tolist(), xv.ravel().tolist()))
    rs_cells = rs_input / bin_size

    # Binned shear (uses ssw.x, ssw.y globals in get_bin_stat_weight)
    ssw.x = x
    ssw.y = y
    _set_pool_globals(x_bin, y_bin, xv, yv, None, None, None, rs_cells)
    e1_b = ssw.get_bin_stat_weight(e1, weight)
    e2_b = ssw.get_bin_stat_weight(e2, weight)
    esq_b = ssw.get_bin_stat_weight(e1**2 + e2**2, weight**2)
    _set_pool_globals(x_bin, y_bin, xv, yv, e1_b, e2_b, esq_b, rs_cells)

    n_gal, shape_std = ssw.compute_bin_diagnostic_maps(x, y, e1, e2, x_bin, y_bin)

    with Pool(cpu_num) as pool:
        result = pool.map(ssw.compute_M_ap_at_pixel, coord_list)
    arr = np.asarray(result, dtype=np.float64)
    m_e = arr[:, 0].reshape((nrow, ncol))
    m_b = arr[:, 1].reshape((nrow, ncol))
    n_m = arr[:, 2].reshape((nrow, ncol))
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_e = m_e / n_m
        snr_b = m_b / n_m

    se = snr_e.ravel()
    sb = snr_b.ravel()
    ok = np.isfinite(se) & np.isfinite(sb)
    if ok.sum() >= 2:
        pearson = float(np.corrcoef(se[ok], sb[ok])[0, 1])
    else:
        pearson = np.nan

    abs_se = np.abs(se[ok])
    metrics = {
        "rs_input_pix": rs_input,
        "bin_size_pix": bin_size,
        "rs_arcmin": rs_input * PIX_SCALE_ARCSEC / 60.0,
        "bin_arcsec": bin_size * PIX_SCALE_ARCSEC,
        "rs_cells": rs_cells,
        "nrow": nrow,
        "ncol": ncol,
        "n_cells": nrow * ncol,
        "n_gal_mean": float(np.nanmean(n_gal)),
        "n_gal_median": float(np.nanmedian(n_gal)),
        "shape_noise_std_mean": float(np.nanmean(shape_std)),
        "finite_snr_frac": float(np.isfinite(snr_e).sum()) / snr_e.size,
        "snr_e_max": float(np.nanmax(abs_se)) if abs_se.size else np.nan,
        "snr_e_p90": float(np.nanpercentile(abs_se, 90)) if abs_se.size else np.nan,
        "snr_e_rms": float(np.sqrt(np.nanmean(se[ok] ** 2))) if ok.sum() else np.nan,
        "pearson_snr_e_b": pearson,
        "elapsed_s": time.perf_counter() - t0,
    }
    maps = {"m_e": m_e, "m_b": m_b, "snr_e": snr_e, "snr_b": snr_b, "x_bin": x_bin, "y_bin": y_bin}
    return metrics, maps


def sweep_map_stem(out_dir, tag0, tag, rs_input, bin_size, rs_cells):
    maps_dir = out_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    return maps_dir / f"{tag0}_{tag}_Rs{int(rs_input)}_rc{rs_cells:g}_b{int(round(bin_size))}"


def plot_case_e_b_maps(maps, out_dir, tag0, rs_input, bin_size, rs_cells):
    """E/B M_ap and S/N maps for one sweep case (same style as schirmer_snr_weight)."""
    ssw.set_plot_context(
        tag0, rs_input, bin_size, maps["x_bin"], maps["y_bin"], folder_name_in=str(out_dir)
    )
    stem = sweep_map_stem(out_dir, tag0, "M_ap", rs_input, bin_size, rs_cells)
    ssw.plot_E_B(maps["m_e"], maps["m_b"], "M_ap", sym_percentile=95.0, out_prefix=stem)
    stem_snr = sweep_map_stem(out_dir, tag0, "M_ap_SNR", rs_input, bin_size, rs_cells)
    ssw.plot_E_B(maps["snr_e"], maps["snr_b"], "M_ap_SNR", sym_percentile=95.0, out_prefix=stem_snr)


def write_summary_csv(rows, out_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def plot_sweep_summary(rows, out_dir, tag0):
    """Heatmaps of SNR metrics vs (Rs/cell, Rs_input)."""
    rs_vals = sorted({r["rs_input_pix"] for r in rows})
    rc_vals = sorted({r["rs_cells"] for r in rows})
    nr, nc = len(rs_vals), len(rc_vals)

    def rc_index(rc):
        return int(np.argmin(np.abs(np.asarray(rc_vals) - rc)))

    def grid_for(key):
        g = np.full((nr, nc), np.nan)
        for r in rows:
            i = rs_vals.index(r["rs_input_pix"])
            j = rc_index(r["rs_cells"])
            g[i, j] = r[key]
        return g

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in zip(
        axes,
        ("snr_e_p90", "pearson_snr_e_b"),
        (r"$|{\rm SNR}_E|_{90}$", r"Pearson(SNR$_E$, SNR$_B$)"),
    ):
        g = grid_for(key)
        im = ax.imshow(
            g,
            aspect="auto",
            origin="lower",
            cmap="magma" if "p90" in key else "coolwarm",
            vmin=-0.5 if "pearson" in key else None,
            vmax=0.5 if "pearson" in key else None,
        )
        ax.set_xticks(range(nc))
        ax.set_xticklabels([f"{rc:g}" for rc in rc_vals])
        ax.set_yticks(range(nr))
        ax.set_yticklabels([str(int(r)) for r in rs_vals])
        ax.set_xlabel(r"$R_{\rm ap}$ / bin_size")
        ax.set_ylabel(r"$R_{\rm ap}$ input [tract pix]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{tag0} parameter sweep", y=1.02)
    out = out_dir / f"{tag0}_sweep_summary"
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    dedupe, dedupe_sep_pix, argv_rest = parse_dedupe_flags(sys.argv[1:])
    if not dedupe and ENABLE_DEDUPE:
        dedupe = True
    if len(argv_rest) not in (1, 2):
        print(
            "python schirmer_snr_sweep.py [--dedupe] [--no-dedupe] [--dedupe-sep-pix SEP] "
            "<fits_file> [n_cpus]"
        )
        sys.exit(1)

    fits_path = Path(argv_rest[0])
    cpu_num = int(argv_rest[1]) if len(argv_rest) == 2 else max(1, mp.cpu_count() - 1)
    tag0 = fits_path.name.split("_")[0]
    out_dir = fits_path.parent / "sweep_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading catalog once:", fits_path)
    x, y, e1, e2, weight, cat_info = load_shear_arrays(
        str(fits_path),
        weight_floor=WEIGHT_FLOOR,
        dedupe=dedupe,
        dedupe_sep_pix=dedupe_sep_pix,
    )
    print(f"  {format_catalog_summary(cat_info)}")

    grid = build_sweep_grid()
    print(
        f"Sweep: {len(grid)} (Rs, bin) pairs "
        f"(Rs in {RS_INPUT_LIST}, Rs/cell in {RS_CELLS_LIST}), {cpu_num} workers per run"
    )

    rows = []
    for i, (rs_in, bin_sz) in enumerate(grid, 1):
        print(
            f"[{i}/{len(grid)}] Rs={rs_in:.0f} pix ({rs_in * PIX_SCALE_ARCSEC / 60:.2f}'), "
            f"bin={bin_sz:.0f} pix ({bin_sz * PIX_SCALE_ARCSEC:.1f}\")"
        )
        metrics, maps = run_one_case(x, y, e1, e2, weight, rs_in, bin_sz, cpu_num)
        rows.append(metrics)
        plot_case_e_b_maps(maps, out_dir, tag0, rs_in, bin_sz, metrics["rs_cells"])
        print(
            f"    grid {metrics['nrow']}x{metrics['ncol']}, Rs/cell={metrics['rs_cells']:.1f}, "
            f"|SNR_E|_90={metrics['snr_e_p90']:.3f}, r_EB={metrics['pearson_snr_e_b']:.3f}, "
            f"t={metrics['elapsed_s']:.1f}s"
        )

    csv_path = out_dir / f"{tag0}_sweep_summary.csv"
    write_summary_csv(rows, csv_path)
    plot_sweep_summary(rows, out_dir, tag0)
    print("Wrote", csv_path)
    print("Wrote", out_dir / f"{tag0}_sweep_summary.png")
    print("E/B maps:", out_dir / "maps/")


if __name__ == "__main__":
    main()
