# --- Seam check: run after run_aperture_map on a tract that crosses RA = 0 (e.g. 9226, 8742) ---
theta_max_deg = theta_max_arcmin / 60.0
ra_c, dec_c = hp.pix2ang(nside, ipix_eval, nest=False, lonlat=True)
x_seam = wrap_dra(ra_c, 0.0) * np.cos(np.deg2rad(dec_c)) * 60.0  # arcmin east (+) / west (-) of RA = 0

n_east = int(np.sum(wrap_dra(ra_deg, 0.0) >= 0.0))
print(f"sources east / west of RA = 0: {n_east:,} / {ra_deg.size - n_east:,}")

# Aperture counts the pre-fix way (flat RA/Dec tree, unwrapped offsets), near the seam only.
near = np.abs(x_seam) < 2.0 * theta_max_arcmin
_flat = cKDTree(np.column_stack([ra_deg, dec_deg]))
old_counts = counts_map.astype(np.int64).copy()
for k in np.flatnonzero(near):
    rc, dc = ra_c[k], dec_c[k]
    i = np.asarray(_flat.query_ball_point([rc, dc], r=theta_max_deg / np.cos(np.deg2rad(dc))), dtype=np.int64)
    old_counts[k] = 0 if i.size == 0 else np.sum(
        np.hypot((ra_deg[i] - rc) * np.cos(np.deg2rad(dc)), dec_deg[i] - dc) <= theta_max_deg)
old_compl = old_counts / (N_DENSITY * np.pi * theta_max_arcmin**2)

band = np.abs(x_seam) < theta_max_arcmin   # apertures that reach across the seam
ctrl = near & ~band                        # just outside that reach
at0 = np.abs(x_seam) < 2.0                 # within 2' of RA = 0
diff = old_counts != counts_map
print(f"cells evaluated: {ipix_eval.size:,}; within {theta_max_arcmin:.0f}' of seam: {band.sum():,}")
print(f"median completeness  |x|<{theta_max_arcmin:.0f}': new {np.median(completeness_map[band]):.3f}, old {np.median(old_compl[band]):.3f}")
print(f"median completeness  {theta_max_arcmin:.0f}'-{2*theta_max_arcmin:.0f}': new {np.median(completeness_map[ctrl]):.3f}, old {np.median(old_compl[ctrl]):.3f}")
print(f"old / new counts within 2' of RA = 0: median {np.median(old_counts[at0] / np.maximum(counts_map[at0], 1)):.2f}")
print(f"cells where old != new: {diff.sum():,}, all within {np.abs(x_seam[diff]).max() if diff.any() else 0:.1f}' of the seam")

# Completeness vs distance from the seam: the pre-fix curve dips at x = 0, the fixed one stays flat.
edges = np.arange(-2.0 * theta_max_arcmin, 2.0 * theta_max_arcmin + 1e-9, 2.0)
mid = 0.5 * (edges[1:] + edges[:-1])
_bin = np.digitize(x_seam, edges) - 1
def _med(v):
    return np.array([np.median(v[_bin == b]) if np.any(_bin == b) else np.nan for b in range(mid.size)])
fig_s, ax_s = plt.subplots(figsize=(7, 4))
ax_s.plot(mid, _med(completeness_map), "o-", label="fixed (wrapped RA, unit-sphere tree)")
ax_s.plot(mid, _med(old_compl), "s--", mfc="none", label="pre-fix (flat tree, raw RA offsets)")
for s in (-1, 1):
    ax_s.axvline(s * theta_max_arcmin, color="0.6", lw=0.8, ls=":")
ax_s.axvline(0.0, color="k", lw=0.8)
ax_s.set_xlabel("distance east of RA = 0 (arcmin)")
ax_s.set_ylabel("median aperture completeness")
ax_s.set_title(f"Tract {CONFIG['TRACT']}: RA = 0 seam check")
ax_s.legend()
fig_s.tight_layout()
