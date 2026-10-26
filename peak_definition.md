DP2 Shear Peak Definition

A HEALPix cell $p$ of the E-mode S/N map is a peak if it passes 1–3. Items 4–5 are recorded for every peak but are never cuts. Values live in the CONFIG block of each notebook.

1. **Local maximum:** $\mathcal{S}_E(p) \ge \mathcal{S}_E(q)$ for every evaluated cell $q$ within $r_{\rm excl} = 0.4\,R_s = 6'$ of $p$ (`PEAK_EXCL_RS_MULT`).

2. **Completeness:** $C(p) \ge 0.6$ (`PEAK_COMPLETENESS_MIN`). Completeness is the number of sources in the aperture divided by the number expected at the tract's mean source density:

$$
C(p) = \frac{N(\theta \le \theta_{\max})}{\bar n\,\pi\,\theta_{\max}^2}, \qquad \theta_{\max} = 1.2\,R_s = 18', \qquad \bar n = \frac{N_{\rm target}}{A_{\rm tract}} \;(\texttt{N\_DENSITY})
$$

   It falls below 1 where the aperture runs off the data (tract edges without neighbor tracts, survey edges, masks) and can exceed 1 where the local density is above the mean.

3. **Threshold:** $\mathcal{S}_E(p) \ge \mathcal{S}_{\min}$ (`PEAK_SN_MIN`): 4.0 in dp2_shearmap_PSZ2_G309.43-72.86, 3.5 in dp2-shear-map-tract.

4. **B-mode flag:** `B_FLAG` is true when $|\mathcal{S}_B(p)| \ge 3$ at the peak cell (`PEAK_B_SN_FLAG`).

5. **Nearest B-mode peak:** B-mode peaks are found with 1–2 applied to $+\mathcal{S}_B$ and $-\mathcal{S}_B$, keeping $|\mathcal{S}_B| \ge 3$. For each peak, the nearest one is recorded (`B_PEAK_SN`, `B_PEAK_RA`, `B_PEAK_DEC`, `B_PEAK_SEP_ARCMIN`), and `B_PEAK_MATCH` is true when it lies within $0.4\,R_s = 6'$ (`PEAK_B_MATCH_RS_MULT`). The chance of a match at a random position, $1 - e^{-n_B \pi r_{\rm match}^2}$, is stored in the header as `B_CHANCE`.

6. **Output:** peaks are written to `output/<tract>_shapeHSM_peaks.fits`, highest $\mathcal{S}_E$ first.
