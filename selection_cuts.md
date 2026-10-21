DP2 Shear Starting Cuts

A set of beginning cuts to get started on looking through the DP2 Metadetect shear catalogs. The cuts actually applied in this repo (object table + HSM regauss shapes) are summarized at the end.

Minimal Cuts

Start here! Especially while we are getting cuts settled and exploring the data.

Set all _flags columns to false / 0.

```
{'bmask_flags',
 'g_pgaussFlux_flags',
 'gauss_flags',
 'gauss_object_flags',
 'gauss_psfReconvolved_flags',
 'gauss_shape_flags',
 'i_gaussFlux_flags',
 'i_pgaussFlux_flags',
 'image_flags',
 'pgauss_flags',
 'pgauss_object_flags',
 'pgauss_shape_flags',
 'psfOriginal_flags',
 'r_gaussFlux_flags',
 'r_pgaussFlux_flags',
 'z_gaussFlux_flags',
 'z_pgaussFlux_flags'}
 ```

Set is_primary==True (deduplication flag)

* Objects set to true will be unique to their regions (tract, patch, and cell level)


Set mfrac < 0.1

* Cut objects with high levels of masked pixels 

Typical Cuts

These cuts are more focused on for quality of the objects, such as SNR and star-galaxy separation. These are in flux and will likely change as we learn more about the data. Initial values based on section 4.4 in http://arxiv.org/abs/2501.05665.


* SNR > 10
* reso_factor > 0.5
    * Star-galaxy separation by comparing sizes of objects compared to size of the PSF. Stars are generally the same size as the PSF and will be centered on gauss_T=0 (centered on 0 rather than 1 as gauss measures pre-PSF).
* Color cuts
    * |g-r| < 5 
    * |r-i| < 5 
    * |i-z| < 5 
* Size cut (potentially, requires visual inspection in LSST data first)
    * gauss_T < 20 

Convert flux to magnitudes using (flux_column*u.nJy).to(u.ABmag)

```
shear_table[f'g'] = (t1[f'g_sersicFlux']*u.nJy).to(u.ABmag)
shear_table[f'r'] = (t1[f'r_sersicFlux']*u.nJy).to(u.ABmag)
shear_table[f'i'] = (t1[f'i_sersicFlux']*u.nJy).to(u.ABmag)
shear_table[f'z'] = (t1[f'z_sersicFlux']*u.nJy).to(u.ABmag)
```

Summary: cuts used in this repo

The notebooks (dp2-shear-map-tract, dp2_shearmap_PSZ2_G309.43-72.86, dp2_tract_stats) and python/sky_coverage.py use the object table and HSM regauss shapes, so each cut above is applied through its object-table counterpart. All values live in the CONFIG block of each file.

* Flags: i-band measurement, PSF, shape, GAaP, and Sersic flags false, plus i_pixelFlags_* (bad, nodata, edge, saturatedCenter, crCenter, interpolatedCenter, suspectCenter). There is no per-object mfrac in the object table.
* Primary: detect_isDeblendedModelSource == True
* S/N: griz Sersic flux S/N > 5 in every band (SNR_MIN)
* Star-galaxy separation
    * reso_factor = 1 - T_PSF / T_obj from the i-band second moments > 0.125 (RESO_FACTOR_MIN)
    * griz_model_extendedness > 0.2 (EXTENDEDNESS_MIN)
* Shape: e1^2 + e2^2 <= 4 for the HSM regauss distortion (E_SQ_MAX)
* Size: optional cut on i_ixx + i_iyy (T_MAX_ARCSEC2), disabled by default
* Milky Way dust correction (DEREDDEN = True), following Rubin tutorial 311.1
    * m_0 = m - R_band * E(B-V), with E(B-V) from the object-table ebv column (SFD98)
    * R_band from rubin_sim DustValues().r_x (CCM89, R_V = 3.1, flat SED, LSST baseline throughputs): R_g, R_r, R_i, R_z = 3.661, 2.701, 2.054, 1.590
    * All color cuts below use the dereddened magnitudes. Objects with a non-finite ebv fail the color cuts.
* Color cuts (dereddened Sersic AB magnitudes)
    * |g-r|, |r-i|, |i-z| < 5 (COLOR_ABS_MAX)
    * Red-sequence veto: reject objects with g-i > 2.0484 (r-z) - 0.10 (RED_SEQ_SLOPE, RED_SEQ_INTERCEPT). These are likely cluster members rather than lensed background sources. The previous line, g-i > 2.276 (r-z) - 0.20, was set on colors without the dust correction.

