DP2 Shear Starting CutsA set of beginning cuts to get started on looking through the DP2 Metadetect shear catalogs.

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

