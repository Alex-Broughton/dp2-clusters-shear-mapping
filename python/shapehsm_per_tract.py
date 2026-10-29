import numpy as np
import astropy.units as u
import pandas as pd

from astropy.table import Table, join, vstack
from lsst.daf.butler import Butler
import lsst.geom as geom
from astropy.coordinates import SkyCoord
import healsparse as hsp
import healpy as hp
import scipy.stats as stats
import os, sys, gc, pickle, argparse

from multiprocessing import Pool

pdir = os.getenv("PSCRATCH")

output_dir = f"{pdir}/clusters/dp2_massmaps"

os.environ['DAF_BUTLER_REPOSITORY_INDEX'] = '/global/cfs/cdirs/lsst/production/gen3/shared/data-repos.yaml'
butler = Butler('/repo/DP2', collections='dp2')
skymap = butler.get('skyMap', skymap='lsst_cells_v2')

arcsec = 1 / 3600.
EXTINCTION_R =  {"u": 4.757, "g": 3.661, "r": 2.701, "i": 2.054, "z": 1.590, "y": 1.308}

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

OBJECT_COLUMNS = [
    "objectId",
    "parentObjectId",
    "coord_ra",
    "coord_dec",
    "ebv",  # SFD98 E(B-V) at the object position, for the dust correction
    "detect_isDeblendedModelSource",
    "detect_isIsolated",
    "griz_model_extendedness",
    "i_hsmShapeRegauss_e1",
    "i_hsmShapeRegauss_e2",
    "i_hsmShapeRegauss_sigma",
    "i_ixx",
    "i_iyy",
    "i_ixxPSF",
    "i_iyyPSF",
    "i_inputCount",
    "g_sersicFlux",
    "g_sersicFluxErr",
    "r_sersicFlux",
    "r_sersicFluxErr",
    "i_sersicFlux",
    "i_sersicFluxErr",
    "z_sersicFlux",
    "z_sersicFluxErr",
]

SHEAR_COLUMNS = [
    "objectId",
    "coord_ra",
    "coord_dec",
    "i_hsmShapeRegauss_e1",
    "i_hsmShapeRegauss_e2",
    "i_hsmShapeRegauss_sigma",
]


all_columns = FLAG_COLUMNS + OBJECT_COLUMNS + SHEAR_COLUMNS


def shapeHSM_redseq_cut(catalog):
    RED_SEQ_SLOPE = 2.0484  # Red-sequence line: g-i = slope * (r-z) + intercept
    RED_SEQ_INTERCEPT = -0.10
    gi = catalog['g_sersicMag_Dered'] - catalog['i_sersicMag_Dered']
    rz = catalog['r_sersicMag_Dered'] - catalog['z_sersicMag_Dered']

    rs_cut = ~(gi > RED_SEQ_SLOPE * rz + RED_SEQ_INTERCEPT)
    rs_cut &= np.isfinite(gi) * np.isfinite(rz)
    return rs_cut


def shapeHSM_quality_cuts(catalog, wrong_primary=True):
    good_objs = np.ones(len(catalog)).astype(bool)

    for flag in FLAG_COLUMNS:
        good_objs &= (catalog[flag] == False)
    

    if wrong_primary:
        good_objs &= catalog['detect_isDeblendedModelSource']
    else:
        good_objs &= np.logical_or(catalog['detect_isDeblendedModelSource'], catalog['detect_isIsolated'])
    

    esq = catalog['i_hsmShapeRegauss_e1']**2 + catalog['i_hsmShapeRegauss_e2']**2
    good_objs &= esq <= 4.0


    resolution_factor = 1 - ((catalog['i_ixxPSF'] + catalog['i_iyyPSF'])/
                             (catalog['i_ixx'] + catalog['i_iyy']))
    catalog['resolution_factor'] = resolution_factor
    good_objs &= resolution_factor > 0.125

    for b in 'griz':
        good_objs &= (catalog[f'{b}_sersicFlux']/catalog[f'{b}_sersicFluxErr']) > 5.0
    good_objs &= catalog['griz_model_extendedness'] > 0.2
    # 5 objects difference between this set of cuts and the original notebook
    # not sure why that is but 5 is pretty small...
    
    good_objs &= np.abs(catalog['g_sersicMag_Dered'] - catalog['r_sersicMag_Dered']) < 5
    good_objs &= np.abs(catalog['r_sersicMag_Dered'] - catalog['i_sersicMag_Dered']) < 5
    good_objs &= np.abs(catalog['i_sersicMag_Dered'] - catalog['z_sersicMag_Dered']) < 5


    return good_objs

def dust_correction(catalog):
    for b in 'griz':
        rel_flux = catalog[f'{b}_sersicFlux']
        rel_mag = u.nJy.to(u.ABmag, rel_flux)
        catalog[f'{b}_sersicMag_Dered'] = rel_mag - catalog['ebv'] * EXTINCTION_R[b]
    return None

def shear_calibration(catalog):
    R, m_mean = 0.8511839657185559, -0.13584282915829576

    e1 = catalog['i_hsmShapeRegauss_e1']
    e2 = catalog['i_hsmShapeRegauss_e2']

    g1 = e1/(2*R*(1+m_mean))
    g2 = e2/(2*R*(1+m_mean))

    msmt_variance = np.nanmean(e1**2 + e2**2) / 2.0 - np.nanmedian(catalog['i_hsmShapeRegauss_sigma']**2)

    weight = 1/(catalog['i_hsmShapeRegauss_sigma']**2 + msmt_variance)

    comp_g = g1 + 1j*g2
    return comp_g, weight


def get_inner_pixels(tract_ndx):
    inner_bbox = skymap[tract_ndx].getInnerSkyRegion()
    ra_A = inner_bbox.getLon().getA().asDegrees()
    ra_B = inner_bbox.getLon().getB().asDegrees()
    dec_A = inner_bbox.getLat().getA().asDegrees()
    dec_B = inner_bbox.getLat().getB().asDegrees()
    kx = hsp.Box(ra1=ra_A, ra2=ra_B, dec1=dec_A, dec2=dec_B, value=1.0)
    inner_pixs = kx.get_pixels(nside=4096)
    return inner_pixs

def create_hsp(tract_ndx, comp_g, weights, tract_pixs, inner_pixs):
    N_gals = hsp.HealSparseMap.make_empty(32, 4096, np.int16)
    w = hsp.HealSparseMap.make_empty(32, 4096, np.float64)
    w2 = hsp.HealSparseMap.make_empty(32, 4096, np.float64)
    g1 = hsp.HealSparseMap.make_empty(32, 4096, np.float64)
    g2 = hsp.HealSparseMap.make_empty(32, 4096, np.float64)

    for ip in inner_pixs:
        tp_filt = tract_pixs==ip
        if np.sum(tp_filt)==0:
            continue
        N_gals[ip] = np.sum(tp_filt)
        w[ip] = np.sum(weights[tp_filt])
        w2[ip] = np.sum(weights[tp_filt]**2)
        g1[ip] = np.sum(comp_g.real[tp_filt] * weights[tp_filt])
        g2[ip] = np.sum(comp_g.imag[tp_filt] * weights[tp_filt])

    return N_gals, w, w2, g1, g2


def process_tract(tract_ndx):
    tract_table = butler.get("object", dataId={"tract": tract_ndx, 
                                               "skymap": 'lsst_cells_v2'},
                         parameters={"columns": all_columns})
    _ = dust_correction(tract_table)
    all_cuts = shapeHSM_quality_cuts(tract_table, wrong_primary=False)
    rs_cut = shapeHSM_redseq_cut(tract_table)

    clean_hsm = tract_table[all_cuts * rs_cut]
    comp_g, weights = shear_calibration(clean_hsm)
    inner_pixs = get_inner_pixels(tract_ndx)
    tract_pixs = hp.ang2pix(4096, clean_hsm['coord_ra'], clean_hsm['coord_dec'], nest=True, lonlat=True)

    N_gals, w, w2, g1, g2 = create_hsp(tract_ndx, comp_g, weights, tract_pixs, inner_pixs)

    N_gals.write(f'{output_dir}/shapeHSM_v0_{tract_ndx}_ngals.hs', clobber=True)
    w.write(f'{output_dir}/shapeHSM_v0_{tract_ndx}_w.hs', clobber=True)
    w2.write(f'{output_dir}/shapeHSM_v0_{tract_ndx}_w2.hs', clobber=True)
    g1.write(f'{output_dir}/shapeHSM_v0_{tract_ndx}_g1.hs', clobber=True)
    g2.write(f'{output_dir}/shapeHSM_v0_{tract_ndx}_g2.hs', clobber=True)

    return None


if __name__ == "__main__":

    tracts = np.array([2535, 2536, 2702, 2703, 2704, 2705, 2875, 2876, 2877, 2878, 3052, 3053, 3054, 3055])

    for tct in tracts:
        process_tracts(tct)

    # TO-DO: Make this actually parallelized with Butler calls...
    # with Pool(8) as p:
    #    p.map(process_tract, tracts) 
