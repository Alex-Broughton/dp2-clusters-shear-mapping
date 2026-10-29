# Author: Prakruth Adari + Shenming Fu (shenming.fu.astro@gmail.com)
# Adapted from Shenming's original code (schirmer_snr_weight_shenming.py
# Use the original SNR formula of the Schirmer aperture mass


import numpy as np
from astropy.coordinates import SkyCoord
import sys 
import os
from astropy.table import Table
from scipy import stats
from multiprocessing import Pool
import astropy.io.fits as pyfits
from astropy.wcs import WCS
import glob
import healsparse as hsp
import healpy as hp


#=======================
# Constants
bin_size = 500. # pixels

#=======================
# Functions

def Schirmer_weight(r, Rs):
    x = r/Rs

    a = 6.
    b = 150.
    c = 47.
    d = 50.
    xc = 0.15

    Q =  1./(1. + np.exp(a-b*x) + np.exp(d*x-c) )
    Q *= np.tanh( x/xc ) / ( x/xc )

    return Q


def get_bin_stat_weight(arr, weight, x, y):
    '''
        Get stat from 2D binned region (x, y 1D arrays for coordinates)
        
        Note binned_statistic_2d output statistic: x: row, y: col
        So we transpose the 2d result.
        Then the y-axis is flipped.
    '''
    
    statistic_0, x_edge, y_edge, binnumber = stats.binned_statistic_2d(
                                    x, y,
                                    arr*weight,
                                    statistic="sum",
                                    bins=[x_bin, y_bin],
                                )

    statistic_1, x_edge, y_edge, binnumber = stats.binned_statistic_2d(
                                    x, y,
                                    weight,
                                    statistic="sum",
                                    bins=[x_bin, y_bin],
                                )

    return (statistic_0/statistic_1).T

def compute_M_ap_at_pixel(ind, xv, yv, e1_binned,
                          e2_binned, e_sq_binned, schirmer_rs = 1.5): 
    # ind is for the aperture mass at a specific coordinate position
    # ind might look like tuple or list of row and column indices: (row, col) or [row, col]
    row, col = ind[0], ind[1]

    # Here we calculate the schirmer_weight Q for each pixel relative to the ind coordinate
    weight = Schirmer_weight(((xv - col)**2 + (yv - row)**2)**0.5, schirmer_rs)

    # Note the et or ex here is for each pixel relative to the ind coordinate
    # e1, e2 are the values at each pixel and their shapes should match xv, yv
    dx, dy = xv - col, yv - row
    #d = np.sqrt(dx*dx+dy*dy)
    # Note y-axis is flipped (top -> bottom)!
    # Note the binned e1, e2 are also y-axis-flipped.
    # Note the e1, e2 from the catalog have their origin's direction
    angle = np.arctan2(dy, dx)
    et = - e1_binned * np.cos( 2. * angle ) - e2_binned * np.sin( 2. * angle )
    ex = + e1_binned * np.sin( 2. * angle ) - e2_binned * np.cos( 2. * angle )

    # Get aperture mass
    M_ap_E_tmp = np.nansum(weight * et )
    M_ap_B_tmp = np.nansum(weight * ex )

    tmp = weight**2 * e_sq_binned
#    n_gal = np.sum(~np.isnan(tmp))
    n_M_ap_tmp = np.sqrt(np.nansum(tmp))/np.sqrt(2)
    
    return M_ap_E_tmp, M_ap_B_tmp, n_M_ap_tmp

def transform_comp_e(coord, coords, complex_binned_e):
    ang_dist = coords.separation(coord).deg
    phi = coords.position_angle(coord).rad

    transformed_e = complex_binned_e * np.exp(-2j * phi)
    return transformed_e, ang_dist

def compute_M_ap_at_coord(coord, coords, complex_binned_e,
                          e_sq_binned, schirmer_rs):

    transformed_e, ang_dist = transform_comp_e(coord, coords, complex_binned_e)
    weight = Schirmer_weight(ang_dist, schirmer_rs)    
    # TO-DO Update this to use the average value over a cell width instead of the value at the center/edge of a bin
    et = transformed_e.real
    ex = transformed_e.imag
    # Get aperture mass
    M_ap_E_tmp = np.nansum(weight * et )
    M_ap_B_tmp = np.nansum(weight * ex )

    tmp = weight**2 * e_sq_binned
#    n_gal = np.sum(~np.isnan(tmp))
    n_M_ap_tmp = np.sqrt(np.nansum(tmp))/np.sqrt(2)
    
    return M_ap_E_tmp, M_ap_B_tmp, n_M_ap_tmp

def prep_hsp_maps(g1, g2, weights, weights_sqr):
    valid_pixs = g1.valid_pixels
    comp_g_hp = (g1[valid_pixs] + 1j*g2[valid_pixs]) / (weights[valid_pixs])
    g_sqr_hp = (g1[valid_pixs]**2 + g2[valid_pixs]**2) / (weights_sqr[valid_pixs])
    all_hp_coords = hp.pix2ang(4096, valid_pixs, nest=True, lonlat=True)
    full_hp_coords = SkyCoord(all_hp_coords[0], all_hp_coords[1], unit='degree')

    return full_hp_coords, comp_g_hp, g_sqr_hp

def process_maps(coords, comp_g, g_sqr, pixs):
    e_mode = hsp.HealSparseMap.make_empty(32, 4096, dtype=float)
    b_mode = hsp.HealSparseMap.make_empty(32, 4096, dtype=float)
    v_mode = hsp.HealSparseMap.make_empty(32, 4096, dtype=float)

    for i,vpix in enumerate(pixs):
        E, B, V = compute_M_ap_at_coord(coords[i], coords,
                                        comp_g, g_sqr, 0.25)
        e_mode[vpix] = E
        b_mode[vpix] = B
        v_mode[vpix] = V

    return e_mode, b_mode, v_mode

def parallel_process_maps(vpix, coords, comp_g, g_sqr):

    E, B, V = compute_M_ap_at_coord(coords[vpix], coords,
                                    comp_g, g_sqr, 0.25)

    return E, B, V

