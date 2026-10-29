import numpy as np
import healpy as hp
import healsparse as hsp
import astropy.units as u
from astropy.coordinates import SkyCoord, angular_separation, position_angle
from astropy.io import fits
from astropy.table import Table, vstack
import scipy.stats as stats
import importlib
import multiprocessing
from functools import partial

import os, gc, sys, argparse

import schirmer_map_fns as smf

rng = np.random.default_rng()

output_dir = '/global/cfs/cdirs/lsst/groups/CL/DP2/dp2_massmaps/hsp_maps'

def combine_tracts(tracts, output_dir=output_dir):
    
    g1s = []
    g2s = []
    ws = []
    w2s = []
    Ns = []
    
    for tract in tracts:
        g1s.append(hsp.HealSparseMap.read(f'{output_dir}/shapeHSM/shapeHSM_v0_{tract}_g1.hs'))
        g2s.append(hsp.HealSparseMap.read(f'{output_dir}/shapeHSM/shapeHSM_v0_{tract}_g2.hs'))
        ws.append(hsp.HealSparseMap.read(f'{output_dir}/shapeHSM/shapeHSM_v0_{tract}_w.hs'))
        w2s.append(hsp.HealSparseMap.read(f'{output_dir}/shapeHSM/shapeHSM_v0_{tract}_w2.hs'))
        Ns.append(hsp.HealSparseMap.read(f'{output_dir}/shapeHSM/shapeHSM_v0_{tract}_ngals.hs'))

    deepfield_g1 = hsp.operations.sum_union(g1s)
    deepfield_g2 = hsp.operations.sum_union(g2s)
    deepfield_w = hsp.operations.sum_union(ws)
    deepfield_w2 = hsp.operations.sum_union(w2s)
    deepfield_N = hsp.operations.sum_union(Ns)

    return deepfield_g1, deepfield_g2, deepfield_w, deepfield_w2, deepfield_N



if __name__ == "__main__":

    deep_field_tracts = {
       'ELAIS_S1': [2535, 2536, 2702, 2703, 2704, 2705, 2875, 2876, 2877, 2878, 3052, 3053, 3054, 3055],
       'ECDFS': [4636, 4637, 4638, 4847, 4848, 4849, 4850, 5062, 5063, 5064, 5279, 5280, 5281],
       'COSMOS': [ 9569,  9570,  9571,  9572,  9812,  9813,  9814, 10054, 10055, 10056],
       'EDFS': [2078, 2079, 2080, 2233, 2234, 2235, 2236, 2237, 2393, 2394, 2395, 2396, 2397, 2558, 2559, 2560, 2561, 2562, 2729, 2730]
    }

    pool = multiprocessing.Pool(processes=64)

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--field", help="Field name: ['ELAIS_S1', 'ECDFS', 'COSMOS', 'EDFS', 'WIDE']")
    args = parser.parse_args()
    field_name = args.field

    if field_name == 'WIDE':
        tracts = np.load('./data/tract_list.npy')
    else:
        tracts = deep_field_tracts[field_name]

    g1, g2, w, w2, N = combine_tracts(tracts)

    coords, comp_g, g_sqr = smf.prep_hsp_maps(g1, g2, w, w2)
    full_pixs = g1.valid_pixels

    partial_process = partial(smf.parallel_process_maps, coords=coords, comp_g=comp_g, g_sqr=g_sqr)
    # partial_process = lambda x : smf.process_maps(coords=coords, comp_g=comp_g, g_sqr=g_sqr, pixs=x)

    pool_pixs = np.arange(len(full_pixs))

    result_list = pool.map(partial_process, pool_pixs)
    
    all_es = np.array([rl[0] for rl in result_list])
    all_bs = np.array([rl[1] for rl in result_list])
    all_vs = np.array([rl[2] for rl in result_list])
    
    final_e = hsp.HealSparseMap.make_empty(32, 4096, float)
    final_b = hsp.HealSparseMap.make_empty(32, 4096, float)
    final_v = hsp.HealSparseMap.make_empty(32, 4096, float)

    final_e[full_pixs] = all_es
    final_b[full_pixs] = all_bs
    final_v[full_pixs] = all_vs

    # final_e = hsp.operations.sum_union(all_es)
    # final_b = hsp.operations.sum_union(all_bs)
    # final_v = hsp.operations.sum_union(all_vs)

    final_e.write(f'{output_dir}/{field_name}_emap.hs', clobber=True)
    final_b.write(f'{output_dir}/{field_name}_bmap.hs', clobber=True)
    final_v.write(f'{output_dir}/{field_name}_vmap.hs', clobber=True)
    
