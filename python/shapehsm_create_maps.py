import numpy as np
import healpy as hp
import healsparse as hsp
import healpy as hp
import astropy.units as u
from astropy.coordinates import SkyCoord, angular_separation, position_angle
from astropy.io import fits
from astropy.table import Table, vstack
import scipy.stats as stats
import importlib
from functools import partial
from mpi4py import MPI
from scipy.spatial import KDTree
import tqdm
import os, gc, sys, argparse

import schirmer_map_fns as smf

rng = np.random.default_rng()

output_dir = '/global/cfs/cdirs/lsst/groups/CL/DP2/dp2_massmaps/hsp_maps'
verbose = True

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


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


    if rank==0:
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--field", help="Field name: ['ELAIS_S1', 'ECDFS', 'COSMOS', 'EDFS', 'WIDE']")
        args = parser.parse_args()
        field_name = args.field

        if field_name == 'WIDE':
            if os.path.isfile(f'{output_dir}/widefield_g1.hs'):
                if verbose: print("Loading files")
                g1 = hsp.HealSparseMap.read(f'{output_dir}/widefield_g1.hs')
                g2 = hsp.HealSparseMap.read(f'{output_dir}/widefield_g2.hs')
                w = hsp.HealSparseMap.read(f'{output_dir}/widefield_w.hs')
                w2 = hsp.HealSparseMap.read(f'{output_dir}/widefield_w2.hs')
                N = hsp.HealSparseMap.read(f'{output_dir}/widefield_N.hs')
            else:
                tracts = np.load('./data/tract_list.npy')
                g1, g2, w, w2, N = combine_tracts(tracts)
        else:
            if os.path.isfile(f'{output_dir}/{field_name}_g1.hs'):
                if verbose: print("Loading files")
                g1 = hsp.HealSparseMap.read(f'{output_dir}/{field_name}_g1.hs')
                g2 = hsp.HealSparseMap.read(f'{output_dir}/{field_name}_g2.hs')
                w = hsp.HealSparseMap.read(f'{output_dir}/{field_name}_w.hs')
                w2 = hsp.HealSparseMap.read(f'{output_dir}/{field_name}_w2.hs')
                N = hsp.HealSparseMap.read(f'{output_dir}/{field_name}_N.hs')
            else:
                tracts = deep_field_tracts[field_name]
                g1, g2, w, w2, N = combine_tracts(tracts)
        coords, comp_g, g_sqr = smf.prep_hsp_maps(g1, g2, w, w2)
        full_pixs = g1.valid_pixels
        # full_pixs = g1.valid_pixels[:1000]
        Nfp = len(full_pixs)

        np_coords = np.vstack((coords.ra.deg, coords.dec.deg)).T
        coord_tree = KDTree(np_coords)

        split_pixs = np.array_split(full_pixs, size)
        split_ndxs = np.array_split(np.arange(Nfp), size)
        comm.Barrier()
    else:
        split_ndxs = None
        coords = None
        coord_tree = None
        Nfp = None
        comm.Barrier()

    Nfp = comm.bcast(Nfp, root=0)
    coord_tree = comm.bcast(coord_tree, root=0)
    split_ndxs = comm.scatter(split_ndxs, root=0)
    coords = comm.bcast(coords, root=0)

    print(f"At {rank} we have {len(split_ndxs)=} with {Nfp=}")

    if rank != 0:
        comp_g = np.empty(Nfp, dtype=np.complex64)
        g_sqr = np.empty(Nfp, dtype=np.float64)

    comm.Bcast(comp_g, root=0)
    comm.Bcast(g_sqr, root=0)

    all_sigs = np.zeros((3, Nfp))
    for i, sndx in enumerate(split_ndxs):
        if i%1000==0:
            print(f"Finished with {i} at {rank} out of {len(split_ndxs)}")
        query_coord = np.array([coords[sndx].ra.deg, coords[sndx].dec.deg])
        tree_filt = coord_tree.query_ball_point(query_coord, r=0.33)
        E, B, V = smf.compute_M_ap_at_coord(coords[sndx], coords[tree_filt],
                                        comp_g[tree_filt], g_sqr[tree_filt], 0.25)
        all_sigs[0, sndx] = E
        all_sigs[1, sndx] = B
        all_sigs[2, sndx] = V


    recv_sigs = None
    if rank == 0:
        recv_sigs = np.empty([size, 3, Nfp], dtype=np.float64)

    comm.Gather(all_sigs, recv_sigs, root=0)

    # partial_process = partial(smf.parallel_process_maps, coords=coords, comp_g=comp_g, g_sqr=g_sqr, ktree=coord_tree)
    # partial_process = partial(smf.parallel_process_maps, coords=coords, comp_g=comp_g, g_sqr=g_sqr)
    # partial_process = lambda x : smf.process_maps(coords=coords, comp_g=comp_g, g_sqr=g_sqr, pixs=x)

    # if verbose: print("Pooling...")
    # pool = multiprocessing.Pool(processes=16)
    # result_list = list(tqdm.tqdm(pool.imap(partial_process, pool_pixs), total=len(pool_pixs)))
    # # result_list = pool.map(partial_process, pool_pixs)
    
    if rank==0:
        collapse_sig = np.sum(recv_sigs, axis=0)
        all_es = collapse_sig[0,:]
        all_bs = collapse_sig[1,:]
        all_vs = collapse_sig[2,:]
        
        final_e = hsp.HealSparseMap.make_empty(32, 4096, float)
        final_b = hsp.HealSparseMap.make_empty(32, 4096, float)
        final_v = hsp.HealSparseMap.make_empty(32, 4096, float)

        final_e[full_pixs] = all_es
        final_b[full_pixs] = all_bs
        final_v[full_pixs] = all_vs

        m_emode = hsp.HealSparseMap.make_empty(32, 4096, dtype=float)
        m_bmode = hsp.HealSparseMap.make_empty(32, 4096, dtype=float)

        m_emode[full_pixs] = all_es / np.sqrt(all_vs)
        m_bmode[full_pixs] = all_bs / np.sqrt(all_vs)

        final_e.write(f'{output_dir}/{field_name}_emap.hs', clobber=True)
        m_emode.write(f'{output_dir}/{field_name}_eSNRmap.hs', clobber=True)
        final_b.write(f'{output_dir}/{field_name}_bmap.hs', clobber=True)
        m_bmode.write(f'{output_dir}/{field_name}_bSNRmap.hs', clobber=True)
        final_v.write(f'{output_dir}/{field_name}_vmap.hs', clobber=True)
        
