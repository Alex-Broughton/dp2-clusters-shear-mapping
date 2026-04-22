# Author: Shenming Fu (shenming.fu.astro@gmail.com)
# Use the original SNR formula of the Schirmer aperture mass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import sys 
from astropy.table import Table
from scipy import stats
from multiprocessing import Pool
import astropy.io.fits as pyfits
from astropy.wcs import WCS
import glob



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



def get_bin_stat_weight(arr, weight):
    '''
        Get stat from 2D binned region (x, y 1D arrays for coordinates)
        
        Note binned_statistic_2d output statistic: x: row, y: col
        So we transpose the 2d result.
        Then the y-axis is flipped.

        Note we use global variables here.
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



def compute_M_ap_at_pixel(ind): 
    '''
        Note we use global variables here.
    '''    
    # ind is for the aperture mass at a specific coordinate position
    # ind might look like tuple or list of row and column indices: (row, col) or [row, col]
    row, col = ind[0], ind[1]

    # Here we calculate the schirmer_weight Q for each pixel relative to the ind coordinate
    weight = Schirmer_weight(((xv - col)**2 + (yv - row)**2)**0.5, Rs)

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



def plot_E_B(E_mat, B_mat, tag, fits=False, zero=False):
    
#    threshold = np.max(
#                        [
#                            np.abs( np.nanmin(E_mat[np.isfinite(E_mat)]) ), 
#                            np.abs( np.nanmax(E_mat[np.isfinite(E_mat)]) ), 
#                            np.abs( np.nanmin(B_mat[np.isfinite(B_mat)]) ),
#                            np.abs( np.nanmax(B_mat[np.isfinite(B_mat)]) ),
#                        ]
#                    )
#
#    
#
#    if zero:
#        threshold_upper = threshold
#        threshold_lower = 0.
#    else:
#        threshold_upper = threshold
#        threshold_lower = -threshold 
    
        

    print("Plotting E B figure...")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    axes0 = axes[0]
    axes1 = axes[1]
    
    threshold_lower = -threshold #-3.
    threshold_upper = +threshold #+3.

    im = axes0.imshow(np.flipud(E_mat), vmin=threshold_lower, vmax=threshold_upper, cmap="viridis", extent=[np.min(x_bin), np.max(x_bin), np.min(y_bin), np.max(y_bin) ])
    axes0.set_xlabel('x [pix]')
    axes0.set_ylabel('y [pix]')
    
    im = axes1.imshow(np.flipud(B_mat), vmin=threshold_lower, vmax=threshold_upper, cmap="viridis", extent=[np.min(x_bin), np.max(x_bin), np.min(y_bin), np.max(y_bin) ])
    axes1.set_xlabel('x [pix]')
    axes1.set_ylabel('y [pix]')
    
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal")
    cbar.ax.set_xlabel("S/N")
    #fig.suptitle("%s_%s | Rs: %.0f pix | Max abs: %.3e"%(tag0, tag, Rs_input, threshold) )
    #fig.suptitle("%s_%s | Rs: %.0f pix"%(tag0, tag, Rs_input) )
    fig.suptitle(r"%s aperture mass S/N map at $R_{\rm{ap}}$ = %.0f pix"%(tag0, Rs_input) )
    
    
    print("Saving figure...")
    plt.savefig("%s_%s_b%d_Rs%d.png"%(tag0, tag, int(bin_size), int(Rs_input)) )
    plt.savefig("%s_%s_b%d_Rs%d.pdf"%(tag0, tag, int(bin_size), int(Rs_input)) )


    # Also, we save an (purely) image for E-mode
#    plt.imsave("%s/%s_%s_b%d_Rs%d_E.png"%(folder_name, tag0, tag, int(bin_size), int(Rs_input)) , np.flipud(E_mat), vmin=threshold_lower, vmax=threshold_upper, cmap="viridis")


    if fits:

    # Here assume we have some coadd "patch" image
    # Note we can not directly use x,y in the catalog because they are not exact

#        patch = "5,5"
#        patch_filename = "%s/rerun/coadd2/deepCoadd/r/0/%s.fits"%(DATA_path, patch)
        tmp = glob.glob("%s/*r44-77.fits"%combine_patch_color_output) 
        patch_filename = tmp[0] 
        
        if os.path.exists(patch_filename)==False: 
            print('%s does NOT exist! Exiting...'%patch_filename)
            sys.exit(1)
        else:
            with pyfits.open(patch_filename) as hdul:
                #w_old = WCS(hdul[1].header)
                w_old = WCS(hdul[0].header)
        
        
        
        #-----------------------
        # Make header for massmap FITS image WCS
        
        w_new = WCS(naxis=2)
        
#        # Consider the 100-pix edge
#        w_old_image_x_real = w_old.wcs.crpix[0] - 100.
#        w_old_image_y_real = w_old.wcs.crpix[1] - 100.
        w_old_image_x_real = w_old.wcs.crpix[0]
        w_old_image_y_real = w_old.wcs.crpix[1]
        #print(w_old_image_x_real, w_old_image_y_real)
        
        # Consider the physical x,y (skymap)
#        w_old_physical_x_real = w_old_image_x_real + int(patch.split(',')[0])*4000.
#        w_old_physical_y_real = w_old_image_y_real + int(patch.split(',')[1])*4000.
        w_old_physical_x_real = w_old_image_x_real + 4*4000.
        w_old_physical_y_real = w_old_image_y_real + 4*4000.
        #print(w_old_physical_x_real, w_old_physical_y_real)
        
        # Convert the catalog x,y to massmap x,y
        # Note FITS image x,y start from 1,1 (lower-left corner 1/2,1/2); catalog starts from 0,0
        w_new_massmap_x = (w_old_physical_x_real - 1. - x_min)/bin_size + 1./2.
        w_new_massmap_y = (w_old_physical_y_real - 1. - y_min)/bin_size + 1./2.
        #print(w_old_physical_x_real - 1. - xmin, w_old_physical_y_real - 1. - ymin)
        #print(w_new_massmap_x, w_new_massmap_y)
        
        # Turn into WCS CRPIX
        w_new.wcs.crpix = [ 
                            w_new_massmap_x,
                            w_new_massmap_y, 
                        ]   
        #print(type(w_old.wcs.crpix))
        print('w_new.wcs.crpix:\n', w_new.wcs.crpix)
            
        w_new.wcs.crval = w_old.wcs.crval
        #print(type(w_old.wcs.crval))
        print('w_new.wcs.crval:\n', w_new.wcs.crval)
            
        w_new.wcs.ctype = w_old.wcs.ctype
        #print(type(w_old.wcs.ctype))
        print('w_new.wcs.ctype:\n', w_new.wcs.ctype)
        
        #w_new.wcs.cd = w_old.wcs.cd*bin_size
        ##print(type(w_old.wcs.cd))
        #print('w_new.wcs.cd:\n', w_new.wcs.cd)
        w_new.wcs.pc = w_old.wcs.pc*bin_size
        #print(type(w_old.wcs.cd))
        print('w_new.wcs.pc:\n', w_new.wcs.pc)
        
        w_new.wcs.mjdobs = w_old.wcs.mjdobs
        #print(type(w_old.wcs.mjdobs))
        print('w_new.wcs.mjdobs:\n', w_new.wcs.mjdobs)
        
        w_new.wcs.dateobs = w_old.wcs.dateobs
        #print(type(w_old.wcs.dateobs))
        print('w_new.wcs.dateobs:\n', w_new.wcs.dateobs)    
        
        w_new.wcs.radesys = w_old.wcs.radesys
        #print(type(w_old.wcs.radesys))
        print('w_new.wcs.radesys:\n', w_new.wcs.radesys)
        
        
        #-----------------------
        massmap_fits_image = E_mat
        header = w_new.to_header()
        hdu = pyfits.PrimaryHDU(massmap_fits_image, header=header)

        massmap_fits_image_filename = "%s/%s_%s_b%d_Rs%d.fits"%(folder_name, tag0, tag, int(bin_size), int(Rs_input))
        hdu.writeto(massmap_fits_image_filename, overwrite=True)
        




def write_peak_to_file(mat_in, tag):
    '''
        Give peak: coord, Rs, [SNR]
    '''
    # Find (max) peak of a matrix mat, then get the index of the (1st near-center) peak
    # Note because some secondary/spurious peak could show up near edge/corner
    # we need to constrain the peak near the center using a radial filter 
    # Note astropy/fits automatically flip the matrix so x->col, y->row (with a small difference at origin 0/1)
    #mat = np.flipud(mat_in)
    mat = mat_in
    #max_val = np.nanmax(mat[np.isfinite(mat)])
    mat_center_y = np.shape(mat)[0]*1./2 - 0.5
    mat_center_x = np.shape(mat)[1]*1./2 - 0.5
    radial_distance = np.sqrt((xv-mat_center_x)**2 + (yv-mat_center_y)**2)
    # Consider a half-degree radius at DECam (FoV 2.2deg)
    select_center = radial_distance <= (0.5*3600/0.263/bin_size)
    #max_val = np.nanmax(mat[np.isfinite(mat[select_center])])
    mat_tmp = mat[select_center]
    max_val = np.nanmax(mat_tmp[np.isfinite(mat_tmp)])
    indices = np.where(mat == max_val)
    print("indices where(mat == max_val): ", indices)
    # Select the 1st one
    index = np.asarray(indices).T.tolist()[0]
    print("index from indices: ", index)    

    # index0: row (up->down) -> true y (because un-flipped)
    # index1: col (left-right) -> true x
    # Transform index to catalog coordinate
    # Assume the peak is at the bin center
    x_out = x_min + index[1]*bin_size + 0.5*bin_size
    y_out = y_min + index[0]*bin_size + 0.5*bin_size
    print("x_min, y_min, x_out, y_out: ", x_min, y_min, x_out, y_out)

    # Write to file
    np.savetxt(
        "%s/%s_%s_b%d_Rs%d_max.txt"%(folder_name, tag0, tag, int(bin_size), int(Rs_input) ), 
        [x_out, y_out, Rs_input, max_val], 
        header="x,y,Rs,SNR"
            )    



#=======================
# Constants

bin_size = 500. # pixels



#=======================
# Script

if len(sys.argv)!=5:
    print("python this.py filename Rs_input[pix] cpu_num #combine_patch_color_output threshold")
    sys.exit(1)

filename = sys.argv[1]
Rs_input = float(sys.argv[2])
cpu_num = int(sys.argv[3])
combine_patch_color_output = None # sys.argv[4]
threshold = float(sys.argv[4])

data = Table.read(filename, format="fits")  # format="ascii.csv")
#print(data)
Rs = Rs_input/bin_size

filename_split = filename.split('/') 
if len(filename_split)==1: 
    folder_name = '.'
else:
    folder_name = filename_split[-2]

tag0 = filename_split[-1].split('_')[0]

print("")



#=======================
# Note: we need to double check the difference between elp and eps
x = data["x"]
y = data["y"]
#e1 = data["e1"]
#e2 = data["e2"]
e1 = data["gauss_g1"]
e2 = data["gauss_g2"]

cov_matrix = (data["gauss_g1_g1_Cov"] + data["gauss_g2_g2_Cov"])

# The factors will be reduced
weight = 1.0 / cov_matrix


#-----------------------
x_min = np.min(x)
x_max = np.max(x)

y_min = np.min(y)
y_max = np.max(y)

#print(x_min, x_max, y_min, y_max)

x_bin = np.arange(x_min, x_max+bin_size, bin_size)
y_bin = np.arange(y_min, y_max+bin_size, bin_size)
#print(x_bin, y_bin)
#print(len(x_bin), len(y_bin) )

ncol = int(np.ceil((x_max - x_min)/bin_size))
nrow = int(np.ceil((y_max - y_min)/bin_size))
print("nrow, ncol: ", nrow, ncol)

# x, y start from 0
# so that x -> col, y -> row
# Note the direction! Different from above
# xv: 
# 0 1 2
# 0 1 2
# 0 1 2
# yv:
# 0 0 0
# 1 1 1
# 2 2 2
xv, yv = np.meshgrid(np.arange(ncol), np.arange(nrow))

# [ (Row, Col) ]
coord_list = list(zip(yv.flatten(), xv.flatten() ) )


#-----------------------
# START calculation

print("Running bin stat...")
e1_binned = get_bin_stat_weight(e1, weight)
e2_binned = get_bin_stat_weight(e2, weight)
#e_sq_binned = e1_binned**2 + e2_binned**2 
e_sq_binned = get_bin_stat_weight(e1**2+e2**2, weight**2)


#-----------------------
print("Computing aperture mass...")


p = Pool(cpu_num)
result = p.map(compute_M_ap_at_pixel, coord_list)

M_ap_E = np.array(result)[:, 0].reshape((nrow, ncol))
M_ap_B = np.array(result)[:, 1].reshape((nrow, ncol))
n_M_ap = np.array(result)[:, 2].reshape((nrow, ncol))

p.close()
p.join()


# END of M_ap calculation!


#-----------------------
#plot_E_B(M_ap_E, M_ap_B, "M_ap")

plot_E_B(M_ap_E/n_M_ap, M_ap_B/n_M_ap, "M_ap_SNR", fits=False)

write_peak_to_file(M_ap_E/n_M_ap, "M_ap_SNR")
