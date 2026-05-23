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

from catalog_dedupe import format_catalog_summary, load_shear_arrays, parse_dedupe_flags



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



def _sym_color_limit(e_mat, b_mat, percentile):
    """Symmetric |value| limit for diverging E/B maps."""
    z = np.concatenate([np.abs(e_mat.ravel()), np.abs(b_mat.ravel())])
    z = z[np.isfinite(z)]
    if z.size == 0:
        return float(threshold)
    t = float(np.nanpercentile(z, percentile))
    return t if t > 0 else float(threshold)


def _map_extent():
    return [np.min(x_bin), np.max(x_bin), np.min(y_bin), np.max(y_bin)]


def compute_bin_diagnostic_maps(x, y, e1, e2, x_bin, y_bin):
    """Per-bin galaxy count and unweighted std(|e|) for shape-noise map."""
    e_amp = np.sqrt(e1**2 + e2**2)
    n_gal, _, _, _ = stats.binned_statistic_2d(
        x,
        y,
        np.ones(x.size, dtype=np.float64),
        statistic="count",
        bins=[x_bin, y_bin],
    )
    shape_std, _, _, _ = stats.binned_statistic_2d(
        x,
        y,
        e_amp,
        statistic="std",
        bins=[x_bin, y_bin],
    )
    return n_gal.T, shape_std.T


def plot_scalar_map(mat, tag, cbar_label, cmap="viridis", out_prefix=None):
    """Single-panel imshow on the binned grid (same orientation as plot_E_B)."""
    print(f"Plotting diagnostic map ({tag})...")
    extent = _map_extent()
    plot_mat = np.flipud(mat)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        plot_mat,
        cmap=cmap,
        extent=extent,
        aspect="auto",
    )
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    fig.suptitle(
        rf"{tag0} | {tag} | bin = {bin_size:.0f} pix, $R_{{\rm ap}}$ = {Rs_input:.0f} pix",
        y=1.02,
    )
    out = _figure_stem(tag, out_prefix)
    print("Saving figure...", out)
    plt.savefig(out + ".png", bbox_inches="tight")
    plt.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_e_b_correlation(e_mat, b_mat, tag, sym_percentile=95.0, out_prefix=None):
    """Cell-by-cell E vs B (same quantity as tag: M_ap or S/N)."""
    print(f"Plotting E–B correlation ({tag})...")
    e_flat = e_mat.ravel()
    b_flat = b_mat.ravel()
    ok = np.isfinite(e_flat) & np.isfinite(b_flat)
    e_use = e_flat[ok]
    b_use = b_flat[ok]
    if e_use.size < 2:
        print("  skip: fewer than 2 finite cells for correlation plot")
        return

    pearson = float(np.corrcoef(e_use, b_use)[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].scatter(e_use, b_use, s=6, alpha=0.35, c="0.2", edgecolors="none")
    axes[0].axhline(0.0, color="0.75", lw=0.8)
    axes[0].axvline(0.0, color="0.75", lw=0.8)
    axes[0].set_xlabel(r"E-mode " + ("S/N" if "SNR" in tag else r"$M_{\rm ap}$"))
    axes[0].set_ylabel(r"B-mode " + ("S/N" if "SNR" in tag else r"$M_{\rm ap}$"))
    axes[0].set_title(rf"Cell values ($N$ = {e_use.size:,}, $r$ = {pearson:.3f})")

    hb = axes[1].hexbin(
        e_use,
        b_use,
        gridsize=40,
        cmap="magma",
        mincnt=1,
        linewidths=0,
    )
    axes[1].axhline(0.0, color="w", lw=0.8, alpha=0.7)
    axes[1].axvline(0.0, color="w", lw=0.8, alpha=0.7)
    axes[1].set_xlabel(r"E-mode " + ("S/N" if "SNR" in tag else r"$M_{\rm ap}$"))
    axes[1].set_ylabel(r"B-mode " + ("S/N" if "SNR" in tag else r"$M_{\rm ap}$"))
    axes[1].set_title("Density (hexbin)")
    fig.colorbar(hb, ax=axes[1], fraction=0.046, pad=0.04, label="counts")

    if sym_percentile is not None:
        lim = _sym_color_limit(e_mat, b_mat, sym_percentile)
        for ax in axes:
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        rf"{tag0} | {tag} | $R_{{\rm ap}}$ = {Rs_input:.0f} pix, bin = {bin_size:.0f} pix",
        y=1.03,
    )
    out = _figure_stem(tag, out_prefix)
    print("Saving figure...", out, f"(Pearson r = {pearson:.4f})")
    plt.savefig(out + ".png", bbox_inches="tight")
    plt.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_E_B(E_mat, B_mat, tag, fits=False, zero=False, sym_percentile=95.0, out_prefix=None):
    print(f"Plotting E/B figure ({tag})...")
    extent = _map_extent()

    if sym_percentile is not None:
        thr = _sym_color_limit(E_mat, B_mat, sym_percentile)
    else:
        thr = float(threshold)
    threshold_lower = 0.0 if zero else -thr
    threshold_upper = thr

    is_snr = "SNR" in tag
    cmap = "RdBu_r"
    cbar_label = "S/N" if is_snr else r"$M_{\rm ap}$"
    mode_title = "S/N" if is_snr else r"$M_{\rm ap}$"

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    im0 = axes[0].imshow(
        np.flipud(E_mat),
        vmin=threshold_lower,
        vmax=threshold_upper,
        cmap=cmap,
        extent=extent,
        aspect="auto",
    )
    axes[0].set_xlabel("x [pix]")
    axes[0].set_ylabel("y [pix]")
    axes[0].set_title(r"E-mode " + mode_title)

    im1 = axes[1].imshow(
        np.flipud(B_mat),
        vmin=threshold_lower,
        vmax=threshold_upper,
        cmap=cmap,
        extent=extent,
        aspect="auto",
    )
    axes[1].set_xlabel("x [pix]")
    axes[1].set_ylabel("y [pix]")
    axes[1].set_title(r"B-mode " + mode_title)

    fig.colorbar(im1, ax=axes, orientation="horizontal", label=cbar_label)
    fig.suptitle(
        rf"{tag0} | {tag} | $R_{{\rm ap}}$ = {Rs_input:.0f} pix, bin = {bin_size:.0f} pix "
        rf"(|val|$_{{{sym_percentile:.0f}\%}}$ = {thr:.3g})",
        y=1.02,
    )

    out = _figure_stem(tag, out_prefix)
    print("Saving figure...", out)
    plt.savefig(out + ".png", bbox_inches="tight")
    plt.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)


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
# Constants (native tract pixels @ PIX_SCALE_ARCSEC per pixel)
#
# Cluster-scale presets (Rs_input, bin_size) — Rs_bins = Rs_input / bin_size:
#   Tight (~2'):   600,  60  -> Rs_bins=10,  grid ~450 cells across tract 9813
#   Default (~3'): 900,  90  -> Rs_bins=10,  grid ~300
#   Wide (~5'):   1500, 150  -> Rs_bins=10,  grid ~180
#
# Avoid Rs_bins >> ~15 on a grid of only ~50 cells (old 10000/500 on ~54 bins
# smoothed almost the entire tract).

PIX_SCALE_ARCSEC = 0.2
DEFAULT_BIN_SIZE_PIX = 90.0
DEFAULT_RS_INPUT_PIX = 900.0  # ~3 arcmin R_ap at 0.2"/pix
ENABLE_DEDUPE = True  # set False or pass --no-dedupe on the command line
WEIGHT_FLOOR = 2.0e5  # minimum weight (DP2-tuned) for sources with valid covariance

# Plot context (set by main() or schirmer_snr_sweep via set_plot_context)
tag0 = "map"
folder_name = "."
bin_size = DEFAULT_BIN_SIZE_PIX
Rs_input = DEFAULT_RS_INPUT_PIX
threshold = 1.0
x_bin = np.array([0.0, 1.0])
y_bin = np.array([0.0, 1.0])


def set_plot_context(
    tag0_in,
    rs_input_in,
    bin_size_in,
    x_bin_in,
    y_bin_in,
    threshold_in=1.0,
    folder_name_in=".",
):
    """Set module globals used by plot_E_B and related helpers."""
    global tag0, Rs_input, bin_size, x_bin, y_bin, threshold, folder_name
    tag0 = tag0_in
    Rs_input = float(rs_input_in)
    bin_size = float(bin_size_in)
    x_bin = x_bin_in
    y_bin = y_bin_in
    threshold = float(threshold_in)
    folder_name = folder_name_in


def _figure_stem(tag, out_prefix=None):
    if out_prefix is not None:
        return str(out_prefix)
    return "%s_%s_b%d_Rs%d" % (tag0, tag, int(bin_size), int(Rs_input))


def main():
    global x, y, e1, e2, weight, x_bin, y_bin, xv, yv, Rs
    global x_min, y_min, tag0, folder_name, bin_size, Rs_input, threshold

    dedupe, dedupe_sep_pix, argv_rest = parse_dedupe_flags(sys.argv[1:])
    if not dedupe and ENABLE_DEDUPE:
        dedupe = True

    if len(argv_rest) not in (5, 6):
        print(
            "python schirmer_snr_weight.py [--dedupe] [--dedupe-sep-pix SEP] "
            "filename Rs_input[pix] cpu_num threshold [bin_size_pix]\n"
            f"  defaults: Rs_input={DEFAULT_RS_INPUT_PIX:.0f}, bin_size={DEFAULT_BIN_SIZE_PIX:.0f} "
            f"(@ {PIX_SCALE_ARCSEC} arcsec/pix), dedupe={'on' if ENABLE_DEDUPE else 'off'}\n"
            "  --dedupe: merge nearby (x,y) within sep [tract pix]; default sep=1"
        )
        sys.exit(1)

    filename = argv_rest[0]
    Rs_input = float(argv_rest[1])
    cpu_num = int(argv_rest[2])
    combine_patch_color_output = None
    threshold = float(argv_rest[3])
    bin_size = float(argv_rest[4]) if len(argv_rest) == 6 else DEFAULT_BIN_SIZE_PIX

    Rs = Rs_input / bin_size

    print(
        f"Schirmer R_ap = {Rs_input:.0f} pix ({Rs_input * PIX_SCALE_ARCSEC / 60:.2f} arcmin), "
        f"bin = {bin_size:.0f} pix ({bin_size * PIX_SCALE_ARCSEC:.1f} arcsec), "
        f"Rs/bin = {Rs:.1f} cells"
    )

    filename_split = filename.split("/")
    if len(filename_split) == 1:
        folder_name = "."
    else:
        folder_name = filename_split[-2]

    tag0 = filename_split[-1].split("_")[0]

    print("")

    x, y, e1, e2, weight, cat_info = load_shear_arrays(
        filename,
        weight_floor=WEIGHT_FLOOR,
        dedupe=dedupe,
        dedupe_sep_pix=dedupe_sep_pix,
    )
    print(f"catalog: {format_catalog_summary(cat_info)}")
    if cat_info.get("dedupe"):
        print(
            f"  position dedupe: sep={cat_info.get('dedupe_sep_pix', 1.0):.3g} tract pix "
            f"({cat_info.get('dedupe_sep_pix', 1.0) * PIX_SCALE_ARCSEC:.3g} arcsec)"
        )

    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    x_bin = np.arange(x_min, x_max + bin_size, bin_size)
    y_bin = np.arange(y_min, y_max + bin_size, bin_size)
    set_plot_context(tag0, Rs_input, bin_size, x_bin, y_bin, threshold, folder_name)

    ncol = int(np.ceil((x_max - x_min) / bin_size))
    nrow = int(np.ceil((y_max - y_min) / bin_size))
    print("nrow, ncol: ", nrow, ncol)

    xv, yv = np.meshgrid(np.arange(ncol), np.arange(nrow))
    coord_list = list(zip(yv.flatten(), xv.flatten()))

    print("Running bin stat...")
    e1_binned = get_bin_stat_weight(e1, weight)
    e2_binned = get_bin_stat_weight(e2, weight)
    e_sq_binned = get_bin_stat_weight(e1**2 + e2**2, weight**2)

    n_gal_map, shape_noise_std_map = compute_bin_diagnostic_maps(x, y, e1, e2, x_bin, y_bin)
    plot_scalar_map(n_gal_map, "n_gal", "galaxy count", cmap="viridis")
    plot_scalar_map(shape_noise_std_map, "shape_noise_std", r"std($|e|$)", cmap="plasma")

    print("Computing aperture mass...")

    p = Pool(cpu_num)
    result = p.map(compute_M_ap_at_pixel, coord_list)

    M_ap_E = np.array(result)[:, 0].reshape((nrow, ncol))
    M_ap_B = np.array(result)[:, 1].reshape((nrow, ncol))
    n_M_ap = np.array(result)[:, 2].reshape((nrow, ncol))

    p.close()
    p.join()

    with np.errstate(divide="ignore", invalid="ignore"):
        snr_e = M_ap_E / n_M_ap
        snr_b = M_ap_B / n_M_ap

    plot_E_B(M_ap_E, M_ap_B, "M_ap", sym_percentile=95.0)
    plot_E_B(snr_e, snr_b, "M_ap_SNR", sym_percentile=95.0, fits=False)

    plot_e_b_correlation(M_ap_E, M_ap_B, "M_ap_E_vs_B", sym_percentile=95.0)
    plot_e_b_correlation(snr_e, snr_b, "M_ap_SNR_E_vs_B", sym_percentile=95.0)

    write_peak_to_file(snr_e, "M_ap_SNR")


if __name__ == "__main__":
    main()
