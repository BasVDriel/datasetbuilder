import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.fill import fillnodata
from scipy.stats import binned_statistic_2d
import laspy
import os

def pointcloud_to_chm(las_file_path, resolution, ground_class=2, tree_class=1):
    las_data = laspy.read(las_file_path)
    points = np.vstack((las_data.x, las_data.y, las_data.z)).T
    classifications = las_data.classification
    ground_mask = classifications == ground_class

    # select poinst based on class mask if there is a tree class, the las file must have a ground class though.
    ground_points = points[ground_mask]
    if tree_class:
        tree_mask = classifications == tree_class
        dsm_points = points[tree_mask]
    else:
        dsm_poinst = points
    gx, gy, gz = ground_points[:, 0], ground_points[:, 1], ground_points[:, 2]
    cx, cy, cz = dsm_points[:, 0], dsm_points[:, 1], dsm_points[:, 2]

    # infer the raster bounds from file
    min_x, max_x = cx.min(), cx.max()
    min_y, max_y = cy.min(), cy.max()
    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))

    # Make digital elevation model using s bining from scipy
    dem_stat, _, _, _ = binned_statistic_2d(gx, gy, gz, statistic='min', bins=[width, height])
    dem = np.flipud(dem_stat.T)# flip axis order to convert back to np

    # Digital surface model
    dsm_stat, _, _, _ = binned_statistic_2d(cx, cy, cz, statistic='max', bins=[width, height])
    dsm = np.flipud(dsm_stat.T)

    # fill missing values
    dem_nan_mask = np.where(np.isnan(dem), 0, 255)
    dem_filled = fillnodata(image=dem, mask=dem_nan_mask, smoothing_iterations=1)

    chm = dsm - dem_filled
    chm = np.nan_to_num(chm, nan=0)  # Final cleanup replace any remaining NaNs with 0

    transform = from_origin(min_x, max_y, resolution, resolution)

    return chm, transform


