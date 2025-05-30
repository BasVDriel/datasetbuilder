import numpy as np
from rasterio.transform import from_origin
from rasterio.fill import fillnodata
from scipy.stats import binned_statistic_2d
from skimage import measure
import laspy

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
        dsm_points = points
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


def world_to_pixel(transform, x, y, round=True):
    col, row = ~transform * (x, y)
    if round:
        return int(row), int(col)
    return row, col

def pixel_to_world(transform, row, col):
    x, y = transform * (col, row)
    return x, y


def tree_marker_grid(chm, tree_coords, transform):
    markers_list = []
    markers = np.zeros_like(chm, dtype=np.int32)
    for idx, (x, y) in enumerate(tree_coords, start=1):
        r, c = world_to_pixel(transform, x, y)
        if 0 <= r < markers.shape[0] and 0 <= c < markers.shape[1]:
            markers[r, c] = idx  # Use a different marker label for each point
            markers_list.append([r, c])
    markers_array = np.array(markers_list)

    return markers, markers_array

def filter_labels(labels, regions, markers, markers_array, ratio_thres = 1.5, dist_thres=12, density_thres=0.45):
    """
    Filter labels based on distance to tree coordinates, height, density, and ratio of major axis length to equivalent diameter.
    """
    # find out which labels are in the markers, becaue the watershed loses this information
    tree_regions = {}
    for region in regions:
        if region.label in markers:
            tree_regions[region.label] = region


    filtered_labels = np.zeros_like(labels, dtype=np.int32)
    filtered_tree_regions = {}

    for label, region in tree_regions.items():
        coords = markers_array[label-1]
        centroid = region.centroid
        area = region.area

        dist = np.sqrt((centroid[0] - coords[0])**2 + (centroid[1] - coords[1])**2)

        height = region.intensity_max

        density = region.area / region.area_convex

        # print(f"Label: {label}, Dist: {dist:.2f}, Height: {height:.2f}, Density: {density:.2f}")

        eq_diameter = region.equivalent_diameter_area
        ratio = region.axis_major_length/eq_diameter

        if (area > 5 and ratio < ratio_thres and dist < dist_thres and density > density_thres):
            coords = region.coords
            filtered_tree_regions[region.label] = region

            for r, c in coords:
                filtered_labels[r, c] = label
    return filtered_labels, filtered_tree_regions


def compute_polygons(tree_regions):
    """"
    Takes in a dictionary of tree regions 
    """
    outlines = {}
    for label, region in tree_regions.items():

        # find the extend to properly link it to image tile coordiantes instead of local patch coordinates
        coords = region.coords
        min_x = coords[:, 1].min()
        min_y = coords[:, 0].min()

        # obtain image patches from the raster masks, pad for polygon extraction
        patch = region.image
        pad_width = 1 
        patch_padded = np.pad(patch, pad_width, mode='constant', constant_values=0)

        try:
            outline = measure.find_contours(patch_padded, 0.5)[0]
            if np.array_equal(outline[0, :], outline[-1, :]):

                # convert back to old coordinates and stash for later
                orig_outline = np.zeros_like(outline)
                orig_outline[:, 0] = outline[:, 0] - pad_width + min_y  
                orig_outline[:, 1] = outline[:, 1] - pad_width + min_x 
                outlines[label] = orig_outline
        except Exception as e:
            print(f"Continuing with polygon extraction but somethign went wrong for tree {label}\n {e}")
            continue
        
    return outlines
    