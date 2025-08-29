from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def reproject_image_to_master(master, mydata, outname, res=None):
    mydata_ds = gdal.Open(mydata)
    if mydata_ds is None:
        raise IOError("GDAL could not open mydata file %s" % mydata)
    mydata_proj = mydata_ds.GetProjection()
    mydata_geotrans = mydata_ds.GetGeoTransform()
    data_type = mydata_ds.GetRasterBand(1).DataType
    n_bands = mydata_ds.RasterCount

    master_ds = gdal.Open(master)
    if master_ds is None:
        raise IOError("GDAL could not open master file %s" % master)
    master_proj = master_ds.GetProjection()
    master_geotrans = master_ds.GetGeoTransform()
    w = master_ds.RasterXSize
    h = master_ds.RasterYSize

    if outname is None:
        outname = mydata

    if res is not None:
        master_geotrans[1] = float(res)
        master_geotrans[-1] = -float(res)

    dst_filename = outname.replace(".tif", "_nested.tif")
    dst_ds = gdal.GetDriverByName('GTiff').Create(dst_filename, w, h, n_bands, data_type)
    dst_ds.SetGeoTransform(master_geotrans)
    dst_ds.SetProjection(master_proj)

    gdal.ReprojectImage(mydata_ds, dst_ds, mydata_proj, master_proj, gdal.GRA_NearestNeighbour)

    dst_ds = None  # Flush to disk

    return dst_filename
coarse_grid = r'C:\batimetria_beagle\malla0.tif'
fine_grid = r'C:\batimetria_beagle\malla_A.tif'
out_name = r'C:\batimetria_beagle\malla_AA.tif'

ratio = 8  # RATIO OF REFINEMENT

## READ COARSE GRID L0
ds = gdal.Open(coarse_grid)

## GET THE SPATIAL RESOLUTION
geo_transform = ds.GetGeoTransform()
res_x = geo_transform[1]
res_y = geo_transform[5]

## GET EXTENT OF THE FINE GRID
fine_grid_shape = fine_grid.replace(".tif", ".shp")
os.system('gdaltindex aux_shape.shp ' + fine_grid)

# CROP COARSE GRID TO THE EXTENSION OF THE FINE GRID
dsClip = gdal.Warp("aux_clip.tif", ds, cutlineDSName='aux_shape.shp',
                   cropToCutline=True, dstNodata=np.nan)

# APPLY REFINEMENT OF RATIO 4 TO THE PREVIOUS GRID
dsRes = gdal.Warp("aux_L1.tif", dsClip, xRes=res_x / ratio, yRes=res_y / ratio, resampleAlg="average")

# FINE DATA (UNALIGNED)
src_filename = fine_grid

# GRID WHERE I WANT TO INTERPOLATE THE FINE DATA
match_filename = 'aux_L1.tif'

print("Nesting grids...")
dst_filename = reproject_image_to_master(match_filename, src_filename, out_name)

# OPTIONS TO CONVERT TO GRD
options = gdal.TranslateOptions(format='netCDF', creationOptions=['FORMAT=NC4'])

# CONVERT COARSE TO GRD AND NC4
output_netcdf = coarse_grid.replace(".tif", ".grd")
gdal.Translate(output_netcdf, coarse_grid, options=options)
os.system('ncrename -v Band1,z ' + output_netcdf)

# CONVERT FINER TO GRD AND NC4
output_netcdf = dst_filename.replace(".tif", ".grd")
gdal.Translate(output_netcdf, dst_filename, options=options)
os.system('ncrename -v Band1,z ' + output_netcdf)

print("Nested meshed created. DONE.")
