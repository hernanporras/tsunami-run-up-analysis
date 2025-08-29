#EsteEl código convierte automáticamente todos los archivos .tif en un directorio a formato .grd (NetCDF-4) utilizando GDAL, asegurando que la resolución original se mantenga, almacenando los resultados en una subcarpeta y renombrando la variable Band1 a z.
#Este codigo lo utilice para las mallas tipo e que utilice en elpaper de sensibilidad de manning en PR. 12 Marzo 2025.

from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

input_directory = r'C:\Users\Hernan\OneDrive - University of Puerto Rico\Desktop\Mallas_L4_grd_FV'
output_directory = os.path.join(input_directory, 'grd_outputs')
os.makedirs(output_directory, exist_ok=True)



# Función para reproyectar manteniendo la resolución original
def reproject_same_res(input_tif, output_tif):
    ds = gdal.Open(input_tif)
    geotransform = ds.GetGeoTransform()
    data_type = ds.GetRasterBand(1).DataType

    dst_ds = gdal.GetDriverByName('GTiff').CreateCopy(output_tif, ds, 0)
    dst_ds.SetGeoTransform(geotrans)
    dst_ds.SetProjection(proj)
    dst_ds = None

# Obtener lista de todos los tif en la carpeta
input_tifs = glob.glob(os.path.join(input_directory, '*.tif'))

for tif in input_tifs:
    filename = os.path.basename(tif)
    output_grd = os.path.join(output_directory, filename.replace('.tif', '.grd'))

    # Convertir directamente .tif a .grd
    options = gdal.TranslateOptions(format="NetCDF", creationOptions=['FORMAT=NC4'])
    output_netcdf = os.path.join(output_directory, filename.replace('.tif', '.grd'))
    gdal.Translate(output_netcdf, tif, options=options)
    os.system(f'ncrename -v Band1,z "{output_netcdf}"')
    print(f'Procesado: {output_netcdf}')
