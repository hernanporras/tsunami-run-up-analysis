import os
import netCDF4
import numpy as np

# Ruta de la carpeta donde están los archivos NetCDF
folder_path = r"C:\Users\WINDOWS\OneDrive - University of Puerto Rico\Desktop\batimetrias\Mallas_L4_grd_FV"

# Variables que deben estar presentes en cada archivo
required_variables = ["eta", "ux", "uy"]

# Recorrer todos los archivos en la carpeta
for file in os.listdir(folder_path):
    if file.endswith(".grd") or file.endswith(".nc"):  # Procesa solo archivos NetCDF
        file_path = os.path.join(folder_path, file)
        try:
            with netCDF4.Dataset(file_path, "r+") as nc:
                existing_vars = nc.variables.keys()

                missing_vars = [var for var in required_variables if var not in existing_vars]

                if missing_vars:
                    print(f"Corrigiendo {file}: agregando {missing_vars}")

                    # Tomar dimensiones del archivo
                    dim_x, dim_y = None, None
                    if "lat" in nc.dimensions and "lon" in nc.dimensions:
                        dim_x, dim_y = nc.dimensions["lon"].size, nc.dimensions["lat"].size
                    elif len(nc.dimensions) >= 2:
                        dims = list(nc.dimensions.keys())
                        dim_x, dim_y = nc.dimensions[dims[0]].size, nc.dimensions[dims[1]].size

                    if dim_x is not None and dim_y is not None:
                        # Agregar variables faltantes con valores por defecto (0.0)
                        for var in missing_vars:
                            nc.createVariable(var, "f4", ("lat", "lon"))
                            nc.variables[var][:] = np.zeros((dim_y, dim_x))

                        print(f"{file} corregido correctamente.")
                    else:
                        print(f"No se pudieron determinar las dimensiones de {file}.")
                else:
                    print(f"{file} ya está correcto.")

        except Exception as e:
            print(f"Error procesando {file}: {e}")

print("\nProceso de corrección completado.")
