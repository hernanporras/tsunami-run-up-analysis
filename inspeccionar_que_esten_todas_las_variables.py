import os
import netCDF4

folder_path = r"C:\Users\WINDOWS\OneDrive - University of Puerto Rico\Desktop\batimetrias\Mallas_L4_grd_FV"

# Verificar si la carpeta existe
if not os.path.exists(folder_path):
    print(f"Error: La carpeta '{folder_path}' no existe.")
else:
    for file in os.listdir(folder_path):
        if file.endswith(".grd") or file.endswith(".nc"):
            file_path = os.path.join(folder_path, file)
            try:
                with netCDF4.Dataset(file_path, "r") as nc:
                    print(f"\nArchivo: {file}")
                    print("Variables:", list(nc.variables.keys()))
            except Exception as e:
                print(f"Error al leer {file}: {e}")
