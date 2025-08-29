import os
import netCDF4

folder_path = r"C:\Users\WINDOWS\OneDrive - University of Puerto Rico\Desktop\batimetrias\Mallas_L4_grd_FV"

for file in os.listdir(folder_path):
    if file.startswith("L4_test_") and file.endswith(".grd"):
        file_path = os.path.join(folder_path, file)
        try:
            with netCDF4.Dataset(file_path, "r+") as nc:
                if "Band1" in nc.variables:
                    nc.renameVariable("Band1", "z")
                    print(f"Renombrado correctamente: {file}")
                else:
                    print(f"Advertencia: 'Band1' no encontrado en {file}")
        except Exception as e:
            print(f"Error procesando {file}: {e}")

print("\nProceso de renombrado completado.")
