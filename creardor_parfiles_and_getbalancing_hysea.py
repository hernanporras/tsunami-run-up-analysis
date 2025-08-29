import os
import pandas as pd

# === CONFIGURACIÓN ===
EXCEL_FILE = 'parametros_VH_2025(Sheet1).csv'
OUTPUT_DIR = os.getcwd()
PROCESOS = 8  # Número de procesos para el balanceo de carga

# === LECTURA DEL ARCHIVO DE PARAMETROS ===
df = pd.read_csv(EXCEL_FILE)

# Verificación rápida de que el archivo tiene suficientes columnas
if df.shape[1] < 13:
    raise ValueError("El archivo debe tener al menos 13 columnas: 10 para los parámetros de Okada y 3 para el ID.")

# === CREACIÓN DE PARFILES Y DIRECTORIOS ===
for index, row in df.iterrows():
    # Extraer los parámetros de Okada (primeras 10 columnas)
    okada_params = row.iloc[:10].values
    # Extraer el ID (columnas 11, 12 y 13)
    ID = f"{row.iloc[10]}_{row.iloc[11]}_{row.iloc[12]}"
    
    # Crear el directorio para el ID
    dir_id = os.path.join(OUTPUT_DIR, ID)
    os.makedirs(dir_id, exist_ok=True)
    
    # Crear el archivo parfile
    parfile_path = os.path.join(dir_id, f"{ID}.parfile")
    with open(parfile_path, 'w') as parfile:
        # Escribir encabezado del parfile
        parfile.write(f"{ID} #Bathymetry name\n")
        parfile.write("batimetrias/mallas_inun/L0_test.grd #Bathymetry file\n")
        parfile.write("1 #Initialization of states (1: Standard Okada,...)\n")
        parfile.write("0 #Apply Kajiura filter to the Okada deformation (0: no, 1: yes)\n")
        parfile.write("1 #Number of faults (>= 1)\n")
        parfile.write("#Time(sec) Lon_epicenter Lat_epicenter Depth_hypocenter(km) Fault_lenght(km) Fault_width(km) Strike Dip Rake Slip(m)\n")
        parfile.write(f"0 {' '.join(map(str, okada_params))}\n")
        parfile.write("0 #Use Okada computation\n")
        parfile.write(f"simulaciones/5L_FV/fuentes_victor/{ID}/{ID}_L0_1920m_FV   #NetCDF file prefix\n")
        parfile.write("1 1 0 0 0 0 0 0 1 1\n")  # Variables saved
        
        # Niveles L1 a L3
        levels_1_to_3 = [
            "L1_test_480m_nested",
            "L2_test_ALL_120m_nested",
            "L3_test_E_30m_nested",
            "L3_test_NE_30m_nested",
            "L3_test_NW_30m_nested",
            "L3_test_W_30m_nested",
            "L3_test_S_30m_nested"
        ]
        for level in levels_1_to_3:
            parfile.write(f"batimetrias/mallas_inun/{level}.grd #Bathymetry file\n")
            prefix = level.split('_')[1]
            parfile.write(f"simulaciones/5L_FV/fuentes_victor/{ID}/{ID}_{prefix}_L3_30m_FV   #NetCDF file prefix\n")
            parfile.write("1 1 0 0 0 0 0 0 1 1\n")
        
        # Niveles L4
        levels_4 = [
            "E01", "E02", "E03", "E04", "E05", "E06", "E07",
            "NE01", "NE02", "NE03",
            "NW01", "NW02", "NW03", "NW04",
            "W01", "W02", "W03", "W04", "W05", "W06",
            "S01", "S02"
        ]
        for level in levels_4:
            parfile.write(f"batimetrias/mallas_inun/L4_test_{level}_7_5m_nested.grd #Bathymetry file\n")
            parfile.write(f"simulaciones/5L_FV/fuentes_victor/{ID}/{ID}_{level}_L4_7_5m_FV_c3   #NetCDF file prefix\n")
            parfile.write("1 1 0 0 0 0 0 0 1 1\n")
        
        # Condiciones de frontera
        parfile.write("1 #Upper border condition (1: open, -1: wall)\n")
        parfile.write("1 #Lower border condition\n")
        parfile.write("1 #Left border condition\n")
        parfile.write("1 #Right border condition\n")
        parfile.write("7201 #Simulation time (sec)\n")
        parfile.write("60 #Saving time (sec)\n")
        parfile.write("0 #Virtual gauges record\n")
        parfile.write("0.5 #CFL\n")
        parfile.write("0.005 #Epsilon (h)\n")
        parfile.write("20 #Threshold for the 2s+WAF scheme (m)\n")
        parfile.write("0.2 0.2 0.2 0.2 0.2 #stability coefficients for each level\n")
        parfile.write("2 #Friction type\n")
        parfile.write("#Manning coeff\n")
        for level in range(5):
            parfile.write(f"batimetrias/Mallas_L4_grd_FV_c3/L{level}_FV.grd\n")
        parfile.write("100  #Maximum allowed velocity of water\n")
        parfile.write("100000  #L\n")
        parfile.write("1000 #H\n")
        parfile.write("0.05 #if (arrival times are stored): Threshold for arrival times (m)\n")
        
    # Crear archivo de balanceo de carga
    lb_file_path = os.path.join(dir_id, f"{ID}_get_loadbalancing.txt")
    with open(lb_file_path, 'w') as lb_file:
        lb_file.write(f"# Load balancing for {ID}\n")
        for i in range(PROCESOS):
            lb_file.write(f"{i} {index % PROCESOS}\n")
    
    print(f"Archivos creados para {ID}:")
    print(f" - {parfile_path}")
    print(f" - {lb_file_path}")

print("\n=== Procesamiento completado ===")
