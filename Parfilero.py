import os
import pandas as pd

# === CONFIGURACIÓN ===
EXCEL_FILE = r'C:\Users\Hernan\OneDrive - University of Puerto Rico\Desktop\simulaciones\parametros_VH_2025.xlsx'
OUTPUT_DIR = r'C:\Users\Hernan\OneDrive - University of Puerto Rico\Desktop\simulaciones'
TEMPLATE_FILE = r'C:\Users\Hernan\OneDrive - University of Puerto Rico\Desktop\simulaciones\19N_1_FV_L4_c3.txt'

# === LECTURA DE DATOS ===
df = pd.read_excel(EXCEL_FILE)

# Crear IDs fusionando las columnas 'Fault_NAME' y 'Fault_COD'
df['ID'] = df['Fault_NAME'].astype(str) + '_' + df['Fault_COD'].astype(str)

# Procesar cada fila del dataframe
for index, row in df.iterrows():
    ID = row['ID']
    fault_parameters = row[['Longitude', 'Latitude', 'Depth', 'Length (km', 'Width (km)', 'Strike', 'Dip', 'Rake', 'Slip', 'Mw']].tolist()
    
    # Crear directorio para cada ID
    dir_path = os.path.join(OUTPUT_DIR, ID)
    os.makedirs(dir_path, exist_ok=True)
    
    # Leer plantilla y reemplazar parámetros de Okada
    with open(TEMPLATE_FILE, 'r') as template_file:
        content = template_file.read()
    
    # Reemplazar parámetros en la línea correspondiente
    fault_line = f"0.0 {fault_parameters[0]} {fault_parameters[1]} {fault_parameters[2]} {fault_parameters[3]} {fault_parameters[4]} {fault_parameters[5]} {fault_parameters[6]} {fault_parameters[7]} {fault_parameters[8]} {fault_parameters[9]}"
    content = content.replace("0.0 -65.867 19.033 4 84.7 33.884 72 60 -135 2.3", fault_line)
    
    # Reemplazar nombres de archivos NetCDF y batimetría
    content = content.replace("19N_1_FV_L4_c3", ID + '_FV_L4_c3')
    content = content.replace("simulaciones/19N_1/19N_1_L0_1920m_FV", f"simulaciones/{ID}/{ID}_L0_1920m_FV")
    content = content.replace("simulaciones/19N_1/19N_1", f"simulaciones/{ID}/{ID}")
    
    # Escribir el archivo de salida en el directorio principal
    output_file = os.path.join(OUTPUT_DIR, f'{ID}_parfile.txt')
    with open(output_file, 'w') as out_file:
        out_file.write(content)
    
    print(f"Archivo creado: {output_file}")

print("Proceso completado.")
