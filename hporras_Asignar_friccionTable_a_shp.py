import geopandas as gpd
import pandas as pd
import os
import numpy as np

# Definir el directorio donde se encuentran los shapefiles y la tabla de fricciones
directory = r'C:\Users\Hernan\OneDrive - University of Puerto Rico\Desktop\Tsunamis_tesis_II_part\mallas_FV_areas'

# Cargar el DataFrame de fricciones
fricciones_path = os.path.join(directory, 'fricciones2024.csv')
fricciones = pd.read_csv(fricciones_path)
fricciones.replace(',', '.', regex=True, inplace=True)  # Convertir comas en puntos

# Lista de shapefiles para procesar
shapefiles = [
    'L4_test_W06_7_5m_FVP.shp', 'L4_test_W05_7_5m_FVP.shp', 'L4_test_W04_7_5m_FVP.shp',
    'L4_test_W03_7_5m_FVP.shp', 'L4_test_W02_7_5m_FVP.shp', 'L4_test_W01_7_5m_FVP.shp'
]

# Definición de rangos para cada tipo de ID
id_ranges = {
    'a': 10,  
    'b': 17,   
    'c': 4,   
    'd': 9,   
    'e': 10, 
}

# Generación de todos los IDs basados en los rangos definidos
ids = [f'fv_{letter}{i+1}' for letter, count in id_ranges.items() for i in range(count)]

# Procesar cada shapefile
for shapefile in shapefiles:
    file_path = os.path.join(directory, shapefile)
    if os.path.exists(file_path):
        gdf = gpd.read_file(file_path)
        
        # Crear una columna por cada ID
        # Para los IDs de grupo 'e', se asigna inicialmente NaN para que luego se actualicen con los valores del CSV.
        for id in ids:
            if id.startswith('fv_e'):
                gdf[id] = np.nan
            else:
                gdf[id] = 0.03  # Valor predeterminado para los demás grupos

        # Asignar los valores correctos de fricción basados en 'CLASIF_GEN'
        for index, row in gdf.iterrows():
            clasif_gen = row['CLASIF_GEN']
            if clasif_gen in fricciones.columns:
                for id in ids:
                    # Verificar que el ID exista en la columna 'ID' del DataFrame de fricciones
                    if id in fricciones['ID'].values:
                        # Extraer el valor para el id y el grupo correspondiente
                        valor = fricciones.loc[fricciones['ID'] == id, clasif_gen].values[0]
                        if pd.notna(valor):
                            gdf.at[index, id] = float(valor)
        
        # Guardar el GeoDataFrame modificado (sobrescribe el original)
        output_path = os.path.join(directory, shapefile)
        gdf.to_file(output_path)
        print(f'Processed and saved: {output_path}')
