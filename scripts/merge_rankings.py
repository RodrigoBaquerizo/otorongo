import pandas as pd
import os
import glob
def normalize_name(full_name):
    """
    Normaliza el nombre del ranking al formato de los fixtures: 'F. Lastname'
    Si ya parece estar abreviado (ej. 'J.J. Wolf'), lo deja igual.
    """
    if pd.isna(full_name):
        return ""
    
    parts = str(full_name).strip().split()
    if not parts:
        return ""
    
    # Si ya tiene puntos en la primera parte, asumimos que ya está abreviado
    if "." in parts[0]:
        return full_name
    
    # Formato simple: Inicial del primer nombre + Apellido final
    # Ejemplo: 'Alexander Zverev' -> 'A. Zverev'
    firstName = parts[0]
    lastName = parts[-1]
    
    return f"{firstName[0]}. {lastName}"

def merge_rankings():
    base_path = "data"
    folders = ["ATP Rankings 2025", "ATP Rankings 2026"]
    
    all_dfs = []
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        files = glob.glob(os.path.join(folder_path, "*.tsv"))
        print(f"Procesando {len(files)} archivos en {folder}...")
        
        for file in files:
            # Extraer fecha del nombre del archivo: atp_ranking_YYYY-MM-DD.tsv
            filename = os.path.basename(file)
            file_date_str = filename.replace("atp_ranking_", "").replace(".tsv", "")
            
            try:
                # Los archivos no tienen cabecera: Nombre, Puntos, Fecha (que a veces dice 'Current Week')
                df = pd.read_csv(file, sep='\t', header=None, names=['name', 'points', 'junk_date'])
                
                # Normalizar nombres
                df['player_name'] = df['name'].apply(normalize_name)
                df['date'] = file_date_str
                
                # Seleccionar solo columnas necesarias
                df = df[['player_name', 'points', 'date']]
                all_dfs.append(df)
            except Exception as e:
                print(f"Error procesando {file}: {e}")
                
    if not all_dfs:
        print("No se encontraron datos.")
        return
        
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Convertir fecha a datetime para ordenar
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    
    # Eliminar duplicados (a veces hay solapamientos)
    merged_df = merged_df.drop_duplicates(subset=['player_name', 'date'])
    
    # Ordenar por jugador y fecha
    merged_df = merged_df.sort_values(['player_name', 'date'])
    
    output_path = os.path.join(base_path, "atp_rankings_merged.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"\nArchivo consolidado creado: {output_path}")
    print(f"Total de registros: {len(merged_df)}")

if __name__ == "__main__":
    merge_rankings()
