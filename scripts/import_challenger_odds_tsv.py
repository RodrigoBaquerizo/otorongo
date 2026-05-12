import pandas as pd
from datetime import timedelta
import os

# Rutas de archivos
SOURCE_TSV = "data/repaso-cuotas-challenger.tsv"
TARGET_CSV = "data/Challenger Tour Matches.csv"

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    return name.strip().lower()

def clean_odd(val):
    if pd.isna(val) or val == "":
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    # Reemplazar coma por punto y quitar comillas si existen
    val_str = str(val).replace('"', '').replace(',', '.').strip()
    try:
        return float(val_str)
    except ValueError:
        return 0.0

def run_import():
    if not os.path.exists(SOURCE_TSV):
        print(f"Error: No se encuentra el archivo fuente {SOURCE_TSV}")
        return
    
    print("Cargando datos...")
    # Cargar TSV (separador tabulado)
    try:
        df_tsv = pd.read_csv(SOURCE_TSV, sep='\t')
    except Exception as e:
        print(f"Error leyendo TSV: {e}")
        return

    # Cargar CSV destino
    df_target = pd.read_csv(TARGET_CSV)
    
    # Asegurar que las columnas de cuotas existan y sean numéricas
    for col in ['Cuota J1', 'Cuota J2']:
        if col not in df_target.columns:
            df_target[col] = 0.0
        else:
            df_target[col] = pd.to_numeric(df_target[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)

    # Preparar fechas en el TSV
    # El TSV tiene fechas como 28/1/2026 o 1/2/26
    # Vamos a intentar parsear dinámicamente
    def parse_tsv_date(d):
        try:
            return pd.to_datetime(d, dayfirst=True)
        except:
            return pd.NaT

    df_tsv['Fecha_dt'] = df_tsv['Fecha'].apply(parse_tsv_date)
    
    # Preparar fechas en el Target (usamos la columna Fecha_dt que ya existe)
    df_target['Fecha_dt'] = pd.to_datetime(df_target['Fecha_dt'])

    print(f"Procesando {len(df_tsv)} registros del TSV...")
    
    matches_found = 0
    matches_ambiguous = 0
    updated_rows = 0
    
    # Iterar sobre cada fila del TSV
    for idx_tsv, row_tsv in df_tsv.iterrows():
        if pd.isna(row_tsv['Fecha_dt']):
            continue
            
        p1_tsv = normalize_name(row_tsv['Jugador 1'])
        p2_tsv = normalize_name(row_tsv['Jugador 2'])
        set_tsv = {p1_tsv, p2_tsv}
        
        c1_tsv = clean_odd(row_tsv['Cuota J1'])
        c2_tsv = clean_odd(row_tsv['Cuota J2'])
        
        if c1_tsv == 0 and c2_tsv == 0:
            continue

        # Ventana de ±1 día
        date_min = row_tsv['Fecha_dt'] - timedelta(days=1)
        date_max = row_tsv['Fecha_dt'] + timedelta(days=1)
        
        # Filtrar candidatos por fecha
        candidates = df_target[
            (df_target['Fecha_dt'] >= date_min) & 
            (df_target['Fecha_dt'] <= date_max)
        ]
        
        # Buscar coincidencias de jugadores
        actual_matches = []
        for idx_target, row_target in candidates.iterrows():
            p1_target = normalize_name(row_target['Jugador 1'])
            p2_target = normalize_name(row_target['Jugador 2'])
            set_target = {p1_target, p2_target}
            
            if set_tsv == set_target:
                actual_matches.append(idx_target)
        
        if not actual_matches:
            continue
            
        matches_found += 1
        
        if len(actual_matches) > 1:
            matches_ambiguous += 1
            # Si hay más de uno, tomamos el que tenga la fecha más cercana
            actual_matches.sort(key=lambda i: abs((df_target.at[i, 'Fecha_dt'] - row_tsv['Fecha_dt']).total_seconds()))
        
        # El match ganador
        best_idx = actual_matches[0]
        
        # Determinar qué cuota va a qué jugador en el target
        p1_target = normalize_name(df_target.at[best_idx, 'Jugador 1'])
        
        if p1_target == p1_tsv:
            # Orden coincide
            df_target.at[best_idx, 'Cuota J1'] = c1_tsv
            df_target.at[best_idx, 'Cuota J2'] = c2_tsv
        else:
            # Orden invertido
            df_target.at[best_idx, 'Cuota J1'] = c2_tsv
            df_target.at[best_idx, 'Cuota J2'] = c1_tsv
            
        updated_rows += 1

    print(f"\nResumen de importación:")
    print(f"- Registros TSV procesados: {len(df_tsv)}")
    print(f"- Partidos encontrados: {matches_found}")
    print(f"- Partidos actualizados en CSV: {updated_rows}")
    print(f"- Ambigüedades resueltas (mismos jugadores, misma fecha): {matches_ambiguous}")
    
    if updated_rows > 0:
        # Guardar cambios
        # Mantener el formato original de fecha si es posible o usar YYYY-MM-DD
        df_target.to_csv(TARGET_CSV, index=False)
        print(f"\nArchivo {TARGET_CSV} actualizado con éxito.")
    else:
        print("\nNo se realizaron actualizaciones.")

if __name__ == "__main__":
    run_import()
