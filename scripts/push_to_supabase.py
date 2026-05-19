"""
Script de sincronización masiva: CSV Local → Supabase.
Fase 1 del plan de estabilización del Proyecto Otorongo.
Este script lee los CSV locales más actualizados y los sube a Supabase.
Mantiene el entorno local completamente aislado al ejecutarse únicamente
con variables de entorno explícitas.

Uso:
    SUPABASE_URL="https://xxx.supabase.co" SUPABASE_KEY="eyJ..." \
    python scripts/push_to_supabase.py
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from supabase import create_client

# Configurar path raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. Obtener credenciales de Supabase de manera explícita
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ ERROR: Las variables de entorno SUPABASE_URL y SUPABASE_KEY son requeridas.")
    print("Ejecución recomendada:")
    print("  SUPABASE_URL=\"https://lvjufittmoflbcvvcqtc.supabase.co\" \\")
    print("  SUPABASE_KEY=\"tu_key_aqui\" \\")
    print("  .venv/bin/python scripts/push_to_supabase.py")
    sys.exit(1)

print("=" * 60)
print("🚀 INICIANDO SINCRONIZACIÓN MASIVA LOCAL -> SUPABASE (CORREGIDO)")
print(f"🔗 URL del Servidor: {SUPABASE_URL}")
print("=" * 60)

try:
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"❌ Error crítico inicializando el cliente Supabase: {e}")
    sys.exit(1)

# Definir la configuración de sincronización (Archivos locales, tablas remotas y columnas del esquema)
SYNC_CONFIG = {
    "tournaments": {
        "csv_path": "data/tournaments.csv",
        "table_name": "tournaments",
        "pk": "tournament_key",
        "columns": ["tournament_key", "tournament_name", "tournament_sourface"],
        "dtype": {"tournament_key": str}
    },
    "atp_matches": {
        "csv_path": "data/ATP Tour 2026 Matches.csv",
        "table_name": "atp_matches",
        "pk": "ID Partido",
        "columns": [
            "ID Partido", "Torneo", "Fecha", "Superficie",
            "Jugador 1", "J1 Key", "J1 Puntos ATP",
            "Jugador 2", "J2 Key", "J2 Puntos ATP",
            "J1 H2H", "J1 H2H %", "J2 H2H", "J2 H2H %",
            "J1 Rend. Reciente", "J1 Rend. Superficie", "Rend. Ultra reciente J1",
            "J2 Rend. Reciente", "J2 Rend. Superficie", "Rend. Ultra reciente J2",
            "Cuota J1", "Cuota J2", "Ganador", "Hora"
        ],
        "dtype": {"J1 Key": str, "J2 Key": str, "ID Partido": str}
    },
    "challenger_matches": {
        "csv_path": "data/Challenger Tour Matches.csv",
        "table_name": "challenger_matches",
        "pk": "ID Partido",
        "columns": [
            "ID Partido", "Torneo", "Fecha", "Superficie",
            "Jugador 1", "J1 Key", "J1 Puntos ATP",
            "Jugador 2", "J2 Key", "J2 Puntos ATP",
            "J1 H2H", "J1 H2H %", "J2 H2H", "J2 H2H %",
            "J1 Rend. Reciente", "J1 Rend. Superficie", "Rend. Ultra reciente J1",
            "J2 Rend. Reciente", "J2 Rend. Superficie", "Rend. Ultra reciente J2",
            "Cuota J1", "Cuota J2", "Ganador", "Fecha_dt", "Hora"
        ],
        "dtype": {"J1 Key": str, "J2 Key": str, "ID Partido": str}
    },
    "historical_fixtures": {
        "csv_path": "data/atp_challenger_fixtures_2024_2026.csv",
        "table_name": "historical_fixtures",
        "pk": "ID Partido",
        "columns": [
            "ID Partido", "Fecha", "Torneo", "Superficie",
            "Jugador 1", "J1 Key", "Jugador 2", "J2 Key",
            "Ganador", "Hora"
        ],
        "dtype": {"J1 Key": str, "J2 Key": str, "ID Partido": str}
    }
}

def clean_record(val):
    """Sanitiza los valores individuales para serialización JSON compatible con PostgreSQL."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, pd.Timestamp):
        return str(val)
    return str(val).strip()

def process_and_upload(name, config):
    print(f"\n📂 Procesando tabla: {config['table_name']}...")
    csv_path = config["csv_path"]
    
    if not os.path.exists(csv_path):
        print(f"  ⚠️ Archivo no encontrado: {csv_path}. Saltando.")
        return False

    # 1. Cargar archivo local
    try:
        df = pd.read_csv(csv_path, dtype=config.get("dtype"))
        print(f"  ✅ Leído CSV local: {len(df)} filas.")
    except Exception as e:
        print(f"  ❌ Error leyendo CSV {csv_path}: {e}")
        return False

    # 2. Filtrar columnas para coincidir exactamente con el esquema de Supabase
    target_columns = config["columns"]
    missing_cols = [col for col in target_columns if col not in df.columns]
    
    if missing_cols:
        print(f"  ⚠️ Columnas faltantes en el CSV que se inicializarán como nulas: {missing_cols}")
        for col in missing_cols:
            df[col] = None
            
    # Quedarse solo con las columnas válidas del esquema
    df_filtered = df[target_columns].copy()
    print(f"  ✅ Columnas filtradas a las {len(target_columns)} del esquema de Supabase.")

    # 3. Formatear/limpiar claves primarias e inyectar IDs generados si faltan
    pk_col = config["pk"]
    if pk_col == "ID Partido":
        # Reemplazar representaciones de null por None real
        df_filtered[pk_col] = df_filtered[pk_col].astype(str).str.strip()
        df_filtered[pk_col] = df_filtered[pk_col].replace(["nan", "None", ""], None)
        
        # Generar IDs basados en metadatos para las celdas nulas
        mask = df_filtered[pk_col].isnull()
        if mask.any():
            def generate_metadata_id(row):
                fecha = str(row.get("Fecha", "")).strip()
                torneo = str(row.get("Torneo", "")).strip().replace(" ", "")
                j1 = str(row.get("Jugador 1", "")).strip().replace(" ", "")
                j2 = str(row.get("Jugador 2", "")).strip().replace(" ", "")
                return f"{fecha}_{torneo}_{j1}_{j2}".lower()
            
            df_filtered.loc[mask, pk_col] = df_filtered[mask].apply(generate_metadata_id, axis=1)
            print(f"  🔑 Generados {mask.sum()} IDs de partido basados en metadatos para evitar descartes.")
    else:
        df_filtered[pk_col] = df_filtered[pk_col].astype(str).str.strip()
        df_filtered = df_filtered[~df_filtered[pk_col].isin(["nan", "None", ""])]

    # 4. Sanitizar registros para la subida
    records = df_filtered.to_dict(orient="records")
    clean_records = []
    for rec in records:
        clean_rec = {k: clean_record(v) for k, v in rec.items()}
        clean_records.append(clean_rec)

    # 5. Deduplicar registros en base a la clave primaria (conservando el último registro en aparecer)
    deduped_dict = {}
    for rec in clean_records:
        pk_val = rec.get(pk_col)
        if pk_val:
            deduped_dict[pk_val] = rec
    
    deduped_records = list(deduped_dict.values())
    n_original = len(clean_records)
    n_deduped = len(deduped_records)
    if n_original != n_deduped:
        print(f"  🧹 Deduplicación: Se redujeron de {n_original} a {n_deduped} registros únicos por PK '{pk_col}'.")
    else:
        print(f"  🧹 No se encontraron duplicados por PK '{pk_col}'. Total: {n_deduped} registros.")

    # 6. Subir por lotes (batches de 500 registros)
    batch_size = 500
    total_batches = (len(deduped_records) + batch_size - 1) // batch_size
    print(f"  📤 Iniciando subida en {total_batches} lotes a la tabla '{config['table_name']}'...")

    t_start = time.time()
    for i in range(0, len(deduped_records), batch_size):
        batch = deduped_records[i : i + batch_size]
        batch_num = i // batch_size + 1
        
        # Intentos con backoff simple
        success = False
        for attempt in range(1, 4):
            try:
                client.table(config["table_name"]).upsert(batch).execute()
                success = True
                break
            except Exception as ex:
                print(f"    ⚠️ Intento {attempt} fallido para lote {batch_num}: {ex}")
                time.sleep(2 * attempt)
                
        if not success:
            print(f"  ❌ ERROR CRÍTICO: No se pudo subir el lote {batch_num} después de 3 intentos. Sincronización cancelada.")
            return False
            
        print(f"    [Lote {batch_num}/{total_batches}] {len(batch)} registros subidos exitosamente.")
        
    t_elapsed = time.time() - t_start
    print(f"  🎉 Carga completa para '{config['table_name']}' en {t_elapsed:.1f}s.")
    return True

# ── Ejecución de la sincronización para todas las tablas ──
success_all = True
t_total_start = time.time()

for name, config in SYNC_CONFIG.items():
    success = process_and_upload(name, config)
    if not success:
        success_all = False
        print(f"\n❌ Sincronización interrumpida debido a un error en '{name}'.")
        break

t_total_elapsed = time.time() - t_total_start
print("\n" + "=" * 60)
if success_all:
    print(f"🎉 ¡SINCRONIZACIÓN EXITOSA COMPLETADA EN {t_total_elapsed:.1f}s!")
else:
    print(f"❌ LA SINCRONIZACIÓN FALLÓ O FUE INCOMPLETA. Tiempo: {t_total_elapsed:.1f}s")
print("=" * 60)
