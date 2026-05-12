import pandas as pd
import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import constants (adjust if they are not in root)
ATP26_FILE = "data/ATP Tour 2026 Matches.csv"
CHA26_FILE = "data/Challenger Tour Matches.csv"
HIST_FILE  = "data/atp_challenger_fixtures_2024_2026.csv"

def run_health_scan():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("="*50)
    print("📋 REPORTE DE SALUD DE DATOS - OTORONGO")
    print("Fecha:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*50)
    
    files_to_scan = [
        ("ATP 2026", ATP26_FILE),
        ("Challenger 2026", CHA26_FILE),
        ("Histórico", HIST_FILE)
    ]
    
    total_errors = 0
    
    for label, path in files_to_scan:
        if not os.path.exists(path):
            print(f"⚠️ [WARNING] Archivo {label} no encontrado en {path}")
            continue
            
        print(f"\n🔍 Escaneando {label} ({path})...")
        try:
            df = pd.read_csv(path)
            
            # 1. Auditoría de Keys (Gaps)
            missing_keys = df[df["J1 Key"].isna() | df["J2 Key"].isna()]
            if not missing_keys.empty:
                count = len(missing_keys)
                total_errors += count
                print(f"❌ [FAIL] Se encontraron {count} filas con J1 Key o J2 Key faltantes.")
                # Mostrar ejemplos
                print("   Ejemplos (Índices CSV):", missing_keys.index[:5].tolist())
            else:
                print("✅ [OK] Todas las J1/J2 Keys están presentes.")
                
            # 2. Auditoría de Puntos (0 o N/D en contextos de apuestas)
            if "Puntos" in "".join(df.columns):
                p1_col = "J1 Puntos ATP" if "J1 Puntos ATP" in df.columns else None
                p2_col = "J2 Puntos ATP" if "J2 Puntos ATP" in df.columns else None
                
                if p1_col and p2_col:
                    mask_0 = (df[p1_col].astype(str).isin(['0', '0.0', 'N/D', 'nan', ''])) | \
                             (df[p2_col].astype(str).isin(['0', '0.0', 'N/D', 'nan', '']))
                    zero_pts = df[mask_0]
                    if not zero_pts.empty:
                        count = len(zero_pts)
                        # No sumamos a total_errors porque 0 puede ser real, pero alertamos
                        print(f"⚠️ [ALERT] {count} filas tienen jugadores con 0 puntos o N/D.")
                    else:
                        print("✅ [OK] No se detectaron celdas de puntos vacías o en 0.")
            
            # 3. Auditoría de Superficies
            if "Superficie" in df.columns:
                valid_surfaces = {"Hard", "Clay", "Grass"}
                invalid_surf = df[~df["Superficie"].isin(valid_surfaces)]
                if not invalid_surf.empty:
                    unique_invalid = invalid_surf["Superficie"].unique()
                    total_errors += len(invalid_surf)
                    print(f"❌ [FAIL] Superficies inválidas detectadas: {unique_invalid}")
                else:
                    print("✅ [OK] Todas las superficies son válidas (Hard, Clay, Grass).")
                    
        except Exception as e:
            print(f"🔥 [ERROR] No se pudo leer el archivo {label}: {e}")
            total_errors += 1

    print("\n" + "="*50)
    if total_errors == 0:
        print("🎉 ESTADO DE SALUD: EXCELENTE")
    else:
        print(f"🚩 ESTADO DE SALUD: SE ENCONTRARON {total_errors} PUNTOS DE FALLO")
    print("="*50)

if __name__ == "__main__":
    run_health_scan()
