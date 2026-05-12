"""
Script para regenerar data/tournaments.csv con todos los torneos de la API,
aplicando la normalización de superficies definida en el proyecto:
  - Cualquier valor que contenga 'hard'  → 'Hard'
  - Cualquier valor que contenga 'clay'  → 'Clay'
  - Cualquier valor que contenga 'grass' → 'Grass'
  - Cualquier otro valor                 → se conserva tal cual

Fuente: api.api-tennis.com → method=get_tournaments
Output: data/tournaments.csv
"""

import os
import requests
import pandas as pd
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
API_KEY    = os.getenv("ATP_API_KEY", "0ac1103dc814b18c8425331baf0a8b9597f1337a6c0291621becd0277ecb7a1a")
API_URL    = "https://api.api-tennis.com/tennis/"
OUTPUT_PATH = "data/tournaments.csv"


DAVIS_CUP_KEYWORDS = ["davis cup"]

# Overrides específicos solicitados por el usuario
TOURNAMENT_OVERRIDES = {
    "Brasilia 2": "Clay",
    "Fujairah":   "Hard",
    "Metepec":    "Hard"
}

def normalize_surface(raw: str, tournament_name: str = "") -> str:
    """
    Aplica la normalización de superficies del proyecto:
      - Overrides específicos (Brasilia 2, Fujairah, Metepec)
      - 'hard' (incl. 'Hard (Indoor)', 'Hard (Outdoor)') → 'Hard'
      - 'clay'                                           → 'Clay'
      - 'grass'                                          → 'Grass'
      - Davis Cup sin superficie                         → 'Hard'
      - Cualquier otro valor                             → se conserva tal cual
    """
    t_name = str(tournament_name).strip()
    
    # 1. Overrides específicos (tienen la máxima prioridad)
    if t_name in TOURNAMENT_OVERRIDES:
        return TOURNAMENT_OVERRIDES[t_name]

    s = str(raw).lower().strip()

    # 2. Superficie explícita de la API
    if "hard" in s:
        return "Hard"
    if "clay" in s:
        return "Clay"
    if "grass" in s:
        return "Grass"

    # 3. Davis Cup: superficie varía por sede pero se asigna Hard como default del proyecto
    t_lower = t_name.lower()
    if any(kw in t_lower for kw in DAVIS_CUP_KEYWORDS):
        return "Hard"

    return str(raw).strip()  # conservar valor original para otros casos no reconocidos


def fetch_tournaments() -> list:
    """Llama a get_tournaments y devuelve la lista de resultados."""
    print("  Llamando a get_tournaments... ", end="", flush=True)
    resp = requests.get(
        API_URL,
        params={"method": "get_tournaments", "APIkey": API_KEY},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("success") != 1:
        raise ValueError(f"La API devolvió success≠1: {data}")
    results = data["result"]
    print(f"✓  ({len(results):,} torneos)")
    return results


def build_dataframe(results: list) -> pd.DataFrame:
    """Convierte la lista de torneos en un DataFrame normalizado."""
    rows = []
    for t in results:
        surface_raw = t.get("tournament_sourface", "")
        rows.append({
            "tournament_key":      t.get("tournament_key"),
            "tournament_name":     t.get("tournament_name"),
            "event_type_key":      t.get("event_type_key"),
            "event_type_type":     t.get("event_type_type"),
            "tournament_sourface": normalize_surface(surface_raw, t.get("tournament_name", "")),
            "download_time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("  Generando tournaments.csv desde api-tennis.com")
    print("=" * 60)

    # 1. Fetch
    print("\n[1/3] Descargando torneos...")
    results = fetch_tournaments()

    # 2. Build & normalize
    print("\n[2/3] Normalizando superficies...")
    df = build_dataframe(results)

    # Resumen de superficies
    surf_counts = df["tournament_sourface"].value_counts()
    print("  Distribución de superficies:")
    for surf, cnt in surf_counts.items():
        print(f"    {surf:<20} {cnt:>5}")

    # 3. Save
    print(f"\n[3/3] Guardando en {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  ✓ {len(df):,} torneos guardados.")

    # Quick validation
    print("\n" + "=" * 60)
    print("  VALIDACIÓN")
    print("=" * 60)
    df_check = pd.read_csv(OUTPUT_PATH)
    print(f"  Filas en CSV: {len(df_check):,}")
    print(f"  Columnas:     {list(df_check.columns)}")
    print(f"\n  Primeras 3 filas:")
    print(df_check.head(3).to_string(index=False))
    print("\n  ✅ Listo.")


if __name__ == "__main__":
    main()
