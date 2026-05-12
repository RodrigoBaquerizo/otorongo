"""
enrich_rankings.py
Añade la columna player_key al archivo atp_rankings_merged.csv
usando el maestro de normalizacion de nombres.
"""
import pandas as pd
import json
import os
import shutil
import unicodedata
import re

RANKINGS_FILE  = "data/atp_rankings_merged.csv"
BACKUP_FILE    = "data/atp_rankings_merged_BACKUP.csv"
MASTER_CSV     = "data/master-normalizacion-players-key.csv"
MASTER_JSON    = "data/player_master.json"

# ── Normalización idéntica a la de refresh_data.py ─────────────────────────
def normalize_name_robust(name: str) -> str:
    if not name or pd.isna(name) or str(name).strip() == "":
        return ""
    s = str(name).strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = s.replace(".", ". ")
    s = re.sub(r"[^a-z0-9\s.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ── Cargar maestro (igual que load_player_master en refresh_data.py) ────────
def load_alias_map():
    alias_map = {}   # normalized_alias -> player_key (str limpio)

    # 1. JSON
    if os.path.exists(MASTER_JSON):
        try:
            with open(MASTER_JSON, "r") as f:
                master = json.load(f)
            for key, info in master.get("by_key", {}).items():
                p_key = str(key).replace(".0", "").strip()
                for a in [info.get("canonical_name", "")] + info.get("aliases", []):
                    if a:
                        alias_map[a.lower().strip()] = p_key
                        alias_map[normalize_name_robust(a)] = p_key
        except Exception as e:
            print(f"[WARN] Error leyendo player_master.json: {e}")

    # 2. CSV Maestro (tiene prioridad si duplica)
    if os.path.exists(MASTER_CSV):
        try:
            df_m = pd.read_csv(MASTER_CSV)
            for _, row in df_m.iterrows():
                p_name     = str(row.get("player_name", "")).strip()
                p_fullname = str(row.get("player_full_name", "")).strip()
                ppn        = str(row.get("ppn", "")).strip()
                raw_key    = row.get("player_key")
                if pd.isna(raw_key): continue
                p_key = str(int(float(raw_key)))

                for a in [p_name, p_fullname, ppn]:
                    if a and a.lower() != "nan":
                        alias_map[a.lower().strip()] = p_key
                        alias_map[normalize_name_robust(a)] = p_key
        except Exception as e:
            print(f"[WARN] Error leyendo master CSV: {e}")

    return alias_map

def main():
    # ── 0. Backup ─────────────────────────────────────────────────────────
    shutil.copy(RANKINGS_FILE, BACKUP_FILE)
    print(f"✅ Backup creado: {BACKUP_FILE}")

    # ── 1. Cargar rankings ────────────────────────────────────────────────
    df = pd.read_csv(RANKINGS_FILE)
    total = len(df)
    print(f"\n📊 Registros cargados: {total}")

    # ── 2. Cargar maestro ─────────────────────────────────────────────────
    alias_map = load_alias_map()
    print(f"🔑 Entradas en el alias map: {len(alias_map)}")

    # ── 3. Mapear player_key ──────────────────────────────────────────────
    def find_key(name):
        if not name or pd.isna(name): return ""
        raw  = str(name).lower().strip()
        norm = normalize_name_robust(name)
        return alias_map.get(raw) or alias_map.get(norm) or ""

    df["player_key"] = df["player_name"].apply(find_key)

    # ── 4. Reordenar columnas ─────────────────────────────────────────────
    cols = ["player_name", "player_key", "points", "date"]
    extra = [c for c in df.columns if c not in cols]
    df = df[cols + extra]

    # ── 5. Guardar ────────────────────────────────────────────────────────
    df.to_csv(RANKINGS_FILE, index=False, encoding="utf-8")
    print(f"\n💾 Archivo sobreescrito: {RANKINGS_FILE}")

    # ── 6. Reporte ────────────────────────────────────────────────────────
    mapped   = (df["player_key"] != "").sum()
    unmapped = (df["player_key"] == "").sum()
    pct      = mapped / total * 100 if total else 0

    print(f"\n{'='*55}")
    print(f"  Total registros procesados : {total}")
    print(f"  Mapeados con éxito         : {mapped}  ({pct:.1f}%)")
    print(f"  Sin mapear                 : {unmapped}")

    unmapped_names = sorted(df[df["player_key"] == ""]["player_name"].dropna().unique())
    print(f"\n  Nombres únicos sin mapear  : {len(unmapped_names)}")
    if unmapped_names:
        print(f"\n{'─'*55}")
        for n in unmapped_names:
            print(f"  • {n}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
