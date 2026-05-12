"""
sanitize_ids.py
Estandariza todos los player_key / J1 Key / J2 Key / ID Partido del proyecto
de "123.0" (float-string) a "123" (int-string limpio).

Archivos cubiertos:
  - data/player_master.json
  - data/master-normalizacion-players-key.csv
  - data/atp_rankings_merged.csv
  - data/ATP Tour 2026 Matches.csv
  - data/Challenger Tour Matches.csv
  - data/atp_challenger_fixtures_2024_2026.csv
"""

import pandas as pd
import json
import os
import shutil

# ── Rutas ────────────────────────────────────────────────────────────────────
BACKUP_DIR   = "data/backups_sanitization"
FILES_CSV = [
    ("data/atp_rankings_merged.csv",                  ["player_key"]),
    ("data/master-normalizacion-players-key.csv",     ["player_key"]),
    ("data/ATP Tour 2026 Matches.csv",                ["J1 Key", "J2 Key", "ID Partido"]),
    ("data/Challenger Tour Matches.csv",              ["J1 Key", "J2 Key", "ID Partido"]),
    ("data/atp_challenger_fixtures_2024_2026.csv",    ["J1 Key", "J2 Key", "ID Partido"]),
]
JSON_FILE = "data/player_master.json"


# ── Utilidad ─────────────────────────────────────────────────────────────────
def clean_id(val) -> str:
    """Convierte cualquier representación de ID a string entero limpio.
    Devuelve '' si el valor es nulo, vacío o no numérico."""
    if val is None:
        return ""
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return ""
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s  # no es un número → lo dejamos tal cual


# ── 0. Crear backups ──────────────────────────────────────────────────────────
os.makedirs(BACKUP_DIR, exist_ok=True)
for path, _ in FILES_CSV:
    if os.path.exists(path):
        shutil.copy(path, os.path.join(BACKUP_DIR, os.path.basename(path)))
if os.path.exists(JSON_FILE):
    shutil.copy(JSON_FILE, os.path.join(BACKUP_DIR, os.path.basename(JSON_FILE)))
print(f"✅ Backups creados en: {BACKUP_DIR}/\n")


# ── 1. Sanitizar CSVs ────────────────────────────────────────────────────────
for path, id_cols in FILES_CSV:
    if not os.path.exists(path):
        print(f"⚠️  No encontrado: {path}")
        continue

    # Leer con todas las columnas de ID como string para evitar que pandas infiera float
    dtype_map = {col: str for col in id_cols}
    df = pd.read_csv(path, dtype=dtype_map, low_memory=False)
    rows_before = len(df)
    original_cols = df.columns.tolist()

    for col in id_cols:
        if col not in df.columns:
            continue
        before_sample = df[col].dropna().head(3).tolist()
        df[col] = df[col].apply(clean_id)
        after_sample = df[col].dropna().head(3).tolist()

    df = df[original_cols]  # mantener orden exacto de columnas
    rows_after = len(df)

    assert rows_before == rows_after, f"ERROR: pérdida de filas en {path}!"
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"✅ {path}")
    print(f"   Filas: {rows_before} → {rows_after}  |  Cols sanitizadas: {id_cols}")


# ── 2. Sanitizar player_master.json ─────────────────────────────────────────
print()
if not os.path.exists(JSON_FILE):
    print(f"⚠️  No encontrado: {JSON_FILE}")
else:
    with open(JSON_FILE, "r") as f:
        master = json.load(f)

    # ── 2a. by_alias: valores limpios ─────────────────────────────────────
    old_alias = master.get("by_alias", {})
    new_alias = {}
    for alias, key_val in old_alias.items():
        clean = clean_id(key_val)
        if clean:
            new_alias[alias] = clean
        else:
            new_alias[alias] = key_val  # preservar aunque no sea número
    master["by_alias"] = new_alias

    # ── 2b. by_key: fusionar claves duplicadas (ej: "395.0" y "395") ──────
    old_by_key = master.get("by_key", {})
    new_by_key = {}
    merge_count = 0

    for raw_key, info in old_by_key.items():
        clean_key = clean_id(raw_key)
        if not clean_key:
            clean_key = raw_key  # preservar si no es número

        if clean_key not in new_by_key:
            # Primera vez que vemos esta key limpia
            new_by_key[clean_key] = {
                "canonical_name": info.get("canonical_name", ""),
                "aliases": list(info.get("aliases", []))
            }
        else:
            # Fusión: esta key ya existe con distinto raw_key (ej: "395.0" vs "395")
            merge_count += 1
            existing = new_by_key[clean_key]
            # Mantener canonical_name más largo o el existente
            if len(info.get("canonical_name", "")) > len(existing["canonical_name"]):
                existing["canonical_name"] = info["canonical_name"]
            # Fusionar aliases sin duplicar
            merged_aliases = list(set(existing["aliases"] + list(info.get("aliases", []))))
            existing["aliases"] = merged_aliases

    master["by_key"] = new_by_key

    with open(JSON_FILE, "w") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)

    print(f"✅ {JSON_FILE}")
    print(f"   by_alias: {len(old_alias)} entradas procesadas")
    print(f"   by_key  : {len(old_by_key)} → {len(new_by_key)} llaves ({merge_count} fusiones)")


# ── 3. Verificación rápida de integridad ────────────────────────────────────
print("\n" + "="*55)
print("  VERIFICACIÓN DE INTEGRIDAD")
print("="*55)
for path, id_cols in FILES_CSV:
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path, dtype={col: str for col in id_cols}, low_memory=False)
    # Buscar cualquier valor que aún contenga ".0"
    dirty = 0
    for col in id_cols:
        if col in df.columns:
            dirty += df[col].fillna("").str.endswith(".0").sum()
    status = "✅ limpio" if dirty == 0 else f"⚠️  {dirty} IDs con '.0' restantes"
    print(f"  {os.path.basename(path):45s} {status}")
print("="*55)
