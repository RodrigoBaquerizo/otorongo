"""
curate_challenger_v2.py
Versión 2 de la curación del historial de Challenger:
  1. Fuerza IDs oficiales para 5 jugadores de control (aunque ya tengan un ID asignado).
  2. Rellena IDs faltantes via player_master.json.
  3. Restaura Puntos ATP reales (0.0 / vacíos) desde atp_rankings_merged.csv.
"""
import pandas as pd
import json
import shutil
import unicodedata
import re

CHA_FILE      = "data/Challenger Tour Matches.csv"
BACKUP_FILE   = "data/Challenger Tour Matches_BACKUP.csv"
RANKINGS_FILE = "data/atp_rankings_merged.csv"
MASTER_FILE   = "data/player_master.json"

# ── Lista de control de identidad (cualquier ID que no coincida → sobrescribir) ──
FORCE_IDS = {
    # normalized_aliases: id_oficial
    "jesper de jong":               "412",
    "jesper de jong (j. de jong)":  "412",
    "j. de jong":                   "412",
    "j. jong":                      "412",
    "p. dias":                      "2003",
    "pedro boscardin dias":         "2003",
    "p. boscardin dias":            "2003",
    "m. almeida":                   "1743",
    "matheus pucinelli de almeida": "1743",
    "m. pucinelli de almeida":      "1743",
    "j. c. prado angelo":           "21704",
    "juan carlos prado angelo":     "21704",
    "j.c. prado angelo":            "21704",
    "m. rivero":                    "100600",
}


# ── Utilidades ────────────────────────────────────────────────────────────────
def normalize(name: str) -> str:
    if not name or pd.isna(name): return ""
    s = str(name).strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = s.replace(".", ". ")
    s = re.sub(r"[^a-z0-9\s.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_empty(val) -> bool:
    if val is None: return True
    s = str(val).strip()
    return s in ("", "nan", "None", "0", "0.0")


def clean_key(val) -> str:
    try:
        return str(int(float(str(val).strip())))
    except (ValueError, TypeError):
        return ""


# ── 0. Backup ─────────────────────────────────────────────────────────────────
shutil.copy(CHA_FILE, BACKUP_FILE)
print(f"✅ Backup: {BACKUP_FILE}\n")


# ── 1. Maestro alias_map ──────────────────────────────────────────────────────
with open(MASTER_FILE) as f:
    master = json.load(f)

alias_map: dict[str, str] = {}
for alias_raw, key_val in master.get("by_alias", {}).items():
    clean = clean_key(key_val)
    if clean:
        alias_map[alias_raw.lower().strip()] = clean
        alias_map[normalize(alias_raw)] = clean

def name_to_key(name: str) -> str:
    raw = str(name).lower().strip()
    return alias_map.get(raw) or alias_map.get(normalize(name)) or ""


# ── 2. Rankings index: key → [(date, points)], name → [(date, points)] ───────
print("Cargando rankings históricos…")
df_rank = pd.read_csv(RANKINGS_FILE, dtype={"player_key": str}, low_memory=False)
df_rank["date"] = pd.to_datetime(df_rank["date"], errors="coerce")
df_rank["player_key"] = df_rank["player_key"].fillna("").str.strip()
df_rank["points"] = pd.to_numeric(df_rank["points"], errors="coerce")

rank_by_key: dict[str, list] = {}
for pk, grp in df_rank[df_rank["player_key"] != ""].groupby("player_key"):
    sorted_g = grp.sort_values("date", ascending=False)
    rank_by_key[pk] = list(zip(sorted_g["date"], sorted_g["points"]))

rank_by_name: dict[str, list] = {}
for norm_name, grp in df_rank.groupby(df_rank["player_name"].apply(normalize)):
    if not norm_name: continue
    sorted_g = grp.sort_values("date", ascending=False)
    rank_by_name[norm_name] = list(zip(sorted_g["date"], sorted_g["points"]))

def get_points_at(key: str, name: str, match_date) -> float:
    if pd.isna(match_date): return 0.0
    for src in [rank_by_key.get(key, []), rank_by_name.get(normalize(name), [])]:
        for d, pts in src:
            if pd.notna(d) and d <= match_date and pd.notna(pts) and pts > 0:
                return float(pts)
    return 0.0


# ── 3. Cargar Challenger ──────────────────────────────────────────────────────
df = pd.read_csv(CHA_FILE, dtype={"J1 Key": str, "J2 Key": str}, low_memory=False)
original_cols = df.columns.tolist()
rows_total = len(df)
df["_match_date"] = pd.to_datetime(df["Fecha"], format="%m/%d/%y", errors="coerce")

# Contadores
forced_j1 = forced_j2 = 0
keys_added_j1 = keys_added_j2 = 0
pts_fixed_j1 = pts_fixed_j2 = 0

for idx, row in df.iterrows():
    match_date = row["_match_date"]
    name1 = str(row["Jugador 1"]).strip()
    name2 = str(row["Jugador 2"]).strip()

    # ── J1 Key ──────────────────────────────────────────────────────────────
    # Paso A: ¿está en la lista de control forzado?
    forced_id1 = FORCE_IDS.get(name1.lower().strip()) or FORCE_IDS.get(normalize(name1))
    current_j1 = clean_key(row.get("J1 Key", ""))
    if forced_id1 and current_j1 != forced_id1:
        df.at[idx, "J1 Key"] = forced_id1
        current_j1 = forced_id1
        forced_j1 += 1
    # Paso B: si sigue vacío, buscar en maestro
    if not current_j1:
        found = name_to_key(name1)
        if found:
            df.at[idx, "J1 Key"] = found
            current_j1 = found
            keys_added_j1 += 1

    # ── J2 Key ──────────────────────────────────────────────────────────────
    forced_id2 = FORCE_IDS.get(name2.lower().strip()) or FORCE_IDS.get(normalize(name2))
    current_j2 = clean_key(row.get("J2 Key", ""))
    if forced_id2 and current_j2 != forced_id2:
        df.at[idx, "J2 Key"] = forced_id2
        current_j2 = forced_id2
        forced_j2 += 1
    if not current_j2:
        found = name_to_key(name2)
        if found:
            df.at[idx, "J2 Key"] = found
            current_j2 = found
            keys_added_j2 += 1

    # ── J1 Puntos ATP ────────────────────────────────────────────────────────
    if is_empty(row.get("J1 Puntos ATP")):
        pts = get_points_at(current_j1, name1, match_date)
        if pts > 0:
            df.at[idx, "J1 Puntos ATP"] = int(pts)
            pts_fixed_j1 += 1

    # ── J2 Puntos ATP ────────────────────────────────────────────────────────
    if is_empty(row.get("J2 Puntos ATP")):
        pts = get_points_at(current_j2, name2, match_date)
        if pts > 0:
            df.at[idx, "J2 Puntos ATP"] = int(pts)
            pts_fixed_j2 += 1

# ── 4. Guardar ────────────────────────────────────────────────────────────────
df = df[original_cols]
assert len(df) == rows_total, "ERROR: pérdida de filas detectada."
df.to_csv(CHA_FILE, index=False, encoding="utf-8")

# ── 5. Reporte ────────────────────────────────────────────────────────────────
df2 = pd.read_csv(CHA_FILE, dtype={"J1 Key": str, "J2 Key": str}, low_memory=False)
j1k_remain = df2["J1 Key"].fillna("").isin(["", "nan"]).sum()
j2k_remain = df2["J2 Key"].fillna("").isin(["", "nan"]).sum()
j1p_remain = (pd.to_numeric(df2["J1 Puntos ATP"], errors="coerce").fillna(0) == 0).sum()
j2p_remain = (pd.to_numeric(df2["J2 Puntos ATP"], errors="coerce").fillna(0) == 0).sum()

print(f"\n{'='*60}")
print(f"  Filas procesadas                 : {rows_total}")
print(f"  IDs forzados por lista control   : J1={forced_j1}  J2={forced_j2}")
print(f"  IDs nuevos por maestro           : J1={keys_added_j1}  J2={keys_added_j2}")
print(f"  Total IDs incorporados/corregidos: {forced_j1+forced_j2+keys_added_j1+keys_added_j2}")
print(f"  Puntos ATP restaurados           : J1={pts_fixed_j1}  J2={pts_fixed_j2}")
print(f"{'─'*60}")
print(f"  Aún sin J1 Key                   : {j1k_remain}")
print(f"  Aún sin J2 Key                   : {j2k_remain}")
print(f"  Aún sin J1 Puntos ATP (=0)       : {j1p_remain}")
print(f"  Aún sin J2 Puntos ATP (=0)       : {j2p_remain}")
print(f"{'='*60}")
