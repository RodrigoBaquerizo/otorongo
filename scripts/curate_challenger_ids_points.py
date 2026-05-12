"""
curate_challenger_ids_points.py
Rellena J1 Key / J2 Key faltantes y corrige los Puntos ATP que son 0 o NaN
en data/Challenger Tour Matches.csv, usando:
  - data/player_master.json  (nombre → ID)
  - data/atp_rankings_merged.csv (ID/nombre → puntos históricos por fecha)
"""
import pandas as pd
import json
import shutil
import unicodedata
import re

CHA_FILE     = "data/Challenger Tour Matches.csv"
BACKUP_FILE  = "data/Challenger Tour Matches_CURATION_BACKUP.csv"
RANKINGS_FILE = "data/atp_rankings_merged.csv"
MASTER_FILE  = "data/player_master.json"


# ── Normalización (idéntica a refresh_data.py) ────────────────────────────
def normalize_name(name: str) -> str:
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


# ── 0. Backup ─────────────────────────────────────────────────────────────
shutil.copy(CHA_FILE, BACKUP_FILE)
print(f"✅ Backup: {BACKUP_FILE}\n")


# ── 1. Cargar maestro de jugadores ────────────────────────────────────────
with open(MASTER_FILE) as f:
    master = json.load(f)

# alias_map: normalized_name → key_str
alias_map: dict[str, str] = {}
for alias_raw, key_val in master.get("by_alias", {}).items():
    clean = clean_key(key_val)
    if clean:
        alias_map[alias_raw.lower().strip()] = clean
        alias_map[normalize_name(alias_raw)] = clean

def name_to_key(name: str) -> str:
    raw = str(name).lower().strip()
    return alias_map.get(raw) or alias_map.get(normalize_name(name)) or ""


# ── 2. Construir índice de rankings: key → [(date, points)] ────────────────
print("Cargando rankings históricos…")
df_rank = pd.read_csv(RANKINGS_FILE, dtype={"player_key": str}, low_memory=False)
df_rank["date"] = pd.to_datetime(df_rank["date"], errors="coerce")
df_rank["player_key"] = df_rank["player_key"].fillna("").str.strip()
df_rank["points"] = pd.to_numeric(df_rank["points"], errors="coerce")

# Índice por key
rank_by_key: dict[str, list] = {}
for pk, grp in df_rank[df_rank["player_key"] != ""].groupby("player_key"):
    sorted_g = grp.sort_values("date", ascending=False)
    rank_by_key[pk] = list(zip(sorted_g["date"], sorted_g["points"]))

# Índice por nombre normalizado (fallback)
rank_by_name: dict[str, list] = {}
for norm_name, grp in df_rank.groupby(df_rank["player_name"].apply(normalize_name)):
    if not norm_name: continue
    sorted_g = grp.sort_values("date", ascending=False)
    rank_by_name[norm_name] = list(zip(sorted_g["date"], sorted_g["points"]))

def get_points_at(key: str, name: str, match_date) -> float:
    """Devuelve los puntos más recientes <= match_date para el jugador."""
    if pd.isna(match_date):
        return 0.0
    for src in [rank_by_key.get(key, []), rank_by_name.get(normalize_name(name), [])]:
        for d, pts in src:
            if pd.notna(d) and d <= match_date and pd.notna(pts) and pts > 0:
                return float(pts)
    return 0.0


# ── 3. Cargar Challenger y reparar ────────────────────────────────────────
df = pd.read_csv(CHA_FILE, dtype={"J1 Key": str, "J2 Key": str}, low_memory=False)
original_cols = df.columns.tolist()
rows_total = len(df)

# Parsear fecha del partido
df["_match_date"] = pd.to_datetime(df["Fecha"], format="%m/%d/%y", errors="coerce")

keys_added_j1 = keys_added_j2 = 0
pts_fixed_j1  = pts_fixed_j2  = 0

for idx, row in df.iterrows():
    match_date = row["_match_date"]

    # ── J1 Key ──────────────────────────────────────────────────────────
    j1k = clean_key(row.get("J1 Key", ""))
    if not j1k:
        j1k = name_to_key(row["Jugador 1"])
        if j1k:
            df.at[idx, "J1 Key"] = j1k
            keys_added_j1 += 1
    
    # ── J2 Key ──────────────────────────────────────────────────────────
    j2k = clean_key(row.get("J2 Key", ""))
    if not j2k:
        j2k = name_to_key(row["Jugador 2"])
        if j2k:
            df.at[idx, "J2 Key"] = j2k
            keys_added_j2 += 1

    # ── J1 Puntos ATP ────────────────────────────────────────────────────
    if is_empty(row.get("J1 Puntos ATP")):
        pts = get_points_at(j1k, row["Jugador 1"], match_date)
        if pts > 0:
            df.at[idx, "J1 Puntos ATP"] = int(pts)
            pts_fixed_j1 += 1

    # ── J2 Puntos ATP ────────────────────────────────────────────────────
    if is_empty(row.get("J2 Puntos ATP")):
        pts = get_points_at(j2k, row["Jugador 2"], match_date)
        if pts > 0:
            df.at[idx, "J2 Puntos ATP"] = int(pts)
            pts_fixed_j2 += 1

# ── 4. Guardar (conservar columnas originales, eliminar temporal) ─────────
df = df[original_cols]
assert len(df) == rows_total, "ERROR: pérdida de filas detectada."
df.to_csv(CHA_FILE, index=False, encoding="utf-8")

# ── 5. Reporte ────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Filas totales procesadas   : {rows_total}")
print(f"  J1 Key incorporados        : {keys_added_j1}")
print(f"  J2 Key incorporados        : {keys_added_j2}")
print(f"  J1 Puntos ATP corregidos   : {pts_fixed_j1}")
print(f"  J2 Puntos ATP corregidos   : {pts_fixed_j2}")
print(f"{'='*55}")

# Cuántos siguen sin key / sin puntos tras la curación
df2 = pd.read_csv(CHA_FILE, dtype={"J1 Key": str, "J2 Key": str}, low_memory=False)
j1k_remain = df2["J1 Key"].fillna("").isin(["", "nan"]).sum()
j2k_remain = df2["J2 Key"].fillna("").isin(["", "nan"]).sum()
j1p_remain = (pd.to_numeric(df2["J1 Puntos ATP"], errors="coerce").fillna(0) == 0).sum()
j2p_remain = (pd.to_numeric(df2["J2 Puntos ATP"], errors="coerce").fillna(0) == 0).sum()
print(f"\n  Aún sin J1 Key             : {j1k_remain}")
print(f"  Aún sin J2 Key             : {j2k_remain}")
print(f"  Aún sin J1 Puntos ATP      : {j1p_remain}")
print(f"  Aún sin J2 Puntos ATP      : {j2p_remain}")
print(f"{'='*55}")
