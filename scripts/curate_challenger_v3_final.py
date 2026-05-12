"""
curate_challenger_v3_final.py
Script determinista de curación de IDs y Puntos ATP en Challenger Tour Matches.csv.
Versión final con lista de auditoría ampliada (6 jugadores) e índice de rankings
estructurado como {player_key: sorted_list_of_(date, points)}.
"""
import pandas as pd
import json
import shutil
import unicodedata
import re
from collections import defaultdict

CHA_FILE      = "data/Challenger Tour Matches.csv"
BACKUP_FILE   = "data/Challenger Tour Matches_BACKUP.csv"
RANKINGS_FILE = "data/atp_rankings_merged.csv"
MASTER_FILE   = "data/player_master.json"

# ── Lista de Auditoría: si el nombre normalizado coincide → ID MANDATORIO ──────
AUDIT_IDS: dict[str, str] = {
    # nombre normalizado (sin acentos, puntos, minúsculas) → id oficial
    "j barranco cosano":                "1877",
    "j  barranco cosano":               "1877",
    "barranco cosano":                  "1877",
    "jesper de jong":                   "412",
    "j de jong":                        "412",
    "j jong":                           "412",
    "p dias":                           "2003",
    "pedro boscardin dias":             "2003",
    "p boscardin dias":                 "2003",
    "m almeida":                        "1743",
    "matheus pucinelli de almeida":     "1743",
    "m pucinelli de almeida":           "1743",
    "j c prado angelo":                 "21704",
    "juan carlos prado angelo":         "21704",
    "jc prado angelo":                  "21704",
    "m rivero":                         "100600",
}


# ── Normalización ─────────────────────────────────────────────────────────────
def normalize(name: str) -> str:
    if not name or pd.isna(name): return ""
    s = str(name).strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9\s]", " ", s)   # quitar todo excepto letras, dígitos, espacios
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_key(val) -> str:
    try:
        return str(int(float(str(val).strip())))
    except (ValueError, TypeError):
        return ""


def is_empty(val) -> bool:
    s = str(val).strip() if val is not None and not (isinstance(val, float) and pd.isna(val)) else ""
    return s in ("", "nan", "None", "0", "0.0")


# ── 0. Backup ─────────────────────────────────────────────────────────────────
shutil.copy(CHA_FILE, BACKUP_FILE)
print(f"✅ Backup: {BACKUP_FILE}\n")


# ── 1. Alias map desde player_master.json ─────────────────────────────────────
with open(MASTER_FILE) as f:
    master = json.load(f)

alias_map: dict[str, str] = {}
for alias_raw, key_val in master.get("by_alias", {}).items():
    clean = clean_key(key_val)
    if clean:
        alias_map[alias_raw.lower().strip()] = clean
        alias_map[normalize(alias_raw)] = clean


def name_to_key(name: str) -> str:
    """Busca el player_key para un nombre, primero en auditoría, luego en alias_map."""
    norm = normalize(name)
    raw  = str(name).lower().strip()
    # 1. Lista de auditoría (prioridad máxima)
    if norm in AUDIT_IDS:
        return AUDIT_IDS[norm]
    if raw in AUDIT_IDS:
        return AUDIT_IDS[raw]
    # 2. alias_map general
    return alias_map.get(raw) or alias_map.get(norm) or ""


# ── 2. Índice de rankings: {player_key: [(date, points), ...]} ────────────────
print("Indexando rankings históricos…")
df_rank = pd.read_csv(RANKINGS_FILE, dtype={"player_key": str}, low_memory=False)
df_rank["date"]       = pd.to_datetime(df_rank["date"], format="mixed", errors="coerce")
df_rank["player_key"] = df_rank["player_key"].fillna("").str.strip()
df_rank["points"]     = pd.to_numeric(df_rank["points"], errors="coerce")

# {key: [(date, points), ...] ordenado DESC por fecha}
rank_idx: dict[str, list] = {}
for pk, grp in df_rank[df_rank["player_key"] != ""].groupby("player_key"):
    sg = grp.dropna(subset=["date", "points"]).sort_values("date", ascending=False)
    rank_idx[pk] = list(zip(sg["date"], sg["points"].astype(float)))

# Índice fallback por nombre normalizado
rank_by_name: dict[str, list] = {}
for norm_name, grp in df_rank.groupby(df_rank["player_name"].apply(normalize)):
    if not norm_name: continue
    sg = grp.dropna(subset=["date", "points"]).sort_values("date", ascending=False)
    rank_by_name[norm_name] = list(zip(sg["date"], sg["points"].astype(float)))


def get_points_at(key: str, name: str, match_date) -> float:
    """Devuelve el puntaje más reciente <= match_date. Prioriza key, luego nombre."""
    if pd.isna(match_date): return 0.0
    for src in [rank_idx.get(key, []), rank_by_name.get(normalize(name), [])]:
        for d, pts in src:
            if pd.notna(d) and d <= match_date and pts > 0:
                return pts
    return 0.0


# ── 3. Cargar Challenger ──────────────────────────────────────────────────────
df = pd.read_csv(CHA_FILE, dtype={"J1 Key": str, "J2 Key": str}, low_memory=False)
original_cols = df.columns.tolist()
assert len(original_cols) == 25, f"ERROR: se esperaban 25 columnas, hay {len(original_cols)}"
rows_total = len(df)

df["_match_date"] = pd.to_datetime(df["Fecha"], format="%m/%d/%y", errors="coerce")

# Contadores
audit_corrected   = defaultdict(int)   # nombre → cuántas veces se forzó el ID
ids_added_j1 = ids_added_j2 = 0
pts_fixed_j1 = pts_fixed_j2 = 0
rows_updated = 0

for idx, row in df.iterrows():
    match_date  = row["_match_date"]
    name1       = str(row["Jugador 1"]).strip()
    name2       = str(row["Jugador 2"]).strip()
    norm1, norm2 = normalize(name1), normalize(name2)
    row_changed = False

    # ────────── J1 Key ──────────────────────────────────────────────────────
    current_j1 = clean_key(row.get("J1 Key", ""))
    mandatory1 = AUDIT_IDS.get(norm1) or AUDIT_IDS.get(name1.lower().strip())

    if mandatory1:
        if current_j1 != mandatory1:
            df.at[idx, "J1 Key"] = mandatory1
            current_j1 = mandatory1
            audit_corrected[name1] += 1
            row_changed = True
    elif not current_j1:
        found = name_to_key(name1)
        if found:
            df.at[idx, "J1 Key"] = found
            current_j1 = found
            ids_added_j1 += 1
            row_changed = True

    # ────────── J2 Key ──────────────────────────────────────────────────────
    current_j2 = clean_key(row.get("J2 Key", ""))
    mandatory2 = AUDIT_IDS.get(norm2) or AUDIT_IDS.get(name2.lower().strip())

    if mandatory2:
        if current_j2 != mandatory2:
            df.at[idx, "J2 Key"] = mandatory2
            current_j2 = mandatory2
            audit_corrected[name2] += 1
            row_changed = True
    elif not current_j2:
        found = name_to_key(name2)
        if found:
            df.at[idx, "J2 Key"] = found
            current_j2 = found
            ids_added_j2 += 1
            row_changed = True

    # ────────── J1 Puntos ATP ────────────────────────────────────────────────
    if is_empty(row.get("J1 Puntos ATP")):
        pts = get_points_at(current_j1, name1, match_date)
        if pts > 0:
            df.at[idx, "J1 Puntos ATP"] = int(pts)
            pts_fixed_j1 += 1
            row_changed = True

    # ────────── J2 Puntos ATP ────────────────────────────────────────────────
    if is_empty(row.get("J2 Puntos ATP")):
        pts = get_points_at(current_j2, name2, match_date)
        if pts > 0:
            df.at[idx, "J2 Puntos ATP"] = int(pts)
            pts_fixed_j2 += 1
            row_changed = True

    if row_changed:
        rows_updated += 1

# ── 4. Guardar ────────────────────────────────────────────────────────────────
df = df[original_cols]
assert len(df) == rows_total, "ERROR: pérdida de filas detectada."
df.to_csv(CHA_FILE, index=False, encoding="utf-8")

# ── 5. Reporte ────────────────────────────────────────────────────────────────
total_ids_corrected = sum(audit_corrected.values()) + ids_added_j1 + ids_added_j2

print(f"\n{'='*62}")
print(f"  [OK] IDs corregidos por Lista de Auditoría:")
audit_names_map = {
    "j barranco cosano": "J. Barranco Cosano (1877)",
    "jesper de jong":    "Jesper de Jong (412)",
    "j de jong":         "Jesper de Jong (412)",
    "p dias":            "P. Dias (2003)",
    "m almeida":         "M. Almeida (1743)",
    "j c prado angelo":  "J.C. Prado Angelo (21704)",
    "m rivero":          "M. Rivero (100600)",
}
if audit_corrected:
    for name, cnt in sorted(audit_corrected.items()):
        print(f"       • {name}: {cnt} fila(s) corregida(s)")
else:
    print("       • Todos los IDs de auditoría ya eran correctos.")
print(f"{'─'*62}")
print(f"  [INFO] Total de Filas Actualizadas       : {rows_updated}")
print(f"  [INFO] Total de IDs nuevos incorporados  : {ids_added_j1 + ids_added_j2}")
print(f"  [INFO] Puntos ATP restaurados (de 0 → real): J1={pts_fixed_j1}  J2={pts_fixed_j2}  Total={pts_fixed_j1+pts_fixed_j2}")
print(f"{'='*62}")

# ── 6. Validación fila Barranco ───────────────────────────────────────────────
df_check = pd.read_csv(CHA_FILE, dtype={"J1 Key": str, "J2 Key": str}, low_memory=False)
mask_ba = df_check["Jugador 1"].str.contains("Barranco", case=False, na=False) | \
          df_check["Jugador 2"].str.contains("Barranco", case=False, na=False)
found_rows = df_check[mask_ba]

print(f"\n  [VALIDACIÓN] Filas de J. Barranco Cosano tras curación:")
print(f"{'─'*62}")

for _, r in found_rows.iterrows():
    if "Barranco" in str(r["Jugador 1"]):
        key_val = r["J1 Key"]
        pts_val = r["J1 Puntos ATP"]
        pos = "J1"
    else:
        key_val = r["J2 Key"]
        pts_val = r["J2 Puntos ATP"]
        pos = "J2"
    ok = "✅" if str(key_val).strip() == "1877" else "⚠️"
    print(f"  {ok} [{r['Fecha']}] {r['Torneo']} | {pos} Key={key_val} | Puntos={pts_val}")
print(f"{'='*62}")
