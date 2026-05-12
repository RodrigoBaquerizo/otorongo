"""
rescue_rankings_keys.py
Rescata los player_key faltantes en atp_rankings_merged.csv
cruzando con el histórico de partidos atp_challenger_fixtures_2024_2026.csv.
Actualiza también el maestro CSV como doble guardado.
"""
import pandas as pd
import shutil
import os
import re
import unicodedata

RANKINGS_FILE = "data/atp_rankings_merged.csv"
HIST_FILE     = "data/atp_challenger_fixtures_2024_2026.csv"
MASTER_CSV    = "data/master-normalizacion-players-key.csv"

RANKINGS_BACKUP = "data/atp_rankings_merged_RESCUE_BACKUP.csv"
MASTER_BACKUP   = "data/master-normalizacion-players-key_RESCUE_BACKUP.csv"


def normalize(name: str) -> str:
    """Normalización agresiva: sin acentos, sin puntos, minúsculas, sin espacios extra."""
    if not name or pd.isna(name): return ""
    s = str(name).strip().lower()
    # Quitar acentos
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    # Quitar puntos
    s = s.replace(".", "")
    # Quitar caracteres especiales, mantener letras, números, espacios y guiones
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    # Compactar espacios
    s = re.sub(r"\s+", " ", s).strip()
    return s


def initial_plus_surname(name: str) -> str:
    """
    De 'N. Alvarez Varona' extrae 'n alvarez', es decir:
    primera letra del primer token + primer apellido.
    Sirve para cruzar 'A. Alvarez' con 'A. Alvarez Varona'.
    """
    norm = normalize(name)
    tokens = norm.split()
    if len(tokens) < 2:
        return norm
    # primera letra del primer token (inicial) + segundo token (primer apellido)
    return f"{tokens[0][0]} {tokens[1]}"


def is_valid_key(k) -> bool:
    """Valida que la clave sea un número limpio."""
    if k is None or pd.isna(k): return False
    s = str(k).replace(".0", "").strip()
    return s.isdigit() and s != "0"


def clean_key(k) -> str:
    return str(int(float(str(k).strip()))).strip()


def main():
    # ── 0. Backups ────────────────────────────────────────────────────────
    shutil.copy(RANKINGS_FILE, RANKINGS_BACKUP)
    shutil.copy(MASTER_CSV,    MASTER_BACKUP)
    print(f"✅ Backups creados:\n   {RANKINGS_BACKUP}\n   {MASTER_BACKUP}\n")

    # ── 1. Cargar rankings ────────────────────────────────────────────────
    df_rank = pd.read_csv(RANKINGS_FILE, dtype={"player_key": str})
    df_rank["player_key"] = df_rank["player_key"].fillna("").str.strip()
    total = len(df_rank)

    missing_mask = df_rank["player_key"] == ""
    missing_names = set(df_rank[missing_mask]["player_name"].dropna())
    print(f"📊 Registros en rankings  : {total}")
    print(f"⚠️  Nombres únicos sin key : {len(missing_names)}")

    # ── 2. Construir mapa (normalized_name -> key) desde el histórico ─────
    df_hist = pd.read_csv(HIST_FILE, low_memory=False)

    hist_map: dict[str, str] = {}         # norm_name (exacto) -> key
    hist_map_initial: dict[str, list] = {} # inicial+apellido -> [(key, full_norm)]

    for col_name, col_key in [("Jugador 1", "J1 Key"), ("Jugador 2", "J2 Key")]:
        if col_name not in df_hist.columns or col_key not in df_hist.columns:
            continue
        sub = df_hist[[col_name, col_key]].dropna(subset=[col_key])
        for _, row in sub.iterrows():
            k = row[col_key]
            name = row[col_name]
            if not is_valid_key(k): continue
            k_clean = clean_key(k)
            norm = normalize(name)
            ini_sur = initial_plus_surname(name)
            # Cruce exacto
            if norm and norm not in hist_map:
                hist_map[norm] = k_clean
            # Cruce por inicial+apellido (puede haber colisiones -> guardar lista)
            if ini_sur:
                if ini_sur not in hist_map_initial:
                    hist_map_initial[ini_sur] = []
                if not any(e[0] == k_clean for e in hist_map_initial[ini_sur]):
                    hist_map_initial[ini_sur].append((k_clean, norm))

    print(f"🗺️  Entradas en mapa histórico: {len(hist_map)}\n")

    # ── 3. Rescatar keys faltantes ────────────────────────────────────────
    rescued_map: dict[str, str] = {}  # nombre_original -> key rescatada
    collision_skip = 0  # cuantos se saltaron por ambigüedad

    for name in missing_names:
        norm = normalize(name)
        ini_sur = initial_plus_surname(name)

        # Paso 1: cruce exacto
        if norm in hist_map:
            rescued_map[name] = hist_map[norm]
            continue

        # Paso 2: cruce por inicial + apellido (solo si hay exactamente 1 candidato)
        candidates = hist_map_initial.get(ini_sur, [])
        if len(candidates) == 1:
            rescued_map[name] = candidates[0][0]
        elif len(candidates) > 1:
            collision_skip += 1  # ambigüedad: omitir para no asignar key incorrecta

    rescued_count = len(rescued_map)
    print(f"🎯 Nombres rescatados : {rescued_count} / {len(missing_names)}")

    # ── 4. Aplicar al DataFrame de rankings ───────────────────────────────
    name_to_key = rescued_map.copy()

    def apply_key(row):
        if row["player_key"] != "":
            return row["player_key"]
        return name_to_key.get(row["player_name"], "")

    df_rank["player_key"] = df_rank.apply(apply_key, axis=1)

    # ── 5. Guardar rankings actualizado ───────────────────────────────────
    df_rank.to_csv(RANKINGS_FILE, index=False, encoding="utf-8")
    print(f"\n💾 Rankings guardado: {RANKINGS_FILE}")

    # ── 6. Actualizar maestro CSV (sin duplicar IDs) ───────────────────────
    df_master = pd.read_csv(MASTER_CSV, dtype={"player_key": str})
    df_master["player_key"] = df_master["player_key"].fillna("").str.strip()

    # Construir set de (player_name lower, key) ya existentes para deduplicar
    existing_pairs = set(
        zip(df_master["player_name"].str.lower().str.strip(), df_master["player_key"])
    )

    new_rows = []
    for orig_name, key in rescued_map.items():
        pair = (orig_name.lower().strip(), key)
        if pair not in existing_pairs:
            new_rows.append({
                "player_name": orig_name,
                "player_key":  key,
                "ppn":         "",
                "player_full_name": "",
                "matched_by":  "hist_rescue",
                "fuzzy_score": "",
                "is_correct":  "pending",
            })
            existing_pairs.add(pair)

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_master = pd.concat([df_master, df_new], ignore_index=True)
        df_master.to_csv(MASTER_CSV, index=False, encoding="utf-8")
        print(f"📋 Maestro actualizado: +{len(new_rows)} entradas nuevas → {MASTER_CSV}")
    else:
        print("📋 Maestro: no se añadieron entradas (todas ya existían).")

    # ── 7. Reporte final ──────────────────────────────────────────────────
    still_missing = sorted(
        df_rank[df_rank["player_key"] == ""]["player_name"].dropna().unique()
    )
    mapped_total = (df_rank["player_key"] != "").sum()

    print(f"\n{'='*57}")
    print(f"  Total registros         : {total}")
    print(f"  Mapeados tras rescate   : {mapped_total}  ({mapped_total/total*100:.1f}%)")
    print(f"  Rescatados en esta fase : {rescued_count}")
    print(f"  Aún sin key             : {len(still_missing)} nombres únicos")
    print(f"{'='*57}")
    if still_missing:
        print(f"\n  Nombres únicos restantes ({len(still_missing)}):")
        print(f"{'─'*57}")
        for n in still_missing:
            print(f"  • {n}")
    print(f"{'='*57}")


if __name__ == "__main__":
    main()
