"""
recalculate_ultra_challenger.py
Recálculo masivo de Rendimiento Ultra Reciente (J1 y J2)
para data/Challenger Tour Matches.csv.

Algoritmo oficial (METODOLOGIA_CALCULO.md §3.C):
  - Ventana: 30 días previos a la fecha del partido.
  - Filtro: Solo Singles terminados (excluye Retirado/Cancelado/sin ganador).
  - ELO del rival: valor de Escala ATP - ELO.csv según sus puntos en esa fecha.
  - Decaimiento: multiplicador = 1.0 - (días_dif - 1) / 100
  - Fórmula: Ultra% = 50 + 0.5 * mean(signo * elo_val * mult)
  - Identificación: prioridad player_key sobre nombre.

Solo modifica: 'Rend. Ultra reciente J1' y 'Rend. Ultra reciente J2'.
"""
import pandas as pd
import os
import sys
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CHA_FILE  = "data/Challenger Tour Matches.csv"
BACKUP    = "data/Challenger Tour Matches_PRE_ULTRA.csv"
HIST_FILE = "data/atp_challenger_fixtures_2024_2026.csv"

sys.path.insert(0, os.getcwd())
try:
    from scripts.refresh_data import calc_ultra_performance_refresh, load_player_master, _load_stats_resources
except ImportError as e:
    logging.error(f"No se pudo importar refresh_data: {e}")
    sys.exit(1)


def main():
    # ── 0. Backup ─────────────────────────────────────────────────────────
    shutil.copy(CHA_FILE, BACKUP)
    logging.info(f"Backup: {BACKUP}")

    # ── 1. Pre-cargar recursos (escala ELO + rankings index en memoria) ───
    logging.info("Pre-cargando escala ELO y rankings en memoria…")
    _load_stats_resources()  # popula _SCALE_DATA y _RANKINGS_INDEX globales

    # ── 2. Cargar histórico completo (fuente de partidos pasados) ─────────
    logging.info("Cargando base histórica…")
    df_hist = pd.read_csv(HIST_FILE, low_memory=False)
    df_hist["Fecha"] = pd.to_datetime(df_hist["Fecha"], errors="coerce")
    logging.info(f"  {len(df_hist)} registros históricos cargados.")

    # ── 3. Maestro de identidades ─────────────────────────────────────────
    master = load_player_master()

    # ── 4. Cargar Challenger ──────────────────────────────────────────────
    df = pd.read_csv(CHA_FILE, dtype={"J1 Key": str, "J2 Key": str}, low_memory=False)
    original_cols = df.columns.tolist()
    assert len(original_cols) == 25, f"ERROR: se esperaban 25 cols, hay {len(original_cols)}"
    total_rows = len(df)
    logging.info(f"Recalculando Ultra para {total_rows} filas de Challenger…")

    updated = 0
    skipped = 0
    values_all = []

    for idx, row in df.iterrows():
        dt_match = pd.to_datetime(row["Fecha"], format="%m/%d/%y", errors="coerce")
        if pd.isna(dt_match):
            skipped += 1
            continue

        p1      = str(row["Jugador 1"]).strip()
        p2      = str(row["Jugador 2"]).strip()
        p1_key  = str(row.get("J1 Key", "")).replace(".0", "").strip()
        p2_key  = str(row.get("J2 Key", "")).replace(".0", "").strip()
        p1_key  = None if p1_key in ("", "nan", "None") else p1_key
        p2_key  = None if p2_key in ("", "nan", "None") else p2_key

        u1 = calc_ultra_performance_refresh(p1, dt_match, df_hist, master, player_key=p1_key)
        u2 = calc_ultra_performance_refresh(p2, dt_match, df_hist, master, player_key=p2_key)

        df.at[idx, "Rend. Ultra reciente J1"] = u1
        df.at[idx, "Rend. Ultra reciente J2"] = u2
        updated += 1

        # Registrar valores numéricos para validación de rango
        for v in [u1, u2]:
            try:
                values_all.append(float(str(v).replace("%", "")))
            except (ValueError, TypeError):
                pass  # "N/D" se ignora

        if (idx + 1) % 500 == 0 or (idx + 1) == total_rows:
            logging.info(f"  Progreso: {idx + 1}/{total_rows}")

    # ── 5. Guardar — solo las 25 columnas originales ──────────────────────
    df = df[original_cols]
    df.to_csv(CHA_FILE, index=False, encoding="utf-8")
    logging.info(f"Guardado: {CHA_FILE}")

    # ── 6. Reporte ────────────────────────────────────────────────────────
    val_min = min(values_all) if values_all else 0
    val_max = max(values_all) if values_all else 0
    in_range = all(0.0 <= v <= 100.0 for v in values_all) if values_all else True

    print(f"\n{'='*58}")
    print(f"  Filas procesadas          : {total_rows}")
    print(f"  Filas actualizadas        : {updated}")
    print(f"  Filas omitidas (sin fecha): {skipped}")
    print(f"  Rango de valores (Ultra%) : {val_min:.1f} – {val_max:.1f}")
    print(f"  Valores en rango [0,100]  : {'✅ SÍ' if in_range else '⚠️ NO'}")
    print(f"{'='*58}")


if __name__ == "__main__":
    main()
