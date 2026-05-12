"""
Script para generar el CSV histórico de puntos ATP (2024 en adelante, Top 1000).

Fuente de datos: JeffSackmann/tennis_atp (GitHub)
  - atp_rankings_current.csv: rankings semanales desde 2024
  - atp_players.csv: mapeo de player_id → nombre del jugador

Output: data/atp_rankings_historical.csv
  Columnas: ranking_date, ranking, player_name, player_id, points
"""

import pandas as pd
import requests
import io
import sys

# ─── Configuración ───────────────────────────────────────────────────────────
BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
RANKINGS_URL = f"{BASE_URL}/atp_rankings_current.csv"
PLAYERS_URL  = f"{BASE_URL}/atp_players.csv"
OUTPUT_PATH  = "data/atp_rankings_historical.csv"

START_DATE = 20240101   # Incluir desde 2024-01-01
MAX_RANK   = 1000       # Solo Top 1000


def download_csv(url: str, name: str) -> pd.DataFrame:
    """Descarga un CSV desde una URL y lo devuelve como DataFrame."""
    print(f"  Descargando {name}... ", end="", flush=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    print(f"✓  ({len(df):,} filas)")
    return df


def main():
    print("=" * 60)
    print("  Generando CSV histórico de rankings ATP")
    print("=" * 60)

    # 1. Descargar archivos
    print("\n[1/4] Descargando datos de GitHub...")
    rankings = download_csv(RANKINGS_URL, "atp_rankings_current.csv")
    players  = download_csv(PLAYERS_URL,  "atp_players.csv")

    # 2. Filtrar rankings
    print("\n[2/4] Filtrando datos...")
    rankings["ranking_date"] = rankings["ranking_date"].astype(int)
    rankings["rank"]         = rankings["rank"].astype(int)
    rankings["points"]       = pd.to_numeric(rankings["points"], errors="coerce").fillna(0).astype(int)

    before = len(rankings)
    rankings = rankings[
        (rankings["ranking_date"] >= START_DATE) &
        (rankings["rank"] <= MAX_RANK)
    ].copy()
    print(f"  Filas antes del filtro: {before:,}")
    print(f"  Filas después del filtro: {len(rankings):,}")

    # 3. Unir con nombres de jugadores
    print("\n[3/4] Uniendo con nombres de jugadores...")
    players["player_name"] = players["name_first"].str.strip() + " " + players["name_last"].str.strip()
    players = players[["player_id", "player_name"]].rename(columns={"player_id": "player"})

    merged = rankings.merge(players, on="player", how="left")

    sin_nombre = merged["player_name"].isna().sum()
    if sin_nombre > 0:
        print(f"  ⚠️  {sin_nombre} filas sin nombre de jugador (se mantendrán con player_id como nombre)")
        merged["player_name"] = merged["player_name"].fillna("ID_" + merged["player"].astype(str))

    # 4. Formatear y guardar CSV
    print("\n[4/4] Generando CSV final...")
    result = merged[["ranking_date", "rank", "player_name", "player", "points"]].copy()
    result.columns = ["ranking_date", "ranking", "player_name", "player_id", "points"]

    # Convertir ranking_date de YYYYMMDD a YYYY-MM-DD
    result["ranking_date"] = pd.to_datetime(result["ranking_date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")

    # Ordenar por fecha y ranking
    result = result.sort_values(["ranking_date", "ranking"]).reset_index(drop=True)

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"  ✓ Guardado en: {OUTPUT_PATH}")

    # ─── Resumen de validación ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESUMEN DE VALIDACIÓN")
    print("=" * 60)

    semanas = result["ranking_date"].nunique()
    fecha_min = result["ranking_date"].min()
    fecha_max = result["ranking_date"].max()
    jugadores = result["player_id"].nunique()

    print(f"  Fechas: {fecha_min} → {fecha_max}")
    print(f"  Semanas distintas: {semanas}")
    print(f"  Jugadores únicos: {jugadores:,}")
    print(f"  Total filas: {len(result):,}")

    # Verificar semana más reciente
    ultima_semana = result[result["ranking_date"] == fecha_max].head(5)
    print(f"\n  Top 5 de la semana más reciente ({fecha_max}):")
    print(ultima_semana[["ranking", "player_name", "points"]].to_string(index=False))

    # Verificar datos conocidos: Alcaraz #1 con 13,550 pts en 2026-03-16
    check = result[(result["ranking_date"] == "2026-03-16") & (result["ranking"] == 1)]
    if not check.empty:
        p = check.iloc[0]
        ok = "✓" if "Alcaraz" in p["player_name"] and p["points"] == 13550 else "✗"
        print(f"\n  Validación 2026-03-16 Rank #1: {ok} {p['player_name']} - {p['points']} pts")
    else:
        print("\n  ⚠️  No se encontró la semana 2026-03-16 para validación (puede ser que aún no esté en el repo)")

    print("\n  ✅ CSV generado correctamente.")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as e:
        print(f"\n❌ Error de red: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}", file=sys.stderr)
        raise
