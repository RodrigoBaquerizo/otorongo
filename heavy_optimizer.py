import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import argparse
import os
import sys

# --- Constantes y Escala ---
MONTOS_DATA = [
    (0.50, 0.53, 20.0), (0.53, 0.56, 26.4), (0.56, 0.59, 36.0), (0.59, 0.62, 60.0),
    (0.62, 0.65, 100.0), (0.65, 0.68, 100.0), (0.68, 0.71, 100.0), (0.71, 0.74, 100.0),
    (0.74, 0.77, 100.0), (0.77, 0.80, 100.0), (0.80, 0.83, 100.0), (0.83, 0.86, 100.0),
    (0.86, 0.89, 100.0), (0.89, 0.92, 100.0), (0.92, 1.01, 100.0),
]

# Convertimos la escala a arrays de NumPy para búsqueda rápida
MONTOS_MIN = np.array([m[0] for m in MONTOS_DATA])
MONTOS_MAX = np.array([m[1] for m in MONTOS_DATA])
MONTOS_VAL = np.array([m[2] for m in MONTOS_DATA])

def norm_name(s):
    return str(s).lower().replace(".", "").replace(" ", "").strip()

def preprocess_data(file_path, surface=None):
    print(f"--- Cargando datos: {file_path} ---")
    df = pd.read_csv(file_path)
    
    # Filtrado por superficie (si se especifica)
    if surface:
        df = df[df["Superficie"].str.lower() == surface.lower()]
        print(f"--- Filtrando por superficie: {surface} ---")
    
    # Solo partidos con ganador registrado
    df_valid = df[df["Ganador"].notna() & (df["Ganador"] != "-") & (df["Ganador"] != "")]
    total_partidos = len(df_valid)
    
    if total_partidos < 60:
        print(f"⚠️ Muestra insuficiente para optimización fiable ({total_partidos} partidos). Se requiere un mínimo de 60.")
        sys.exit(1)
        
    df = df_valid.copy()
    
    def to_float(series):
        return pd.to_numeric(series.astype(str).str.replace("%", "").str.replace(",", "."), errors='coerce').fillna(0)

    # Performance
    h1 = to_float(df.get("J1 H2H %", "0%")) / 100.0
    h2 = to_float(df.get("J2 H2H %", "0%")) / 100.0
    
    r1 = to_float(df["J1 Rend. Reciente"])
    r2 = to_float(df["J2 Rend. Reciente"])
    s1 = to_float(df["J1 Rend. Superficie"])
    s2 = to_float(df["J2 Rend. Superficie"])
    u1 = to_float(df["Rend. Ultra reciente J1"])
    u2 = to_float(df["Rend. Ultra reciente J2"])
    p1 = to_float(df.get("J1 Puntos ATP", 0))
    p2 = to_float(df.get("J2 Puntos ATP", 0))
    
    # Cuotas
    o1 = to_float(df["Cuota J1"])
    o2 = to_float(df["Cuota J2"])
    
    # Filtro: Ignorar si no hay cuotas
    mask_odds = (o1 > 0) & (o2 > 0)
    df = df[mask_odds].reset_index(drop=True)
    
    # Re-extracting after filtering
    h1 = h1[mask_odds].values
    h2 = h2[mask_odds].values
    r1, r2 = r1[mask_odds].values, r2[mask_odds].values
    s1, s2 = s1[mask_odds].values, s2[mask_odds].values
    u1, u2 = u1[mask_odds].values, u2[mask_odds].values
    p1, p2 = p1[mask_odds].values, p2[mask_odds].values
    o1, o2 = o1[mask_odds].values, o2[mask_odds].values
    
    # Componentes relativos (P1 / (P1+P2))
    def rel(v1, v2):
        sums = v1 + v2
        return np.where(sums > 0, v1 / sums, 0)
    
    # El H2H ya es un porcentaje individual (J1 H2H % y J2 H2H %), no se normaliza entre ellos en el streamlit_app
    # pero el usuario dice "deben sumar 100%" para los componentes.
    # En streamlit_app.py: f1 = (h1/100 * w1) + get_f_vec(r1, r2, w2) + ...
    # Así que el h1 ya viene como 0-1.
    
    # J1 Components
    C1 = np.stack([
        h1, 
        rel(r1, r2), 
        rel(s1, s2), 
        rel(p1, p2), 
        rel(u1, u2)
    ], axis=1) # (N, 5)
    
    # J2 Components
    C2 = np.stack([
        h2, 
        rel(r2, r1), 
        rel(s2, s1), 
        rel(p2, p1), 
        rel(u2, u1)
    ], axis=1) # (N, 5)
    
    # Resultado real
    win_norm = df["Ganador"].apply(norm_name).values
    p1_norm = df["Jugador 1"].apply(norm_name).values
    p2_norm = df["Jugador 2"].apply(norm_name).values
    
    is_p1_winner = (win_norm == p1_norm)
    is_p2_winner = (win_norm == p2_norm)
    
    data_pack = {
        "C1": C1, "C2": C2,
        "odds1": o1, "odds2": o2,
        "pts_sum": p1 + p2,
        "has_h2h": (h1 > 0) | (h2 > 0),
        "is_p1_win": is_p1_winner,
        "is_p2_win": is_p2_winner,
        "N": len(df),
        "total_partidos": total_partidos
    }
    print(f"--- Partidos válidos (con ganador): {total_partidos} ---")
    print(f"--- Partidos con cuotas: {len(df)} ---")
    print(f"--- Filtro frecuencia (1.5%): {total_partidos * 0.015:.1f} apuestas mínimas ---")
    
    return data_pack

def get_bet_amount(scores):
    # Vectorized lookup in MONTOS_DATA
    # scores is (N,) 0-1 values
    idx = np.searchsorted(MONTOS_MIN, scores, side='right') - 1
    # Check bounds
    idx = np.clip(idx, 0, len(MONTOS_VAL) - 1)
    # Validate Max
    valid = (scores >= MONTOS_MIN[idx]) & (scores < MONTOS_MAX[idx])
    return np.where(valid, MONTOS_VAL[idx], 0.0)

def evaluate_combination(data, w, m_odds, m_atp, m_prob, m_prob_no_h2h):
    # w is (5,) weights summing to 1.0 (0-100 transformed to 0-1)
    f1 = np.dot(data["C1"], w) * 100.0
    f2 = np.dot(data["C2"], w) * 100.0
    
    # Effective prob threshold per match
    thresh = np.where(data["has_h2h"], m_prob, m_prob_no_h2h)
    
    # Bet flags
    b1 = (f1 >= thresh) & (data["odds1"] >= m_odds) & (data["pts_sum"] >= m_atp)
    b2 = (f2 >= thresh) & (data["odds2"] >= m_odds) & (data["pts_sum"] >= m_atp)
    
    # Resolve conflicts (pick highest score if both)
    # Actually, if both are true, we only bet on one.
    mask_j1 = b1 & (~b2 | (f1 >= f2))
    mask_j2 = b2 & (~mask_j1)
    
    # Final bet properties
    is_bet = mask_j1 | mask_j2
    if not np.any(is_bet):
        return None
    
    chosen_f = np.where(mask_j1, f1, f2)
    chosen_odds = np.where(mask_j1, data["odds1"], data["odds2"])
    chosen_won = np.where(mask_j1, data["is_p1_win"], data["is_p2_win"])
    
    # Bet amounts
    amounts = get_bet_amount(chosen_f / 100.0)
    amounts = np.where(is_bet, amounts, 0.0)
    
    # Filter bets with amount 0 (out of bounds)
    active = amounts > 0
    if not np.any(active):
        return None
    
    # Metrics
    t_amt = np.sum(amounts[active])
    # PnL: if won (odds-1)*amt, if lost -amt
    pnl = np.where(chosen_won, (chosen_odds - 1) * amounts, -amounts)
    t_pnl = np.sum(pnl[active])
    
    bets_count = np.sum(active)
    wins_count = np.sum((pnl > 0) & active)
    loss_count = np.sum((pnl < 0) & active)
    
    roi = (t_pnl / t_amt * 100) if t_amt > 0 else -100
    wr = (wins_count / (wins_count + loss_count) * 100) if (wins_count + loss_count) > 0 else 0
    
    return {
        "balance": t_pnl,
        "roi": roi,
        "win_rate": wr,
        "bets": bets_count,
        "w": w * 100.0,
        "m_odds": m_odds,
        "m_atp": m_atp,
        "m_prob": m_prob,
        "m_prob_no_h2h": m_prob_no_h2h,
        "total_partidos": data["total_partidos"]
    }

def worker(data, iterations, seed, pipe):
    np.random.seed(seed)
    local_best = []
    
    for _ in range(iterations):
        while True:
            # Constraints (Decimal precision - Challenger Hard Optimization):
            # H2H: 1-14%
            # Reciente: 2-25%
            # Superficie: 10-40%
            # Ranking: 35-60%
            # Ultra: 1-15%
            w = np.zeros(5)
            w[0] = np.random.uniform(1.0, 14.1) # H2H
            w[1] = np.random.uniform(2.0, 25.1) # Reciente
            w[2] = np.random.uniform(10.0, 40.1) # Superficie
            w[3] = np.random.uniform(35.0, 60.1) # Ranking
            w[4] = np.random.uniform(1.0, 15.1) # Ultra
            
            # Re-normalize to 100 if close to it, or just use rejection sampling
            if np.sum(w) > 90 and np.sum(w) < 110:
                w = (w / np.sum(w)) * 100.0
                # Check if still in bounds after normalization
                if (1.0 <= w[0] <= 14.0 and 
                    2.0 <= w[1] <= 25.0 and 
                    10.0 <= w[2] <= 40.0 and 
                    35.0 <= w[3] <= 60.0 and 
                    1.0 <= w[4] <= 15.0):
                    break
                
        w_final = w / 100.0 # to 0-1 scale
        
        # Thresholds (Constrained for Specific Search)
        m_odds = 1.17 # Fixed
        m_atp = 500   # Fixed
        m_prob = float(np.random.randint(60, 86))
        m_prob_no_h2h = float(np.random.randint(58, 86))
        
        res = evaluate_combination(data, w_final, m_odds, m_atp, m_prob, m_prob_no_h2h)
        if res:
            local_best.append(res)
            if len(local_best) > 100:
                # Sort by balance for final, but keep local best for "potential"
                local_best = sorted(local_best, key=lambda x: x["balance"], reverse=True)[:50]
                
    pipe.send(local_best)
    pipe.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="data/Challenger Tour Matches.csv")
    parser.add_argument("--minutes", type=int, default=10)
    parser.add_argument("--surface", type=str, default=None, help="Filtrar por superficie: Hard, Clay, Grass")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: No existe el archivo {args.csv}")
        # Intentar con ATP si Challenger falla
        if "Challenger" in args.csv:
            args.csv = "data/ATP Tour 2026 Matches.csv"
            if not os.path.exists(args.csv):
                print("Error: Tampoco existe el archivo ATP.")
                return

    data = preprocess_data(args.csv, args.surface)
    
    num_cpus = mp.cpu_count()
    print(f"--- Iniciando optimización en {num_cpus} núcleos por {args.minutes} minutos ---")
    
    # Estimación de iteraciones por lote
    # 1 millón de trials tarda ~30s en un CPU (?)
    # Vamos a lanzar lotes dinámicos
    start_time = time.time()
    end_time = start_time + (args.minutes * 60)
    
    global_results = []
    total_trials = 0
    
    while time.time() < end_time:
        batch_size = 50000
        pipes = []
        processes = []
        
        for i in range(num_cpus):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=worker, args=(data, batch_size, np.random.randint(0, 10**6), child_conn))
            p.start()
            processes.append(p)
            pipes.append(parent_conn)
            
        for i, p in enumerate(processes):
            res_list = pipes[i].recv()
            global_results.extend(res_list)
            p.join()
            
        total_trials += batch_size * num_cpus
        elapsed = time.time() - start_time
        remaining = end_time - time.time()
        
        # Limpiar resultados globales periódicamente (aumentado para mayor diversidad en los dos reportes)
        global_results = sorted(global_results, key=lambda x: x["balance"], reverse=True)[:1000]
        
        print(f"Trial {total_trials:,} | Tiempo: {int(elapsed)}s | Restante: {int(remaining)}s | Mejores ROI: {[round(r['roi'],1) for r in global_results[:3]]}")
        
        if remaining < 10: # No empezar nuevo lote si queda poco
            break

    # --- Reporte Final ---
    print("\n" + "="*80)
    print(f"OPTIMIZACIÓN COMPLETADA - TOP 20 COMBINACIONES")
    if args.surface:
        print(f"SUPERFICIE: {args.surface.upper()}")
    else:
        print(f"SUPERFICIE: GLOBAL")
    print("="*80)
    
    # CHALLENGER HIGH PRECISION: Win Rate >= 86.0, Bets >= 50
    qualifying = [
        r for r in global_results 
        if r["win_rate"] >= 86.0 
        and r.get("bets", 0) >= 50
    ]
    
    def print_report(title, results):
        print(f"\n--- {title} ---")
        if not results:
            print("No se encontraron resultados que cumplan con los filtros.")
            return
            
        report_data = []
        for i, r in enumerate(results[:5]):
            report_data.append({
                "Rank": i+1,
                "ROI %": f"{r['roi']:.2f}%",
                "Win Rate %": f"{r['win_rate']:.2f}%",
                "Bets": r["bets"],
                "Balance €": f"{r['balance']:,.2f}",
                "H2H": f"{r['w'][0]:.1f}%",
                "Rec": f"{r['w'][1]:.1f}%",
                "Surf": f"{r['w'][2]:.1f}%",
                "Rank-W": f"{r['w'][3]:.1f}%",
                "Ultra": f"{r['w'][4]:.1f}%",
                "Odds": r["m_odds"],
                "ATP": r["m_atp"],
                "Prob": f"{r['m_prob']:.0f}%",
                "P-NoH2H": f"{r['m_prob_no_h2h']:.0f}%"
            })
        print(pd.DataFrame(report_data).to_string(index=False))

    # Lista 1: Mayor Porcentaje de Acierto
    top_wr = sorted(qualifying if qualifying else global_results, key=lambda x: x["win_rate"], reverse=True)
    print_report("LISTA 1: MAYOR PORCENTAJE DE ACIERTO (WIN RATE)", top_wr)

    # Lista 2: Mayor ROI
    top_roi = sorted(qualifying if qualifying else global_results, key=lambda x: x["roi"], reverse=True)
    print_report("LISTA 2: MAYOR ROI", top_roi)

    if not qualifying:
        print("\n[NOTA] No se encontraron combinaciones que cumplan estrictamente WR >= 86% y Bets >= 50.")
    
    return

if __name__ == "__main__":
    main()
