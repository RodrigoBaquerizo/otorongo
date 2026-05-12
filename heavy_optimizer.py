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

def preprocess_data(file_path):
    print(f"--- Cargando datos: {file_path} ---")
    df = pd.read_csv(file_path)
    
    # Solo partidos con ganador registrado
    df_valid = df[df["Ganador"].notna() & (df["Ganador"] != "-") & (df["Ganador"] != "")]
    total_partidos = len(df_valid)
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

def evaluate_combination(data, w, m_odds, m_atp, m_prob):
    # w is (5,) weights summing to 1.0 (0-100 transformed to 0-1)
    f1 = np.dot(data["C1"], w) * 100.0
    f2 = np.dot(data["C2"], w) * 100.0
    
    # Bet flags
    b1 = (f1 >= m_prob) & (data["odds1"] >= m_odds) & (data["pts_sum"] >= m_atp)
    b2 = (f2 >= m_prob) & (data["odds2"] >= m_odds) & (data["pts_sum"] >= m_atp)
    
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
        "total_partidos": data["total_partidos"]
    }

def worker(data, iterations, seed, pipe):
    np.random.seed(seed)
    local_best = []
    
    for _ in range(iterations):
        while True:
            # Generate weights summing to 100 with 0.5 step
            cuts = np.sort(np.random.choice(np.arange(0, 201), 4, replace=True))
            w_int = np.zeros(5, dtype=int)
            w_int[0] = cuts[0] # H2H
            w_int[1] = cuts[1] - cuts[0]
            w_int[2] = cuts[2] - cuts[1]
            w_int[3] = cuts[3] - cuts[2] # Ranking
            w_int[4] = 200 - cuts[3]
            
            # Constraints: Ranking > 29% (58 * 0.5) AND H2H <= 10% (20 * 0.5)
            if w_int[3] > 58 and w_int[0] <= 20:
                break
                
        w = w_int * 0.5 / 100.0 # to 0-1 scale
        
        # Thresholds (Updated to User Constraints)
        m_odds = np.round(np.random.uniform(1.12, 1.30), 2)
        m_atp = int(np.random.choice(np.arange(250, 541, 10)))
        m_prob = float(np.random.randint(60, 86))
        
        res = evaluate_combination(data, w, m_odds, m_atp, m_prob)
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
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: No existe el archivo {args.csv}")
        # Intentar con ATP si Challenger falla
        if "Challenger" in args.csv:
            args.csv = "data/ATP Tour 2026 Matches.csv"
            if not os.path.exists(args.csv):
                print("Error: Tampoco existe el archivo ATP.")
                return

    data = preprocess_data(args.csv)
    
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
    print("="*80)
    
    # PRECISION CHALLENGE: Win Rate >= 85.0, Bets >= 90
    qualifying = [
        r for r in global_results 
        if r["win_rate"] >= 85.0 
        and r.get("bets", 0) >= 90
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
                "Rank": f"{r['w'][3]:.1f}%",
                "Ultra": f"{r['w'][4]:.1f}%",
                "Odds": r["m_odds"],
                "ATP": r["m_atp"],
                "Prob": f"{r['m_prob']:.0f}%"
            })
        print(pd.DataFrame(report_data).to_string(index=False))

    # Lista 1: Mayor Porcentaje de Acierto
    top_wr = sorted(qualifying if qualifying else global_results, key=lambda x: x["win_rate"], reverse=True)
    print_report("LISTA 1: MAYOR PORCENTAJE DE ACIERTO (WIN RATE)", top_wr)

    # Lista 2: Mayor ROI
    top_roi = sorted(qualifying if qualifying else global_results, key=lambda x: x["roi"], reverse=True)
    print_report("LISTA 2: MAYOR ROI", top_roi)

    if not qualifying:
        print("\n[NOTA] No se encontraron combinaciones que cumplan estrictamente WR >= 85% y Bets >= 90. Los reportes arriba muestran los mejores candidatos generales.")
    
    return

if __name__ == "__main__":
    main()
