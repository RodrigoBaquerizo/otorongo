import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Configurar para ver si st.secrets existe
try:
    import streamlit as st
    st_secrets = st.secrets.to_dict() if hasattr(st, 'secrets') else {}
except Exception:
    st_secrets = {}

load_dotenv(override=True)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_manager import DataManager

def main():
    print("--- DIAGNOSTICO DE PERSISTENCIA ---")
    print(f"API_KEY (os.getenv): {'CONFIGURADA' if os.getenv('API_KEY') else 'FALTANTE'}")
    print(f"API_KEY (st.secrets): {'CONFIGURADA' if 'API_KEY' in st_secrets else 'FALTANTE'}")
    
    manager = DataManager()
    print(f"Modo Produccion (Supabase): {manager.is_production}")
    
    if not manager.is_production:
        print("No hay credenciales de Supabase. Finalizando.")
        return
        
    client = manager._get_client()
    
    # Intento 1: Escribir un registro de prueba mínimo
    test_record = [{
        "ID Partido": "999999999",
        "Torneo": "Test Diag",
        "Fecha": pd.Timestamp("2026-05-01"),
        "Jugador 1": "Test 1",
        "Jugador 2": "Test 2",
        "Cuota J1": 1.5,
        "Ganador": "-"
    }]
    
    df_test = pd.DataFrame(test_record)
    
    print("\n[+] Probando upsert en atp_matches...")
    try:
        df_clean = manager._prepare_df_for_upsert(df_test, "atp_matches")
        data_dict = df_clean.to_dict(orient="records")
        
        import numpy as np
        def _sanitize_record(rec):
            return {k: (None if v is None or (isinstance(v, float) and np.isnan(v))
                        else int(v) if isinstance(v, (np.integer,))
                        else float(v) if isinstance(v, (np.floating,))
                        else str(v) if isinstance(v, (np.str_, np.bytes_))
                        else bool(v) if isinstance(v, (np.bool_,))
                        else str(v) if isinstance(v, pd.Timestamp) # Fix para timestamp!
                        else v) for k, v in rec.items()}
                        
        data_dict = [_sanitize_record(r) for r in data_dict]
        
        response = client.table("atp_matches").upsert(data_dict).execute()
        print(f"Exito! Respuesta: {response}")
        
    except Exception as e:
        print(f"ERROR CAPTURADO durante Upsert:")
        print(f"Tipo de error: {type(e)}")
        print(f"Detalle: {str(e)}")

if __name__ == "__main__":
    main()
