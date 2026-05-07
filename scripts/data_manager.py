import os
import pandas as pd
import streamlit as st
import logging

class DataManager:
    def __init__(self):
        self.supabase_url = None
        self.supabase_key = None
        self.client = None
        
        # Detección de entorno: st.secrets en Streamlit Cloud
        try:
            self.supabase_url = st.secrets.get("SUPABASE_URL")
            self.supabase_key = st.secrets.get("SUPABASE_KEY")
        except Exception:
            # Fallback a variables de entorno para testing local si hiciera falta
            self.supabase_url = os.getenv("SUPABASE_URL")
            self.supabase_key = os.getenv("SUPABASE_KEY")

    @property
    def is_production(self):
        return bool(self.supabase_url and self.supabase_key)

    def _get_client(self):
        if not self.client and self.is_production:
            try:
                from supabase import create_client
                self.client = create_client(self.supabase_url, self.supabase_key)
            except ImportError:
                logging.error("Librería 'supabase' no instalada. Fallback a modo local.")
            except Exception as e:
                logging.error(f"Error inicializando cliente Supabase: {e}")
        return self.client

    def _get_csv_path(self, table_name):
        mapping = {
            "atp_matches": "data/ATP Tour 2026 Matches.csv",
            "challenger_matches": "data/Challenger Tour Matches.csv",
            "historical_fixtures": "data/atp_challenger_fixtures_2024_2026.csv",
            "tournaments": "data/tournaments.csv",
            "rankings": "data/atp_rankings_merged.csv"
        }
        return mapping.get(table_name)

    def _generate_metadata_id(self, row):
        """Genera un ID Partido consistente basado en metadatos para registros huérfanos."""
        fecha = str(row.get("Fecha", "")).strip()
        torneo = str(row.get("Torneo", "")).strip().replace(" ", "")
        j1 = str(row.get("Jugador 1", "")).strip().replace(" ", "")
        j2 = str(row.get("Jugador 2", "")).strip().replace(" ", "")
        return f"{fecha}_{torneo}_{j1}_{j2}".lower()

    def _prepare_df_for_upsert(self, df, table_name):
        """Limpia el DataFrame y asegura la clave primaria para evitar duplicados en Supabase."""
        import math
        df_clean = df.copy()
        
        # Reemplazar NaNs por None para serialización JSON a PostgreSQL
        # pd.isna() maneja tanto float('nan') como numpy.nan universalmente
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(
                lambda x: None if pd.isna(x) else x
            )
        
        # Inviolabilidad: Asegurar ID Partido para upserts
        if table_name in ["atp_matches", "challenger_matches", "historical_fixtures"]:
            pk_col = "ID Partido"
            if pk_col in df_clean.columns:
                df_clean[pk_col] = df_clean[pk_col].astype(str)
                df_clean[pk_col] = df_clean[pk_col].replace(["nan", "None", ""], None)
                
                mask = df_clean[pk_col].isnull()
                if mask.any():
                    df_clean.loc[mask, pk_col] = df_clean[mask].apply(self._generate_metadata_id, axis=1)
            else:
                df_clean[pk_col] = df_clean.apply(self._generate_metadata_id, axis=1)
                
        # Asegurar primary keys correctas para otras tablas
        if table_name == "tournaments" and "tournament_key" in df_clean.columns:
            df_clean["tournament_key"] = df_clean["tournament_key"].astype(str).replace(["nan", "None", ""], None)
        if table_name == "rankings" and "player_key" in df_clean.columns:
            df_clean["player_key"] = df_clean["player_key"].astype(str).replace(["nan", "None", ""], None)
            df_clean["date"] = df_clean["date"].astype(str)
            
        return df_clean

    def load_table(self, table_name, dtype=None):
        """
        Lee datos de Supabase (Nube) o CSV (Local). 
        Con Fallback automático a CSV de solo-lectura si falla la conexión en producción.
        """
        if self.is_production:
            try:
                client = self._get_client()
                if client:
                    # Paginación: Supabase limita a 1000 filas por request (configuración del servidor).
                    # Iteramos páginas de 1000 hasta obtener todos los datos.
                    page_size = 1000
                    all_data = []
                    offset = 0
                    while True:
                        response = (client.table(table_name)
                                    .select("*")
                                    .range(offset, offset + page_size - 1)
                                    .execute())
                        if response.data:
                            all_data.extend(response.data)
                            if len(response.data) < page_size:
                                break  # Última página
                            offset += page_size
                        else:
                            break
                    
                    if all_data:
                        df = pd.DataFrame(all_data)
                        logging.info(f"[NUBE] Cargado {table_name}: {len(df)} filas desde Supabase.")
                        return df
                    else:
                        logging.info(f"[NUBE] Supabase retornó 0 filas para {table_name}.")
                        return pd.DataFrame()
            except Exception as e:
                logging.error(f"[NUBE] Fallo de conexión Supabase ({table_name}): {e}. Ejecutando Fallback a CSV...")
        
        # Comportamiento Local o Fallback de Producción
        csv_path = self._get_csv_path(table_name)
        if csv_path and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, dtype=dtype)
                logging.info(f"[LOCAL] Cargado {table_name} desde {csv_path}.")
                return df
            except Exception as e:
                logging.error(f"Error leyendo CSV local {csv_path}: {e}")
        
        return pd.DataFrame()

    def save_table(self, table_name, df):
        """
        Guarda los datos en Supabase (Nube mediante Upsert) o en CSV (Local mediante overwrite).
        """
        if df.empty:
            logging.warning(f"Intento de guardar DataFrame vacío en {table_name}.")
            return False

        if self.is_production:
            try:
                client = self._get_client()
                if client:
                    df_clean = self._prepare_df_for_upsert(df, table_name)
                    data_dict = df_clean.to_dict(orient="records")
                    
                    # Convertir tipos numpy a tipos nativos de Python para serialización JSON
                    import numpy as np
                    def _sanitize_record(rec):
                        return {k: (None if v is None or (isinstance(v, float) and np.isnan(v))
                                    else int(v) if isinstance(v, (np.integer,))
                                    else float(v) if isinstance(v, (np.floating,))
                                    else str(v) if isinstance(v, (np.str_, np.bytes_))
                                    else bool(v) if isinstance(v, (np.bool_,))
                                    else str(v) if isinstance(v, pd.Timestamp)
                                    else v) for k, v in rec.items()}
                    data_dict = [_sanitize_record(r) for r in data_dict]
                    
                    # Deduplicar por PK dentro de cada batch para evitar ON CONFLICT errors
                    pk_map = {
                        "atp_matches": "ID Partido",
                        "challenger_matches": "ID Partido",
                        "historical_fixtures": "ID Partido",
                        "tournaments": "tournament_key",
                    }
                    pk_col = pk_map.get(table_name)
                    if pk_col:
                        seen = set()
                        deduped = []
                        for rec in data_dict:
                            pk_val = rec.get(pk_col)
                            if pk_val not in seen:
                                seen.add(pk_val)
                                deduped.append(rec)
                        data_dict = deduped
                    # Para rankings (PK compuesta), deduplicar por (player_key, date)
                    if table_name == "rankings":
                        seen = set()
                        deduped = []
                        for rec in data_dict:
                            compound = (rec.get("player_key"), rec.get("date"))
                            if compound not in seen:
                                seen.add(compound)
                                deduped.append(rec)
                        data_dict = deduped
                    
                    # Upsert por bloques de 500 para no saturar la API
                    batch_size = 500
                    for i in range(0, len(data_dict), batch_size):
                        batch = data_dict[i:i + batch_size]
                        client.table(table_name).upsert(batch).execute()
                        
                    logging.info(f"[NUBE] Upsert exitoso en {table_name} ({len(data_dict)} filas).")
                    return True
            except Exception as e:
                logging.error(f"[NUBE] Error crítico en upsert para {table_name}: {e}")
                return False
        
        # Comportamiento Local
        csv_path = self._get_csv_path(table_name)
        if csv_path:
            try:
                df.to_csv(csv_path, index=False, encoding="utf-8")
                logging.info(f"[LOCAL] Sobrescrito exitoso en {csv_path}")
                return True
            except Exception as e:
                logging.error(f"[LOCAL] Error escribiendo en CSV {csv_path}: {e}")
                
        return False
