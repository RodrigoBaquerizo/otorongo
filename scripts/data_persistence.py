import streamlit as st
import pandas as pd
import numpy as np

def norm(val):
    if pd.isna(val) or val is None: return ""
    return str(val).lower().replace(".", "").replace(" ", "").strip()

def apply_edits(state_key, df_display, table_name, manager):
    if state_key not in st.session_state:
        return False
        
    editor_state = st.session_state[state_key]
    edited_rows = editor_state.get("edited_rows", {})
    
    if not edited_rows:
        return False
        
    # Read the dataframes via DataManager
    try:
        df_current = manager.load_table(table_name)
    except Exception as e:
        st.error(f"Error al leer {table_name}: {e}")
        return False

    try:
        df_historical = manager.load_table("historical_fixtures")
        # Forzar dtype str para compatibilidad con la lógica de matching
        df_historical = df_historical.astype(str)
    except Exception as e:
        st.error(f"Error al leer histórico: {e}")
        return False

    changes_made_current = False
    changes_made_historical = False

    for idx_str, changes in edited_rows.items():
        idx = int(idx_str)
        
        # Original row details
        row_original = df_display.loc[idx]
        id_partido = row_original["ID Partido"]
        j1 = row_original["Jugador 1"]
        j2 = row_original["Jugador 2"]
        fecha = str(row_original["Fecha"])
        
        # Validar Ganador
        if "Ganador" in changes:
            nuevo_ganador = changes["Ganador"]
            ganador_norm = norm(nuevo_ganador)
            validos = [norm(j1), norm(j2)]
            
            es_valido = False
            if not ganador_norm or ganador_norm == "-":
                es_valido = True
            elif any(estado in ganador_norm for estado in ["retired", "cancelled", "walkover", "retirado", "cancelado"]):
                es_valido = True
            elif ganador_norm in validos:
                es_valido = True
                
            if not es_valido:
                st.error(f"Validación Fallida para el partido {j1} vs {j2}: El ganador '{nuevo_ganador}' no coincide con ningún jugador ni estado válido.")
                return False
                
        # Identificación del Partido en df_current
        has_id = pd.notna(id_partido) and str(id_partido).strip() != "" and str(id_partido) != "nan"
        
        # Validar edición de Fecha sin ID
        if not has_id and "Fecha" in changes:
            st.error(f"Error en {j1} vs {j2}: No se permite editar la Fecha de un partido sin ID Partido.")
            return False

        def match_id(x):
            s = str(x)
            if s.endswith(".0"): return s[:-2]
            return s

        if has_id:
            id_str = str(id_partido)
            if id_str.endswith(".0"):
                id_str = id_str[:-2]
            matched_current = df_current[df_current["ID Partido"].apply(match_id) == id_str]
        else:
            torneo = str(row_original["Torneo"])
            try:
                fecha_norm = pd.to_datetime(row_original["Fecha"]).strftime("%Y-%m-%d")
            except:
                fecha_norm = str(row_original["Fecha"]).split()[0]
                
            fecha_curr = pd.to_datetime(df_current["Fecha"], errors="coerce").dt.strftime("%Y-%m-%d")
            torneo_curr = df_current["Torneo"].apply(norm)
            j1_curr = df_current["Jugador 1"].apply(norm)
            j2_curr = df_current["Jugador 2"].apply(norm)
            
            matched_current = df_current[
                (fecha_curr == fecha_norm) & 
                (torneo_curr == norm(torneo)) &
                (j1_curr == norm(j1)) & 
                (j2_curr == norm(j2))
            ]

        if len(matched_current) == 0:
            st.warning(f"No se encontró el partido {j1} vs {j2} en el archivo actual. No se pudo guardar este cambio.")
            continue

        # Aplicar cambios al Dataframe Anual (current)
        for match_idx in matched_current.index:
            for col, new_val in changes.items():
                df_current.loc[match_idx, col] = new_val
            changes_made_current = True
            
        # Si se editó Ganador o Fecha, aplicar al Histórico
        if "Ganador" in changes or "Fecha" in changes:
            if has_id:
                matched_historical = df_historical[df_historical["ID Partido"].apply(match_id) == id_str]
            else:
                fecha_historico = pd.to_datetime(df_historical["Fecha"], errors="coerce").dt.strftime("%Y-%m-%d")
                torneo_historico = df_historical["Torneo"].apply(norm)
                j1_historico = df_historical["Jugador 1"].apply(norm)
                j2_historico = df_historical["Jugador 2"].apply(norm)
                
                matched_historical = df_historical[
                    (fecha_historico == fecha_norm) & 
                    (torneo_historico == norm(torneo)) &
                    (j1_historico == norm(j1)) & 
                    (j2_historico == norm(j2))
                ]
                
            if len(matched_historical) == 0:
                st.warning(f"No se encontró el partido {j1} vs {j2} en el archivo histórico. Solo se actualizó el CSV anual.")
            else:
                for match_idx in matched_historical.index:
                    if "Ganador" in changes:
                        df_historical.loc[match_idx, "Ganador"] = changes["Ganador"]
                    if "Fecha" in changes:
                        df_historical.loc[match_idx, "Fecha"] = changes["Fecha"]
                    changes_made_historical = True

    try:
        if changes_made_current:
            manager.save_table(table_name, df_current)
        if changes_made_historical:
            manager.save_table("historical_fixtures", df_historical)
            
        if changes_made_current or changes_made_historical:
            st.success("Cambios guardados con éxito.")
            return True
            
    except Exception as e:
        st.error(f"Error al guardar datos: {e}")
        return False
        
    return False
