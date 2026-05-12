import os
import sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_manager import DataManager

def main():
    manager = DataManager()
    
    # Forzar credenciales desde el .env local para inspeccionar la BD remota
    manager.supabase_url = os.getenv("SUPABASE_URL")
    manager.supabase_key = os.getenv("SUPABASE_KEY")
    
    if not manager.supabase_url or not manager.supabase_key:
        print("No se encontraron credenciales de Supabase en .env")
        return
        
    client = manager._get_client()
    
    print("--- INSPECCIONANDO TABLA 'tournaments' EN SUPABASE ---")
    try:
        response = client.table("tournaments").select("*").limit(5).execute()
        if response.data:
            df = pd.DataFrame(response.data)
            print("\nEstructura de la tabla (columnas y tipos inferidos):")
            print(df.dtypes)
            print("\nPrimeros registros:")
            print(df)
        else:
            print("La tabla 'tournaments' existe pero esta VACIA.")
            
    except Exception as e:
        print(f"Error al leer la tabla: {e}")

if __name__ == "__main__":
    main()
