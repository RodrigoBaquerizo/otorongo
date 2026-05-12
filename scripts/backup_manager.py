import os
import shutil
import datetime
import glob

# Configuración de Rutas
# El script se encuentra en scripts/backup_manager.py, por lo que el ROOT es el padre.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
BACKUPS_DIR = os.path.join(ROOT_DIR, "backups")
MAX_BACKUPS = 3

def get_snapshot_name():
    """Genera un nombre de carpeta basado en el timestamp actual."""
    now = datetime.datetime.now()
    return f"snapshot_{now.strftime('%Y-%m-%d_%H%M')}"

def run_backup():
    """Ejecuta la lógica de backup y rotación."""
    
    # 1. Asegurar que el directorio raíz de backups existe
    if not os.path.exists(BACKUPS_DIR):
        os.makedirs(BACKUPS_DIR, exist_ok=True)

    # 2. Lógica de Rotación (Mantener máximo 3 backups)
    # Listamos solo archivos .zip que empiecen con 'snapshot_'
    existing_backups = sorted([
        f for f in os.listdir(BACKUPS_DIR) 
        if f.startswith("snapshot_") and f.endswith(".zip") and os.path.isfile(os.path.join(BACKUPS_DIR, f))
    ])
    
    deleted_backup = "Ninguno"
    if len(existing_backups) >= MAX_BACKUPS:
        oldest = existing_backups[0]
        os.remove(os.path.join(BACKUPS_DIR, oldest))
        deleted_backup = oldest

    # 3. Preparar nueva carpeta de snapshot (temporal para comprimir)
    snapshot_name = get_snapshot_name()
    temp_snapshot_path = os.path.join(BACKUPS_DIR, snapshot_name)
    zip_path = temp_snapshot_path + ".zip"
    
    try:
        os.makedirs(temp_snapshot_path, exist_ok=True)

        # 4. Copiar Archivos Críticos
        items_to_copy = ["streamlit_app.py", "scripts", "styles.css"]
        for item in items_to_copy:
            src = os.path.join(ROOT_DIR, item)
            dst = os.path.join(temp_snapshot_path, item)
            if os.path.exists(src):
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

        for md_file in glob.glob(os.path.join(ROOT_DIR, "*.md")):
            shutil.copy2(md_file, temp_snapshot_path)

        data_src = os.path.join(ROOT_DIR, "data")
        if os.path.exists(data_src):
            data_dst = os.path.join(temp_snapshot_path, "data")
            os.makedirs(data_dst, exist_ok=True)
            for json_file in glob.glob(os.path.join(data_src, "*.json")):
                shutil.copy2(json_file, data_dst)

        # 5. Comprimir y Eliminar Carpeta Temporal
        shutil.make_archive(temp_snapshot_path, 'zip', temp_snapshot_path)
        shutil.rmtree(temp_snapshot_path)

        # 6. Log de finalización exitosa
        print(f"Backup creado: {zip_path} | Backups eliminados: {deleted_backup}")

    except Exception as e:
        print(f"Error durante el backup: {e}")
        # Si falló a mitad de camino, intentamos limpiar la carpeta corrupta
        if os.path.exists(snapshot_path):
            shutil.rmtree(snapshot_path)

if __name__ == "__main__":
    run_backup()
