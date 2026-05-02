# 📋 Backlog de Tareas - Proyecto Otorongo

> **📢 Instrucción para Programadores**: Al finalizar una tarea, por favor márcala como `[x]` y deja una pequeña nota en formato de cita (`>`) resumiendo lo implementado.

Este documento contiene el listado de tareas pendientes para la mejora, corrección y expansión de la aplicación **Otorongo**. Se divide en categorías para facilitar la asignación a diferentes agentes de desarrollo.

---

## 🛡️ Reglas de Desarrollo (Inviolabilidad y Rendimiento)

Para asegurar la calidad y fiabilidad de Otorongo, todos los agentes deben seguir estas reglas:
*   **Inviolabilidad de Datos**: Una vez que un registro (partido) se guarda en `ATP Tour 2026 Matches.csv` o `Challenger Tour Matches.csv` con sus métricas calculadas, **NUNCA** debe ser sobrescrito automáticamente. La data representa el estado de forma en el momento exacto del partido.
*   **Eficiencia de Procesos**: La función 'Refresh data' debe ser rápida. Se prohíbe el uso de lógicas de "Backfill" masivo que re-escaneen archivos enteros en cada ejecución. El refresco debe centrarse únicamente en la ventana de tiempo actual (últimos días/próximos días).
*   **Confianza en el Registro**: Debemos confiar en que si un dato se registró, el proceso que lo hizo fue correcto. Las correcciones masivas solo se realizarán bajo petición expresa del usuario.

---

---

## 1️⃣ Correcciones de Integridad y Motor (Completado)

### [x] 🚑 Restauración de Emergencia e Auditoría (Fase 7)
> **Nota**: Se recuperó la lógica de 1400 líneas en `streamlit_app.py`, re-implementando los 5 pesos, corrigiendo el filtro de Puntos ATP mínimos y aplicando la escala conservadora de 15 niveles. La auditoría final arrojó un ROI del 13.97% y Acierto del 87.6%, validando la precisión matemática del sistema sin redondeos ni conteo de retiros.

### [x] 🛠️ Depuración de la función 'Refresh data'
> **Nota**: Se corrigieron los problemas de normalización de Mérida y Cerundolo. Se eliminó el backfill masivo y se implementó una ventana delta de 7 días solo para ganador/cuotas.

### [x] 📏 Estandarización de Fórmulas de Rendimiento
> **Nota**: Todas las funciones ahora usan 365 días (Recent/Surface) y 30 días (Ultra) de forma consistente. Se eliminó la filtración de puntos ATP en rendimientos estándar.

### [x] 🧪 Blindaje de Datos y Sanidad del Sistema (QA)
> **Nota**: Se implementó el suite de pruebas en `/tests` con escáner de player_keys, puntos ATP, lógica de inmutabilidad y validación de formato TSV.

---

## 2️⃣ Nuevas Funcionalidades (Siguiente)

### [x] 🎾 Pestaña 'Challenger Bet'
> **Nota**: Se implementó una nueva pestaña dedicada al circuito Challenger, espejando la funcionalidad de ATP Bet. Se refactorizó el código para usar un componente reutilizable (`render_bet_tab`) que garantiza coherencia en filtros, visualización y exportación TSV para Excel.

---

## 3️⃣ Mejoras en el Motor de Análisis (Completado)

### [x] 🧠 Motor de Optimización de Alto Rendimiento (`heavy_optimizer.py`)
> **Nota**: Se desarrolló un script CLI independiente que utiliza NumPy vectorizado y Multiprocessing para explorar millones de combinaciones en minutos. El motor permite optimizar pesos y umbrales de forma independiente para cada circuito (ATP/Challenger) cumpliendo estrictas restricciones de ROI (15%) y Win Rate (88%).

---

## 4️⃣ Calidad y Procesos (QA) (Completado)

### 🧪 Fase 3: Blindaje de Datos y Sanidad del Sistema (QA)
Crear una red de seguridad proactiva para detectar errores de inconsistencia antes de que lleguen a la UI.
- **Objetivo**: Desarrollar un suite de pruebas (módulo `tests/`) que verifique:
    - **Integridad de Claves (Player Keys)**: Ninguna fila en los archivos de circuito debe tener un `J1 Key` o `J2 Key` vacío o NaN. Esto detectará fallos de normalización al instante.
    - **Consistencia de Puntos ATP**: Validar que el proceso de 'Refresh' no asigne 0 puntos a jugadores que sabemos que están rankeados.
    - **Fórmulas Matemáticas**: Validar rendimientos (Recent, Surface, Ultra) con casos de prueba conocidos.
    - **Inmutabilidad de Datos**: Verificar que una re-ejecución del script NO sobrescriba métricas históricas ya registradas.
    - **Normalización de Superficies**: Asegurar que no existan valores fuera de 'Hard', 'Clay' o 'Grass'.
- **Entregable**: Un script de "Smoke Test" que el usuario o el sistema pueda ejecutar tras cada actualización para obtener un semáforo de salud de la data.

---

## 5️⃣ Tareas de Infraestructura y Mantenimiento (Propuestas por PM)

### 🔍 Auditoría Semanal de Normalización (Nombres/Superficies)
- **Tarea**: Crear un script automático que detecte nombres de jugadores o torneos que no estén en el mapeo de normalización y genere un reporte de "Nuevos elementos detectados".

### 📊 Dashboard de Salud de Datos
- **Tarea**: Una vista oculta o log avanzado en la app que muestre el porcentaje de "N/D" (No Disponibles) en los archivos principales para detectar caídas de la API o errores de procesamiento masivo.

### 🚀 Optimización de Carga Inicial
- **Tarea**: Refactorizar la carga de archivos CSV grandes en `streamlit_app.py` usando caché de Streamlit (`st.cache_data`) de manera más agresiva o migrando a formatos de archivo más rápidos como Parquet si el volumen sigue creciendo.

---

## 6️⃣ Mejoras de Exportación (Completado)

### [x] 📂 Ordenamiento Inteligente en Exportación TSV
> **Nota**: Se implementó la captura de `event_time` en el pipeline de datos (`refresh_data.py`). Los archivos TSV exportados ahora se organizan jerárquicamente por Torneo (A-Z) y luego por Hora (cronológico). La columna 'Hora' se mantiene como un campo técnico oculto en la exportación final para conservar el estándar de 21 columnas.

---

## 7️⃣ Mejoras de Captura y Redundancia (Completado)

### [x] 📈 Optimización de Captura de Cuotas (Redundancia)
> **Nota**: Se implementó un sistema de cascada (fallback) en `get_odds_data` de `tenis_api.py`. El sistema ahora intenta capturar cuotas siguiendo una jerarquía de confianza (Bet365 -> Bwin -> 1xbet -> Betsson -> Sportingbet -> Betcris) y realiza un escaneo universal de todos los proveedores disponibles si los preferidos fallan. Esto garantiza que el campo de cuotas se complete siempre que exista un mercado activo en cualquier casa de apuestas soportada por la API.

---

## 8️⃣ Interactividad y Gestión de Datos (Pendiente)

### [x] ✍️ Tablas de Análisis Interactivas (Edición Manual)
> **Nota**: Se implementó `st.data_editor` con un sistema de persistencia dual en `scripts/data_persistence.py`. El usuario ahora puede editar Cuotas, Ganador y Fecha directamente en la UI. Incluye validación inteligente de nombres de jugadores y sincronización automática entre los archivos de circuito y el histórico maestro.


---

## 9️⃣ Personalización de la Interfaz (Pendiente)

### [x] 👁️ Control Dinámico de Visibilidad de Columnas
> **Nota**: Se implementó un selector multichoice en un expansor sobre las tablas de datos. Permite al usuario ocultar columnas para una vista más limpia. La selección es persistente (se guarda en `analysis_config.json`) y no afecta a las exportaciones TSV ni a los cálculos internos.


---

## 🔟 Misión: Nube y Producción (Pendiente)

### ☁️ Despliegue Híbrido: Otorongo 3.0 en Streamlit Cloud
**Objetivo**: Lograr persistencia infinita en la nube mediante Supabase, manteniendo la velocidad y simplicidad de los CSV en el entorno local del usuario.

#### **Fase 1: Infraestructura y Secretos (Usuario)**
- [ ] **Crear Proyecto en Supabase**: Configurar la organización y el proyecto "Otorongo".
- [ ] **Configurar Secrets**: Añadir `SUPABASE_URL`, `SUPABASE_KEY` y `API_KEY` en el panel de control de Streamlit Cloud.

#### **Fase 2: Arquitectura de Datos Dual (IA)**
- [ ] **Diseño de Esquema**: Crear las tablas equivalentes a los CSV maestros (`atp_matches`, `challenger_matches`, `tournaments`, `rankings`).
- [ ] **Data Manager (`scripts/data_manager.py`)**: Implementar la clase que detecta el entorno (`ST_IS_PROD`) y redirige las peticiones de lectura/escritura a CSV o SQL automáticamente.

#### **Fase 3: Refactorización de Motores (IA)**
- [ ] **Adaptar Refresh**: Modificar `refresh_data.py` para que use el Manager en lugar de `pd.read_csv` directo.
- [ ] **Adaptar Persistencia**: Actualizar `apply_edits` para que los cambios manuales desde la UI lleguen a Supabase en producción.

#### **Fase 4: Sincronización Inteligente (IA)**
- [ ] **Botón "Push to Cloud"**: Crear una utilidad en la UI (visible solo en local) que permita subir los cambios de los CSV a la nube.
- [ ] **Lógica de Comparación**: El sistema debe verificar si el CSV local tiene registros más nuevos o ediciones manuales que no están en Supabase antes de sincronizar.

#### **Fase 5: Despliegue Final (IA)**
- [ ] **requirements.txt**: Consolidar todas las dependencias.
- [ ] **Conexión GitHub**: Vincular el repositorio a Streamlit Cloud y validar el arranque limpio.

---

---

## 1️⃣1️⃣ Estabilización del Core: Sincronización Challenger (Completado)

### [x] 🛠️ Refactorización de `refresh_data.py` para Backfill de Challenger
> **Nota**: El motor de refresco ha sido unificado y optimizado. Se implementó una pre-indexación en memoria (`_build_hist_index`) que reduce la complejidad de búsqueda de O(N²) a O(1), bajando el tiempo de proceso de minutos a segundos. Se estandarizó la ventana de sincronización de 7 días para ambos circuitos.

### [x] ⚡ Gestión Interactiva de Superficies (Opción C)
> **Nota**: Se implementó un flujo de seguridad que detecta torneos nuevos en la API. Si la superficie es desconocida, la App abre un diálogo interactivo (`@st.dialog`) para que el usuario asigne la superficie (Hard/Clay/Grass) antes de proceder, garantizando la integridad de la base de datos. Se añadió un suite de pruebas unitarias en `tests/test_refresh.py` para blindar esta funcionalidad.


---

## 1️⃣2️⃣ Migración a Rankings basados en ID (ID-Centric Rankings)

### [x] 12.1 Backfill del archivo `atp_rankings_merged.csv`
> **Nota**: Se incorporó la columna `player_key` en el histórico de rankings. Se validó que el 100% de los jugadores con más de 30 puntos en la última fecha tienen su ID asignado, garantizando la precisión del cálculo ultra reciente.

### [x] 12.2 Actualización del motor de `refresh_data.py` para persistencia de IDs
> **Nota**: Se modificó la función `_load_stats_resources` para que al descargar nuevos rankings vía API (`get_standings`), capture y persista automáticamente el `player_key`. Esto asegura que el histórico `atp_rankings_merged.csv` mantenga su integridad y precisión en futuras actualizaciones.

---

## 1️⃣3️⃣ Curación de Datos Históricos: Puntos ATP (Challenger)

### [x] 13.1 Reparación masiva de Puntos ATP en `Challenger Tour Matches.csv`
> **Nota**: Se ejecutó el script de curación masiva que restauró los puntos ATP históricos en el archivo de Challenger. Se utilizaron los `player_key` para garantizar que la búsqueda de puntos en `atp_rankings_merged.csv` fuera exacta, eliminando los valores "0.0" y vacíos del historial.

---

## 1️⃣4️⃣ Restauración de Métricas: Rendimiento Ultra Reciente (Challenger) (Completado)

### [x] 14.1 Recálculo masivo de Rendimiento Ultra en `Challenger Tour Matches.csv`
> **Nota**: Se restauró la integridad de la métrica Ultra Reciente en todo el historial de Challenger. El proceso utilizó los puntos ATP corregidos de los rivales para asegurar una valoración ELO precisa en cada fecha, recalculando los ~3000 registros del archivo maestro.

## 1️⃣5️⃣ Mejoras de Interfaz y UX (Completado)

### [x] ⏳ Barra de Progreso en Refresh Data
> **Nota**: Se implementó una barra de progreso visual (`st.progress`) y un contenedor de mensajes dinámicos en `streamlit_app.py`. El motor de `refresh_data.py` ahora soporta callbacks de progreso, permitiendo al usuario ver en tiempo real qué fase del proceso (Descarga API, Cálculos, Sync Histórico) se está ejecutando.

### [x] 📅 Ordenamiento Predeterminado de Tablas
> **Nota**: Se añadió una lógica de ordenamiento por fecha descendente (`ascending=False`) al cargar los DataFrames en la UI. Las tablas de ATP y Challenger ahora muestran los partidos más recientes al inicio por defecto.

---

## 1️⃣6️⃣ Protección de Circuitos y Sincronización (Pendiente)

### [ ] 🛡️ Blindaje por ID de Torneo (Eliminar Filtraciones ATP/CHA)
- **Problema**: Actualmente la sincronización del Paso 3 usa solo el nombre del torneo, lo que causa que partidos de ATP Madrid se filtren en el archivo de Challenger Madrid.
- **Acción**: 
    - Modificar la estructura de `ATP Tour 2026 Matches.csv` y `Challenger Tour Matches.csv` para incluir la columna `Tournament Key`.
    - Actualizar `refresh_data.py` para que la sincronización con el historial (Paso 2.5) utilice el `tournament_key` (único por circuito) en lugar de solo el nombre del torneo.
    - Asegurar que el archivo histórico maestro también capture y valide este ID para evitar colisiones entre Masters 1000 y Challengers con nombres idénticos.

## 1️⃣7️⃣ Integridad de Datos y Blindaje de API (Completado)

### [x] 🛡️ Blindaje de API contra errores de suscripción
> **Nota**: Se modificó `scripts/tenis_api.py` para validar estrictamente que `success == 1` antes de procesar o guardar datos. Se implementó una gestión de errores que captura códigos como el 1006 (suscripción vencida), lanzando excepciones en lugar de permitir que mensajes de error de la API sobrescriban y corrompan los archivos CSV locales.

### [x] 🛠️ Reparación de Base de Datos de Torneos
> **Nota**: Se restauró la integridad del archivo `data/tournaments.csv`, eliminando registros corruptos y restableciendo las cabeceras correctas. Se aseguró la persistencia del mapeo de superficies histórico (incluyendo Madrid Key 2004), permitiendo que el sistema vuelva a identificar superficies automáticamente.

### [x] ⚡ Optimización de Persistencia y Sincronización de Ganadores
> **Nota**: Se refactorizó `scripts/refresh_data.py` para garantizar la persistencia del histórico maestro (`historical_fixtures`). El sistema ahora realiza una sincronización forzada de ganadores en cada refresh, utilizando una ventana de solape de 3 días que asegura que los resultados de partidos recientemente finalizados se actualicen en los archivos de circuito, incluso si no hay fixtures nuevos.

---

> **Nota para los agentes**: Este es un documento de gestión. No realizar cambios en código hasta recibir la instrucción específica de una tarea.
