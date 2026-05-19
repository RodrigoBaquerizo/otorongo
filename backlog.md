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

### [x] 🛡️ Blindaje por ID de Torneo (Eliminar Filtraciones ATP/CHA)
> **Nota**: Se implementó el uso de `Tournament Key` como identificador primario en `refresh_data.py` y en los archivos maestros (`ATP Tour 2026 Matches.csv`, `Challenger Tour Matches.csv` y el histórico maestro). Esto elimina la colisión de datos entre circuitos que comparten nombres de torneos (ej: Madrid), garantizando que la sincronización con el histórico sea exacta.

## 1️⃣7️⃣ Integridad de Datos y Blindaje de API (Completado)

### [x] 🛡️ Blindaje de API contra errores de suscripción
> **Nota**: Se modificó `scripts/tenis_api.py` para validar estrictamente que `success == 1` antes de procesar o guardar datos. Se implementó una gestión de errores que captura códigos como el 1006 (suscripción vencida), lanzando excepciones en lugar de permitir que mensajes de error de la API sobrescriban y corrompan los archivos CSV locales.

### [x] 🛠️ Reparación de Base de Datos de Torneos
> **Nota**: Se restauró la integridad del archivo `data/tournaments.csv`, eliminando registros corruptos y restableciendo las cabeceras correctas. Se aseguró la persistencia del mapeo de superficies histórico (incluyendo Madrid Key 2004), permitiendo que el sistema vuelva a identificar superficies automáticamente.

### [x] ⚡ Optimización de Persistencia y Sincronización de Ganadores
> **Nota**: Se refactorizó `scripts/refresh_data.py` para garantizar la persistencia del histórico maestro (`historical_fixtures`). El sistema ahora realiza una sincronización forzada de ganadores en cada refresh, utilizando una ventana de solape de 3 días que asegura que los resultados de partidos recientemente finalizados se actualicen en los archivos de circuito, incluso si no hay fixtures nuevos.

---

## 1️⃣8️⃣ Investigación de Fallo en Refresh Challenger (Pendiente)

### [ ] 🔍 Diagnóstico de Excepción Silenciosa en Circuito Challenger
- **Problema**: El botón "Refresh Challenger" no responde ni abre el diálogo de superficies en la web, a diferencia del modo ATP. Se sospecha de una excepción técnica (Timeout, Rate Limit o Corrupción de Archivo) debido al alto volumen de partidos (~140 diarios) o un fallo en el guardado del historial.
- **Acción**: 
    - Ejecutar el refresh mediante CLI para capturar el traceback exacto.
    - Optimizar el manejo de errores en `streamlit_app.py` para que las excepciones de `refresh_data.py` se muestren siempre en la UI.
    - Verificar la integridad estructural de `Challenger Tour Matches.csv` tras los últimos recálculos.

## 1️⃣9️⃣ Optimización de Umbrales para Partidos sin H2H (Pendiente)

### [x] Fase 1: Implementación en UI y Motor de Análisis (App)
- **Acción**: Añadir el parámetro `min_prob_no_h2h` en `streamlit_app.py`, la configuración JSON y la UI para permitir un umbral diferenciado en partidos sin historial.
- **Detalle**: Implementar la lógica de "Doble Gatillo" en la función `get_stake` de la App. (Completado)

### [ ] Fase 2: Integración en el Optimizador
- **Acción**: Actualizar `heavy_optimizer.py` para que el nuevo umbral sea parte del espacio de búsqueda y se optimice automáticamente para maximizar el ROI.

### [x] Pestaña de 'ATP Análisis'
> **Nota**: Se añadió la nueva pestaña "ATP Análisis" al final de la navegación. El contenido se carga de forma diferida (lazy loading) dentro de su bloque `with tab5:` para mantener la performance. Incluye un subheader y un contenedor para el gráfico de evolución de rendimiento.

## 2️⃣1️⃣ Módulo de Análisis Avanzado (ATP Análisis)

### [x] 21.1 Gráfico: Evolución de Rendimiento
> **Nota**: Se implementó el motor de visualización Plotly en `tab5`. Lee el DataFrame calculado de `st.session_state["atp_computed_df"]`, que se actualiza automáticamente al cambiar parámetros en "ATP data". Incluye agrupación Día/Semana/Mes con `resample()`, doble eje Y para 2 métricas simultáneas, balance acumulado opcional, y tooltip enriquecido con rango de fechas, n° apuestas, Balance, ROI y Acierto. Mini-resumen de KPIs debajo del gráfico.

### [x] 21.2 Ficha de Análisis de Jugador (Mirror Mode)
> **Nota**: Se implementó la sección de análisis de jugador en `tab5`. Se agregó un buscador que permite seleccionar a cualquier jugador con apuestas. Muestra un badge de rentabilidad total, y compara el rendimiento (Apuestas, Win Rate, Cuota Media, Balance, ROI) en modo "A Favor" y "En Contra". Se agregaron visualizaciones Plotly (Semáforo de Superficie en barras y Evolución de Profit en línea) y una lista con el historial de los últimos 5 partidos apostados.

### [x] 21.3 Análisis por Superficie (Tactical Traffic Light)
> **Nota**: Se implementó una sección entre la Evolución de rendimiento y la Ficha de Jugador. Incluye un radar chart (`go.Scatterpolar`) mostrando el ROI por superficie (ajustado dinámicamente a rangos negativos), acompañado de una tabla con N° Apuestas, % Acierto, Balance, ROI y Cuota Media. Las superficies con menos de 10 apuestas incluyen una etiqueta visual de `(⚠️ Muestra pequeña)` para contextualizar la significancia estadística.

### [x] 21.4 Análisis de Rangos de Cuotas (Odds Buckets & Edge)
> **Nota**: Se implementó una sección agrupando las apuestas en 5 buckets de cuotas. El gráfico combinado interactivo muestra el volumen de apuestas en barras (color verde/rojo según si el Edge es positivo o negativo) y el Yield real en una línea. Se calculan dinámicamente el Win Rate Real, el Win Rate Esperado (basado en el inverso de la cuota) y el Edge. Incluye avisos de "Muestra pequeña" si un bucket tiene menos de 20 apuestas.

### [x] 21.5 Rendimiento por Categoría de Torneo
> **Nota**: Se implementó una sección de desglose por categoría de torneo (Grand Slam, Masters 1000, Copa Davis, ATP 500/250). Utiliza un diccionario JSON (`data/tournament_categories.json`) que se auto-genera si no existe y clasifica usando búsqueda por substring. Muestra un gráfico de anillos (Donut chart) con el volumen de apuestas y una tabla detallada con el Edge y Yield real por cada categoría.

### [x] 21.6 Análisis de Drawdown, Volatilidad y Supervivencia
> **Nota**: Se implementó la sección "Radiografía de Riesgo" en tab5. Incluye input de bankroll (default 500 €), 4 KPIs (Profit Factor, Racha Máxima, Time to Recovery, Risk of Ruin), gráfico de estalactitas de Drawdown, termómetro de riesgo actual vs. máximo histórico, y un Stress Test que avisa si el drawdown máximo supera el 50% del bankroll declarado.

### [x] 21.7 Curva de Fiabilidad y Auditoría de Calibración
> **Nota**: Se implementó la sección "Auditoría de Calibración" al final de tab5. Calcula el Brier Score usando todos los partidos con resultado definido. Genera buckets de calibración cada 5% (de 50% a 100%), mostrando un scatter plot de burbujas donde el tamaño es proporcional al volumen y el color indica el nivel de calibración (verde ≤5%, amarillo ≤12%, rojo >12% de diferencial). Incluye diagnóstico automático de sesgo optimista o conservador comparando el WR real vs la probabilidad del modelo en el rango cercano al umbral de entrada configurado.

---

## 3️⃣ Gestión de Estrategias y Pesos
### [x] 22. Configuración Multi-Superficie (ATP)
> **Nota**: Se implementó la configuración multi-superficie en la pestaña ATP. Ahora permite definir pesos y umbrales independientes para Hard (Dura) y Clay (Tierra), manteniendo globales las constantes de bankroll. El motor de cálculo `_compute_bets_df` selecciona dinámicamente los parámetros por fila según la superficie del partido (Grass/Indoor usan Hard por defecto). Se incluyó un mecanismo de migración automática en `load_analysis_config_v2` para conservar la configuración previa del usuario en ambas superficies.

---

## 4️⃣ Análisis y Optimización del Circuito Challenger
### [x] 23. Módulo "Challenger Análisis" (Nueva Pestaña)
> **Nota**: Se implementó la pestaña "Challenger Análisis". Se refactorizó todo el código analítico visual de `tab5` hacia una función modular genérica `render_analytics_dashboard(prefix, df, show_categories)`. Esto permite que tanto el circuito ATP como el Challenger utilicen el mismo motor visual (Evolución, Semáforo, Buckets, Riesgo, Calibración) aislando completamente el `st.session_state` mediante prefijos dinámicos. La sección de Categorías de Torneo se ocultó exitosamente para Challenger. La calibración hereda correctamente el umbral base respectivo de cada circuito (Dura).

---

> **Nota para los agentes**: Este es un documento de gestión. No realizar cambios en código hasta recibir la instrucción específica de una tarea.

