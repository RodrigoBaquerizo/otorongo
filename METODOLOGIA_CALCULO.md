# 📑 Metodología de Cálculo de Rendimientos - Otorongo

Este documento establece las reglas oficiales para el cálculo de los rendimientos de los jugadores en la aplicación Otorongo. Todos los agentes deben seguir estas fórmulas para asegurar la coherencia de los datos.

---

## 📅 1. Periodos de Tiempo
- **Recent Performance**: Últimos **365 días** (1 año) desde la fecha del partido.
- **Surface Recent Performance**: Últimos **365 días** (1 año) desde la fecha del partido.
- **Ultra Recent Performance**: Últimos **30 días** desde la fecha del partido.

---

## 🥎 2. Filtros de Partidos
Para todos los cálculos de rendimiento (excepto H2H si se indica lo contrario), se deben aplicar los siguientes filtros:
1.  **Formato**: Solo partidos de **Singles** (Individuales).
2.  **Estado**: Solo partidos terminados. **EXCLUIR** estrictamente:
    *   Cualquier partido con estado "Retired" (Retirado).
    *   Cualquier partido con estado "Cancelled" (Cancelado).
    *   Cualquier partido sin un ganador claro registrado.
3.  **Deduplicación**: Asegurar que un mismo partido (por `event_key`) no se cuente dos veces.
4.  **Inmutabilidad**: Una vez que los rendimientos se calculan y registran en los archivos de circuito (ATP/Challenger 2026), se consideran **datos históricos inmutables**. No deben ser recalculados ni actualizados en procesos automáticos de rutina para preservar la fidelidad del estado del jugador en esa fecha específica.

---

## 📈 3. Fórmulas de Cálculo

### A. Recent Performance (Rendimiento Reciente)
*   **Fórmula**: `(Victorias / Total de partidos terminados) * 100`
*   **Puntos ATP**: **NO** se consideran los puntos ATP del rival. Es un porcentaje de victorias puro.

### B. Surface Recent Performance (Rendimiento por Superficie)
*   **Fórmula**: `(Victorias en Superficie X / Total de partidos terminados en Superficie X) * 100`
*   **Superficies**: Se deben normalizar a **Hard**, **Clay** o **Grass**.
*   **Puntos ATP**: **NO** se consideran los puntos ATP del rival.

### C. Ultra Recent Performance (Rendimiento Ultra Reciente)
*   **Fórmula**: Algoritmo ponderado de los últimos 30 días.
*   **Puntos ATP**: **SÍ** se consideran. El valor de cada victoria/derrota se escala según los puntos ATP del oponente (usando la tabla `data/Escala ATP - ELO.csv`).
*   **Decaimiento Temporal**: Los partidos más recientes tienen mayor peso. Se aplica un multiplicador de decaimiento lineal:
        *   `Multiplicador = 1.0 - (Días de diferencia - 1) / 100`
        *   Cada día de antigüedad reduce el peso del partido en aproximadamente un 1% (empezando desde el día 2).
*   **Fórmula Final Ultra**: 
        *   `Puntaje Partido = (Signo Victoria/Derrota) * (Valor Escala ELO) * Multiplicador`
        *   `Rendimiento Ultra % = 50% + (0.5 * Promedio de Puntajes de Partidos)`

---

## 👤 4. Normalización de Jugadores
*   Antes de cualquier cálculo, se debe consultar `data/master-normalizacion-players-key.csv` y `data/player_master.json`.
*   Se debe priorizar el `player_key` sobre el nombre para evitar errores por tildes o formatos ("J.M. Cerundolo" vs "Juan Manuel Cerundolo").

---

> **Importante**: Cualquier modificación en estas fórmulas debe ser aprobada y documentada aquí antes de ser implementada en el código.
