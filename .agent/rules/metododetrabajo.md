---
trigger: always_on
---

El objetivo del proyecto es ser una fuente fiable de estadísticas de partidos de tenis.

El enfoque principal es brindar la información de estadísticas para los partidos que el usuario necesite.

Es importante verificar que los datos mostrados son correctos, ya que sirven para tomar decisiones de negocio que involucran dinero. Por lo que la data es sensible.

Los datos más importantes a mostrar son:
- H2H: El número y porcentaje de victorias que tienen los jugadores de un partido en enfrentamientos directos.
- Recent Performance: Es el porcentaje de victorias entre la cantidad total de partidos terminados por un jugador (solo en formato Singles y no se incluyen partidos con retiros o cancelados).
- Sourface Recent Performance: Es el porcentaje de victorias entre la cantidad de partidos terminados por un jugador en la superficie en la que se jugará el partido (solo en formato Singles y no se incluyen partidos con retiros o cancelados).
- Rendimiento Ultra reciente: Es similar al rendimiento reciente, pero coge solamente los partidos que el jugador tuvo en los últimos 30 días y se calcula con un algoritmo específico.
- ATP Points: Puntos ATP de cada jugador en el momento del partido.

Para el archivo 'ATP 2026 Matches' es muy importante que no sobreescribas la data histórica a menos que tengas mi expresa autorización.