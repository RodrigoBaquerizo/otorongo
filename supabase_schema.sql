-- SCHEMA FIX: Recrear tablas de partidos con ALL TEXT columns
-- para evitar conflictos de tipos con los datos del CSV.
-- 
-- Ejecutar en: Supabase Dashboard → SQL Editor → New query
-- URL: https://supabase.com/dashboard/project/lvjufittmoflbcvvcqtc/sql/new

-- 1. atp_matches
DROP TABLE IF EXISTS public.atp_matches;
CREATE TABLE public.atp_matches (
    "ID Partido" TEXT PRIMARY KEY,
    "Torneo" TEXT, "Fecha" TEXT, "Superficie" TEXT,
    "Jugador 1" TEXT, "J1 Key" TEXT, "J1 Puntos ATP" TEXT,
    "Jugador 2" TEXT, "J2 Key" TEXT, "J2 Puntos ATP" TEXT,
    "J1 H2H" TEXT, "J1 H2H %" TEXT, "J2 H2H" TEXT, "J2 H2H %" TEXT,
    "J1 Rend. Reciente" TEXT, "J1 Rend. Superficie" TEXT, "Rend. Ultra reciente J1" TEXT,
    "J2 Rend. Reciente" TEXT, "J2 Rend. Superficie" TEXT, "Rend. Ultra reciente J2" TEXT,
    "Cuota J1" TEXT, "Cuota J2" TEXT, "Ganador" TEXT, "Hora" TEXT
);

-- 2. challenger_matches
DROP TABLE IF EXISTS public.challenger_matches;
CREATE TABLE public.challenger_matches (
    "ID Partido" TEXT PRIMARY KEY,
    "Torneo" TEXT, "Fecha" TEXT, "Superficie" TEXT,
    "Jugador 1" TEXT, "J1 Key" TEXT, "J1 Puntos ATP" TEXT,
    "Jugador 2" TEXT, "J2 Key" TEXT, "J2 Puntos ATP" TEXT,
    "J1 H2H" TEXT, "J1 H2H %" TEXT, "J2 H2H" TEXT, "J2 H2H %" TEXT,
    "J1 Rend. Reciente" TEXT, "J1 Rend. Superficie" TEXT, "Rend. Ultra reciente J1" TEXT,
    "J2 Rend. Reciente" TEXT, "J2 Rend. Superficie" TEXT, "Rend. Ultra reciente J2" TEXT,
    "Cuota J1" TEXT, "Cuota J2" TEXT, "Ganador" TEXT, "Fecha_dt" TEXT, "Hora" TEXT
);

-- Las otras tablas ya están bien:
-- ✅ tournaments (9867 filas cargadas)
-- ✅ historical_fixtures (36321 filas cargadas) 
-- ✅ rankings (97438 filas cargadas)
