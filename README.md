# 🎾 Otorongo Project

Aplicación de análisis estadístico de tenis que proporciona información fiable para la toma de decisiones.

## 📋 Descripción

Otorongo es una aplicación web desarrollada con Streamlit que permite analizar estadísticas de partidos de tenis profesional. El sistema descarga datos de la API de [api-tennis.com](https://api-tennis.com/documentation), los procesa para calcular métricas clave y los presenta en una interfaz intuitiva.

## ✨ Características Principales

- **📊 Análisis de Partidos**: Visualiza estadísticas detalladas de partidos del día
- **🔍 Head to Head (H2H)**: Historial de enfrentamientos directos entre jugadores
- **📈 Rendimiento Reciente**: Análisis de victorias/derrotas recientes
- **🏟️ Rendimiento por Superficie**: Estadísticas específicas por tipo de cancha (Hard, Clay, Grass)
- **🏆 Rankings ATP/WTA**: Puntos actuales de ranking de los jugadores
- **🚀 Optimización Avanzada**: Motor independiente (`heavy_optimizer.py`) para calcular pesos ideales mediante simulación masiva.
- **🔒 Acceso Protegido**: Autenticación por contraseña

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- Cuenta en [api-tennis.com](https://api-tennis.com/) para obtener API key

### Configuración

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/otorongo.git
cd otorongo
```

2. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

3. **Configurar variables de entorno**

Crear archivo `.env` en la raíz del proyecto:
```env
API_KEY=tu_api_key_aqui
APP_PASSWORD=tu_password_aqui
```

4. **Ejecutar la aplicación**
```bash
streamlit run streamlit_app.py
```

La aplicación estará disponible en `http://localhost:8501`

## 📊 Métricas Mostradas

- **H2H**: Victorias y porcentaje de enfrentamientos directos
- **Recent Performance**: Porcentaje de victorias en partidos recientes Singles
- **Surface Performance**: Porcentaje de victorias en la superficie del partido actual
- **ATP/WTA Points**: Puntos de ranking actuales

## 🛠️ Estructura del Proyecto

```
otorongo/
├── streamlit_app.py          # Aplicación principal Streamlit
├── main.py                    # Entry point alternativo
├── scripts/
│   ├── tenis_api.py          # Integración con API
│   ├── process_files.py      # Procesamiento de datos
│   ├── fetch_daily_data.py   # Actualización automática
│   ├── logger_config.py      # Configuración de logs
│   └── my_columns.py         # Configuración de columnas
├── data/                      # Datos descargados (CSV/JSON)
├── .github/workflows/         # GitHub Actions
└── requirements.txt           # Dependencias Python
```

## 📝 Uso

1. **Autenticación**: Ingresa la contraseña configurada en `APP_PASSWORD`
2. **Buscar Partidos**: Selecciona una fecha en el calendario
3. **Ver Detalles**: Haz clic en "See Details" para análisis completo del partido
4. **Explorar**: 
   - Revisa estadísticas H2H
   - Consulta rendimiento reciente de cada jugador
   - Analiza partidos recientes por superficie

## 🔄 Actualización de Datos

El proyecto incluye un workflow de GitHub Actions que actualiza automáticamente:
- Información de torneos
- Rankings ATP y WTA
- Corrección de datos incorrectos de la API

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/NuevaCaracteristica`)
3. Commit tus cambios (`git commit -m 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/NuevaCaracteristica`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es de uso personal.

## 🔗 Links Útiles

- [API Tennis Documentation](https://api-tennis.com/documentation)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📧 Contacto

Para preguntas o sugerencias, abre un issue en el repositorio.

---

**Nota**: Este proyecto requiere una API key válida de api-tennis.com para funcionar correctamente.
