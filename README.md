# simulador-controlador

Simulador de lazo de control PID con interfaz gráfica para entrenamiento y aprendizaje.

## Tabla de Contenidos

- [Acerca de](#acerca-de)
- [Características](#características)
- [Requisitos Previos](#requisitos-previos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Licencia](#licencia)

## Acerca de

Este proyecto es un simulador interactivo de lazos de control PID diseñado para el aprendizaje y entrenamiento de conceptos de control automático. Permite visualizar en tiempo real el comportamiento de un proceso controlado mediante un controlador PID.

## Características

- **Interfaz Gráfica**: Interfaz moderna construida con customtkinter
- **Visualización en Tiempo Real**: Gráficas de tendencia actualizadas dinámicamente con matplotlib
- **Modelo de Proceso**: Simulación FOPDT (First Order Plus Dead Time)
- **Controlador PID**: Implementación completa de controlador PID con:
  - Modo automático y manual
  - Ajuste de ganancias (Kp, Ki, Kd)
  - Límites de salida (CO_MIN, CO_MAX)
- **Sistemas Predefinidos**: Selección de diferentes configuraciones de proceso
- **Parámetros Configurables**:
  - Velocidad de simulación
  - Ruido en la señal
  - Tiempo muerto del proceso
  - Set point
- **Exportación de Datos**: Exporta resultados a Excel (.xlsx) o CSV
- **Modelo de Intercambiador de Calor**: Incluye clases para simulación de intercambiadores de calor (ShellAndTube, Water2Steam)

## Requisitos Previos

Python 3.12 o superior.

Librerías requeridas:

```
customtkinter==5.2.2
matplotlib>=3.10.8
numpy>=2.4.2
pandas>=3.0.0
openpyxl>=3.1.5
scipy>=1.17.0
tk==0.1.0
pyAutoControl @ git+https://github.com/oscarjfs/pyAutoControl.git
pyXSteam
```

## Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/oscarjfs/simulador-controlador.git
```

2. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Opcional: instalar en modo editable:

```bash
pip install -e .
```

## Uso

Ejecutar en consola:

```bash
python main.py
```

### Controles de la Interfaz

- **Pestaña Simulación**: Configurar parámetros del sistema (Kp, Tau, td) y velocidad de simulación
- **Pestaña Controlador**: Ajustar set point y ganancias del PID (Kc, Ki, Kd), activar/desactivar control automático
- **Pestaña Exportado**: Exportar datos de la simulación a Excel o CSV

### Botones Principales

- **Iniciar/Detener**: Inicia o detiene la simulación
- **Reiniciar**: Reinicia la simulación a las condiciones iniciales
- **Finalizar**: Cierra la aplicación

## Estructura del Proyecto

```
.
├── main.py                    # Punto de entrada de la aplicación
├── simulador_controlador.py   # Clase principal del simulador y GUI
├── process.py                 # Modelos de procesos (intercambiadores de calor)
├── config.yaml                # Configuración runtime
├── process.yaml               # Parámetros de procesos
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Este archivo
```

## Licencia

MIT License
