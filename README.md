# simulador-controlador

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Contributing](../CONTRIBUTING.md)

## About <a name = "about"></a>

Simulador de un lazo de control para el uso de entrenamiento y aprendizaje.

## Getting Started <a name = "getting_started"></a>


### Prerequisites

Las librerias que se requieren para su funcionamiento:

```
matplotlib>=3.6.3
customtkinter==5.2.2
matplotlib>=3.10.8
numpy>=2.4.2
pandas>=3.0.0
openpyxl>=3.1.5
scipy>=1.17.0
tk==0.1.0
```

Tambien requiere la librería PIDController del repositorio:
https://github.com/oscarjfs/pyAutoControl


### Installing

Clone el repositorio

```
git clone https://github.com/oscarjfs/simulador-controlador.git
```

Instale las librerías corresponientes.
```
pip install -r requirements.txt
```


También puede descargar la versión ejecutable creada con pyinstaller.
https://github.com/oscarjfs/simulador-controlador/releases


## Usage <a name = "usage"></a>

Ejecute en consola:

```
python main.py
```