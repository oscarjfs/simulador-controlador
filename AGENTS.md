# AGENTS.md - Development Guide for simualdor-controlador

## Project Overview

This is a Python desktop application (Python 3.12+) that provides a GUI simulator for control loops using PID controllers. It uses:
- **GUI**: customtkinter
- **Plotting**: matplotlib
- **Numerical**: numpy, scipy
- **Data**: pandas, openpyxl
- **Config**: pyyaml
- **Control**: pyAutoControl.PIDController

## Running the Application

```bash
python main.py
```

## Build/Test Commands

### Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install -e .
```

### Running Tests

This project does not have a formal test framework (pytest/unittest) configured. The `test.py` file is a manual test script:

```bash
python test.py
```

To run a single test manually, execute specific test functions or scripts directly:

```bash
python -c "from simulador_controlador import SimuladorControlador; print('Import OK')"
```

### Code Quality Tools (Recommended)

Install and run these tools manually if needed:

```bash
# Linting
pip install ruff
ruff check .

# Formatting
pip install black
black .

# Type checking
pip install mypy
mypy .

# Import sorting
pip install isort
isort .
```

## Code Style Guidelines

### General

- **Language**: Spanish for user-facing strings and internal variable names (as per existing codebase)
- **Python version**: 3.12+ (uses structural pattern matching, type unions `|`, etc.)
- **No type hints** currently used, but adding them is encouraged for new code

### Naming Conventions

- **Classes**: PascalCase (e.g., `Configuracion`, `GUI`)
- **Methods/Functions**: camelCase (e.g., `cargar_configuracion`, `crear_gui`)
- **Variables**: snake_case (e.g., `configuracion`, `process_params`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `CONFIG_FILE`)
- **Private methods**: prefix with underscore (e.g., `_crear_parametro_input`)

### Imports

Standard order (isort style):
1. Standard library (yaml, logging, datetime, etc.)
2. Third-party libraries (numpy, customtkinter, matplotlib, scipy, pandas)
3. Local application imports

Example:
```python
import yaml
import logging
import datetime
import numpy as np
from customtkinter import CTk, CTkButton, CTkEntry
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pandas import DataFrame
from pyAutoControl.PIDController import PIDController
```

### Type Hints

When adding type hints, use modern Python 3.12+ syntax:
- `list[str]` instead of `List[str]`
- `dict[str, int]` instead of `Dict[str, int]`
- `int | str` instead of `Union[int, str]`
- Use `None` for optional params: `def foo(x: int | None = None)`

### Error Handling

- Use try/except with specific exceptions
- Log warnings/errors appropriately
- Provide fallback defaults when loading config files fails
- Example pattern from codebase:
```python
try:
    with open(self.CONFIG_FILE, 'r') as file:
        config = yaml.safe_load(file)
except (FileNotFoundError, yaml.YAMLError, ValueError):
    logging.warning(f"Archivo {self.CONFIG_FILE} no válido...")
    self.crear_configuracion_default()
```

### Logging

- Use the `logging` module with INFO level by default
- Format: `logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')`
- Log important events: startup, configuration loading, errors

### GUI Development

- Use customtkinter components (CTk, CTkButton, CTkEntry, etc.)
- Layout: Use `pack()` or `grid()` consistently
- Window management: Set geometry, minsize, title
- Protocol handling for window close events

### Numerical/Scientific Code

- Use numpy for array operations
- Use scipy.integrate.solve_ivp for ODE solving
- Use pandas DataFrame for data export to Excel

### File Structure

```
.
├── main.py              # Entry point
├── simulador_controlador.py  # Main application class
├── process.py           # Process models
├── test.py              # Manual test script
├── config.yaml          # Runtime configuration
├── process.yaml         # Process parameters
├── requirements.txt     # Dependencies
└── README.md            # Documentation
```

### Best Practices

1. **Configuration files**: Use YAML with `yaml.safe_load()` and `yaml.dump()`
2. **Defaults**: Generate default configs if files are missing/invalid
3. **Validation**: Check for None/empty values after loading
4. **Export**: Use pandas with openpyxl engine for Excel export
5. **Documentation**: Add docstrings for public APIs

### What NOT to Do

- Do not commit secrets, API keys, or credentials
- Do not modify config.yaml or process.yaml in production code
- Do not use deprecated libraries (tkinter.tk is replaced by customtkinter)
- Do not skip error handling on file I/O operations
