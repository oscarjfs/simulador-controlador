# Lista de Mejoras (To-Dos) - Simulador Controlador PID

## 1. Funcionalidad de Ingeniería de Control

* [ ] **Inyección de Perturbaciones (Load Disturbances):** El modelo actual asume cambios solo en el *set point* (servocontrol). Agregar un botón o un slider para introducir perturbaciones tipo escalón en la variable manipulada o directamente en la variable de proceso (`d(t)`) enriquecería enormemente el análisis de regulación.
* [ ] **Cálculo de Métricas de Desempeño:** Implementar el cálculo en tiempo real de índices de error estándar como IAE (Integral del Error Absoluto) o ISE (Integral del Error Cuadrático). Estos valores podrían mostrarse en la pestaña "Controlador" o incluirse como columnas adicionales en la exportación de datos.
* [ ] **Saturación del Controlador Anti-Windup:** Aunque la librería `pyAutoControl` maneje el cálculo, visualizar gráficamente en el simulador cuándo el `CO` llega a los límites (`CO_MIN`, `CO_MAX`) y cómo afecta la acción integral ayudaría a visualizar el fenómeno de *windup*.

## 2. Experiencia de Usuario (UX) e Interfaz

* [ ] **Funcionalidad de "Pausa":** Actualmente la simulación se inicia o se detiene por completo. Una función de pausa que congele el tiempo de simulación y mantenga el estado actual en memoria permitiría analizar la gráfica con calma antes de modificar parámetros y continuar.
* [ ] **Tooltips Educativos:** Al pasar el cursor sobre las etiquetas de `Kc`, `tau_i`, `tau_d`, o las variables del FOPDT, se podría mostrar un pequeño globo de texto con una breve definición teórica. Es un detalle menor que mejora la accesibilidad pedagógica.
* [ ] **Guardado Explícito de Configuración:** La clase `Configuracion` tiene el método `guardar_configuracion`, pero la GUI no expone un botón para que el usuario guarde su estado actual como el nuevo estado "por defecto" (sobrescribir el archivo `config.yaml` voluntariamente).

## 3. Arquitectura y Ecosistema

* [ ] **Gestión Moderna de Dependencias:** Incorporar un archivo `pyproject.toml` estructurado para gestionar el proyecto. Utilizar gestores de entorno ultrarrápidos como `uv` garantizará que el simulador se instale y ejecute sin fricciones tanto en distribuciones Linux como Kubuntu, así como en los sistemas operativos de quienes clonen el proyecto.
* [ ] **Desacople Estricto de la GUI y la Lógica:** En métodos como `cambiar_sistema_simulado`, la lógica del negocio manipula directamente el estado visual de los widgets (`grid_remove`, `configure`). Implementar un patrón de observador o señales estandarizadas dejaría el código aún más limpio.
* [ ] **Alternativa de Despliegue Web:** Ya que el núcleo matemático (FOPDT y PID) está bien encapsulado, considerar desarrollar una interfaz paralela utilizando frameworks como Streamlit. Esto eliminaría la necesidad de instalaciones locales y permitiría a los estudiantes acceder al simulador directamente desde un navegador web interactivo.
* [ ] **Type Hinting (Anotación de Tipos):** Incorporar anotaciones de tipo estándar en Python (`def fopdt(self, t: float, y: float, co: float) -> float:`) a lo largo del código para mejorar el autocompletado en los IDEs y la mantenibilidad a largo plazo.