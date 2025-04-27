"""
Nombre: simulador_controlador.py
Autor: Oscar Franco
Versión: 8.4 (2025-04-26)
Descripción: Aplicación para simular el comportamiento de un sistema según su función de transferencia
en lazo abierto o aplicando un controlador PID.
"""

import json
import logging
import datetime
import numpy as np
from customtkinter import CTk, CTkButton, CTkEntry, CTkLabel, CTkFrame, CTkTabview, CTkSlider, CTkSwitch, CTkRadioButton, BooleanVar, StringVar, set_appearance_mode, set_default_color_theme
from tkinter.messagebox import showerror, askyesno
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.integrate import solve_ivp
from pandas import DataFrame
from pyAutoControl.PIDController import PIDController

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Clase para manejar la configuración
class Configuracion:
    CONFIG_FILE = 'config.json'
    
    def __init__(self):
        logging.info("Cargando configuración...")
        self.configuracion = self.cargar_configuracion()
        logging.info("Configuración cargada exitosamente.")
        
    def cargar_configuracion(self):
        try:
            with open(self.CONFIG_FILE) as file:
                config = json.load(file)
                logging.info(f"Configuración cargada desde {self.CONFIG_FILE}")
                return config
        except FileNotFoundError:
            logging.warning(f"Archivo de configuración {self.CONFIG_FILE} no encontrado. Creando configuración por defecto.")
            self.crear_configuracion_default()
            with open(self.CONFIG_FILE) as file:
                config = json.load(file)
                logging.info(f"Configuración por defecto cargada desde {self.CONFIG_FILE}")
                return config
        except json.JSONDecodeError:
            logging.error("Error al decodificar el archivo de configuración. Creando configuración por defecto.")
            self.crear_configuracion_default()
            with open(self.CONFIG_FILE) as file:
                config = json.load(file)
                logging.info(f"Configuración por defecto cargada desde {self.CONFIG_FILE}")
                return config
    
    def crear_configuracion_default(self):
        config_default = {
            "variance": 5e-09,
            "tVel": 50,
            "ruidoSenalEncendido": True,
            "Ts": 0.1,
            "controlAutomaticoEncendido": True,
            "tminGrafica": 120,
            "Kp": 4.59,
            "taup": 15.14,
            "td": 5.0,
            "Kc": 1.0,
            "Ki": 0.0,
            "Kd": 0.0,
            "y0": 50,
            "co0": 50,
            "ysp0": 50,
            "CO_MIN": 0,
            "CO_MAX": 100
        }
        
        with open(self.CONFIG_FILE, 'w') as file:
            json.dump(config_default, file, indent=4)
            logging.info(f"Configuración por defecto creada y guardada en {self.CONFIG_FILE}")
    
    def guardar_configuracion(self, nueva_config):
        """Guarda los cambios de configuración en el archivo JSON."""
        with open(self.CONFIG_FILE, 'w') as file:
            json.dump(nueva_config, file, indent=4)
            logging.info(f"Configuración actualizada y guardada en {self.CONFIG_FILE}")

# Clase para la interfaz gráfica
class GUI:
    def __init__(self, simulador):
        self.simulador = simulador
        self.config = simulador.configuracion
        self.crear_gui()

    def crear_gui(self):
        """Crea la ventana principal de la interfaz gráfica y sus componentes."""
        logging.info("Creando la interfaz gráfica...")
        set_appearance_mode("System")
        set_default_color_theme("blue")
        
        self.ventana = CTk()
        self.ventana.geometry("950x430")
        self.ventana.minsize(950, 430)
        self.ventana.title('Simulador de Lazos de Control by OF')
        self.ventana.protocol('WM_DELETE_WINDOW', self.simulador.finalizar_aplicacion)
        
        self.crear_grafica_tendencia()
        self.crear_comandos_gui()
        
        logging.info("Interfaz gráfica creada exitosamente.")

    def crear_grafica_tendencia(self):
        """Crea y configura la figura y ejes para la gráfica de tendencia."""
        self.fig, self.ax = plt.subplots(facecolor='grey')
        plt.title("Gráfica de Tendencia", color='black', size=16)
        self.ax.set_facecolor('black')
        self.ax.set_xlabel("t [s]", color='black')
        self.ax.set_ylabel("y", color='blue')
        self.ax.grid(axis='x', color='gray', linestyle='dashed')
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.yaxis.set_major_locator(MultipleLocator(1))
        self.ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        self.ax.xaxis.grid(which='minor', linestyle='dotted', color='gray')
        self.ax.tick_params(direction='out', colors='w', grid_color='w', grid_alpha=0.3)
        
        self.twax = self.ax.twinx()
        self.twax.set_ylabel('CO [%]', color='purple')
        self.twax.set_ylim(0, 100)
        self.twax.tick_params(direction='out', length=6, width=1, colors='purple')

        # Crear las líneas de la gráfica y almacenar sus referencias
        self.line_y, = self.ax.plot([], [], color='b', label='Y', linestyle='solid')
        self.line_ysp, = self.ax.plot([], [], color='r', label='Ysp', linestyle='dashed')
        self.line_co, = self.twax.plot([], [], color='purple', label='CO', linestyle='solid')

        # Configurar la leyenda inicial
        handles1, labels1 = self.ax.get_legend_handles_labels()
        handles2, labels2 = self.twax.get_legend_handles_labels()
        self.ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        self.frameGrafico = CTkFrame(self.ventana)
        self.frameGrafico.pack(side="left", expand=True, fill='both')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frameGrafico)
        self.canvas.get_tk_widget().pack(expand=True, padx=10, pady=10, fill='both')

    def crear_comandos_gui(self):
        """Crea el frame para los comandos y la vista de pestañas."""
        self.frameComandos = CTkFrame(self.ventana)
        self.frameComandos.pack(side="right", fill='y')
        
        self.tabview = CTkTabview(self.frameComandos)
        self.tabview.grid(row=0, column=0, columnspan=3, sticky='n')
        
        self.tabview.add("Simulación")
        self.tabview.add("Controlador")
        self.tabview.add("Exportado")
        
        self.crear_tab_simulacion()
        self.crear_tab_controlador()
        self.crear_tab_exportado()
        
        self.boton_iniciar = CTkButton(self.frameComandos, text='Iniciar', width=20, 
                                       command=self.simulador.iniciar_simulacion, fg_color='green')
        self.boton_iniciar.grid(column=0, row=22, padx=10, pady=10)
        
        CTkButton(self.frameComandos, text='Reiniciar', width=20, 
                 command=self.simulador.reiniciar_simulacion).grid(column=1, row=22, padx=5, pady=5)
        
        CTkButton(self.frameComandos, text='Finalizar', width=20, 
                 command=self.simulador.finalizar_aplicacion).grid(column=2, row=22, padx=5, pady=5)
        
        self.labelStatus = CTkLabel(self.frameComandos, text='')
        self.labelStatus.grid(column=0, row=24, columnspan=2)

    def crear_tab_simulacion(self):
        """Crea los elementos de la pestaña 'Simulación'."""
        CTkLabel(self.tabview.tab("Simulación"), text='PARÁMETROS DEL SISTEMA',
                font=('Verdana', 14, 'bold')).grid(padx=10, pady=10, row=0, column=0, columnspan=3)
        
        self.entradaKp = self.crear_parametro_input(self.tabview.tab("Simulación"), 'Kp', 
                                                   self.simulador.Kp, 1, self.simulador.actualizar_kp)
        
        self.entradaTaup = self.crear_parametro_input(self.tabview.tab("Simulación"), 'Tau', 
                                                     self.simulador.taup, 2, self.simulador.actualizar_taup)
        
        self.entradaTd = self.crear_parametro_input(self.tabview.tab("Simulación"), 'td', 
                                                   self.simulador.td, 3, self.simulador.actualizar_td)
        
        CTkLabel(self.tabview.tab("Simulación"), text='Velocidad de simulación:').grid(column=0, row=18)
        
        self.scaleVelocidad = CTkSlider(self.tabview.tab("Simulación"), from_=0, to=100, 
                                       number_of_steps=10, command=self.simulador.actualizar_velocidad)
        self.scaleVelocidad.grid(padx=10, pady=10, row=19, column=0, columnspan=2)
        self.scaleVelocidad.set(self.simulador.tVel)
        
        self.simularRuido = BooleanVar(value=self.simulador.ruidoSenalEncendido)
        self.checkSimularRuido = CTkSwitch(self.tabview.tab("Simulación"), text='Simulación de Señal Ruidosa',
                                          variable=self.simularRuido, command=self.simulador.actualizar_estado_ruido)
        self.checkSimularRuido.grid(padx=10, pady=10, row=20, column=0, columnspan=2)

    def crear_tab_controlador(self):
        """Crea los elementos de la pestaña 'Controlador'."""
        self.entradaSetPoint = self.crear_parametro_input(self.tabview.tab("Controlador"), 'Set point',
                                                        self.simulador.yspActual, 7, self.simulador.actualizar_sp)
        
        CTkButton(self.tabview.tab("Controlador"), text='Actualizar SP', width=20, 
                 command=self.simulador.actualizar_sp).grid(padx=10, pady=10, row=8, column=0, columnspan=2)
        
        CTkLabel(self.tabview.tab("Controlador"), text='GANANCIAS DEL CONTROLADOR', 
                font=('Verdana', 14, 'bold')).grid(padx=10, pady=10, row=10, column=0, columnspan=2)
        
        self.entradaKc = self.crear_parametro_input(self.tabview.tab("Controlador"), 'Kc', 
                                                   self.simulador.Kc, 11, self.simulador.actualizar_ganancias)
        
        self.entradaKi = self.crear_parametro_input(self.tabview.tab("Controlador"), 'Ki', 
                                                   self.simulador.Ki, 13, self.simulador.actualizar_ganancias)
        
        self.entradaKd = self.crear_parametro_input(self.tabview.tab("Controlador"), 'Kd', 
                                                   self.simulador.Kd, 14, self.simulador.actualizar_ganancias)
        
        CTkButton(self.tabview.tab("Controlador"), text='Actualizar Ganancias', width=20, 
                 command=self.simulador.actualizar_ganancias).grid(padx=10, pady=10, row=15, column=0, columnspan=2)
        
        self.controlAutomatico = BooleanVar(value=self.simulador.controlAutomaticoEncendido)
        self.checkControlAutomatico = CTkSwitch(self.tabview.tab("Controlador"), text='Control Automático Activo',
                                              variable=self.controlAutomatico, command=self.simulador.actualizar_estado_control)
        self.checkControlAutomatico.grid(padx=10, pady=10, row=16, column=0, columnspan=2)
        
        self.labelCO = CTkLabel(self.tabview.tab("Controlador"), text='CO: ')
        self.entradaCO = CTkEntry(self.tabview.tab("Controlador"), width=100)
        self.entradaCO.bind('<Return>', self.simulador.actualizar_co)

    def crear_tab_exportado(self):
        """Crea los elementos de la pestaña 'Exportado'."""
        CTkButton(self.tabview.tab("Exportado"), text='Exportar', width=20,
                 command=self.simulador.exportar_datos).grid(column=0, row=4, padx=5, pady=5, columnspan=2)
        
        self.formatoExportado = StringVar(value='xlsx')
        CTkLabel(self.tabview.tab("Exportado"), text="Formato de exportado:").grid(row=0, column=0, columnspan=1, padx=10, pady=10, sticky="")
        
        CTkRadioButton(self.tabview.tab("Exportado"), variable=self.formatoExportado, 
                      value='xlsx', text='xlsx').grid(row=2, column=0, pady=10, padx=20)
        
        CTkRadioButton(self.tabview.tab("Exportado"), variable=self.formatoExportado, 
                      value='csv', text='csv').grid(row=2, column=1, pady=10, padx=20)

    def crear_parametro_input(self, parent, label, def_value, row, command):
        """Crea un par de Label y Entry para un parámetro de entrada."""
        CTkLabel(parent, text=f'{label}: ').grid(pady=5, row=row, column=0)
        entrada = CTkEntry(parent, width=100)
        entrada.insert(0, str(def_value))
        entrada.grid(padx=5, row=row, column=1)
        entrada.bind('<Return>', command)
        return entrada
    
    def actualizar_grafica(self, t, y, ysp, co, tActual, tminGrafica, nDatosGrafica):
        """Actualiza la gráfica de tendencia con los nuevos valores."""
        # Actualizar los datos de las líneas existentes
        self.line_y.set_data(t, y)
        self.line_ysp.set_data(t, ysp)
        self.line_co.set_data(t, co)

        # Ajustar los límites del eje X
        if tActual <= tminGrafica:
            self.ax.set_xlim(0, tminGrafica)
        else:
            self.ax.set_xlim(tActual - tminGrafica, tActual)

        # Ajustar los límites del eje Y principal (y, ysp)
        # Considerar solo los datos dentro de la ventana visible
        t_visible_indices = [i for i, time in enumerate(t) if time >= self.ax.get_xlim()[0] and time <= self.ax.get_xlim()[1]]
        if t_visible_indices:
            y_visible = [y[i] for i in t_visible_indices]
            ysp_visible = [ysp[i] for i in t_visible_indices]
            y_min = np.amin([np.amin(y_visible), np.amin(ysp_visible)])
            y_max = np.amax([np.amax(y_visible), np.amax(ysp_visible), 1]) # Asegurar un límite superior mínimo de 1
            self.ax.set_ylim(y_min * 0.95, y_max * 1.05)
        else:
             self.ax.set_ylim(0, 100) # Límites por defecto si no hay datos visibles

        # Ajustar los límites del eje Y secundario (co)
        if t_visible_indices:
            co_visible = [co[i] for i in t_visible_indices]
            co_min = np.amin(co_visible)
            co_max = np.amax(co_visible)
            self.twax.set_ylim(0 if co_min < 0 else co_min * 0.95,
                               1 if co_max < 1 else co_max * 1.05)
        else:
            self.twax.set_ylim(0, 100) # Límites por defecto si no hay datos visibles

        # Redibujar el canvas
        self.canvas.draw()

    def ejecutar(self):
        """Inicia el loop principal de la interfaz."""
        logging.info("Iniciando el loop principal de la interfaz...")
        self.ventana.mainloop()

# Clase para la simulación del controlador PID
class SimuladorControlador:
    def __init__(self):
        """Inicializa el simulador, carga la configuración y crea la GUI."""
        logging.info("Inicializando SimuladorControlador...")
        self.configuracion_manager = Configuracion()
        self.configuracion = self.configuracion_manager.configuracion
        self.inicializar_parametros()
        self.inicializar_estado_simulacion()
        self.gui = GUI(self)
        logging.info("SimuladorControlador inicializado exitosamente.")
    
    # Inicialización de parámetros de simulación desde la configuración
    def inicializar_parametros(self):
        """Inicializa los parámetros de simulación a partir de la configuración cargada."""
        logging.info("Inicializando parámetros de simulación...")
        try:
            self.variance = self.configuracion['variance']
            self.tVel = self.configuracion['tVel']
            self.Ts = self.configuracion['Ts']
            self.controlAutomaticoEncendido = self.configuracion['controlAutomaticoEncendido']
            self.tminGrafica = self.configuracion['tminGrafica']
            self.Kp = self.configuracion['Kp']
            self.taup = self.configuracion['taup']
            self.td = self.configuracion['td']
            self.Kc = self.configuracion['Kc']
            self.Ki = self.configuracion['Ki']
            self.Kd = self.configuracion['Kd']
            self.cambiosParametros = False
            self.CO_MIN = self.configuracion['CO_MIN']
            self.CO_MAX = self.configuracion['CO_MAX']
            self.controller = PIDController(self.Ts, self.Kc, self.Ki, self.Kd, self.CO_MIN, self.CO_MAX)
            self.controller.set_controller_status(self.controlAutomaticoEncendido)
            logging.info("Parámetros de simulación inicializados exitosamente.")
        except KeyError as e:
            logging.error(f"Error al inicializar parámetros: Falta la clave {e} en el archivo de configuración.")
            showerror("Error", f"Error al inicializar parámetros: Falta la clave {e} en el archivo de configuración.")
    
    # Inicializar variables de estado
    def inicializar_estado_simulacion(self):
        """Inicializa las variables de estado de la simulación."""
        logging.info("Inicializando variables de estado de simulación...")
        try:
            self.tstep = 0
            self.tActual = 0
            self.yActual = self.configuracion['y0']
            self.coActual = self.configuracion['co0']
            self.yspActual = self.configuracion['ysp0']
            self.estadoSimulacion = False
            self.ruidoSenalEncendido = self.configuracion['ruidoSenalEncendido']
            self.t0 = self.tActual
            self.y0 = self.yActual
            self.co0 = self.coActual
            self.ysp0 = self.yspActual
            self.t = [self.tActual]
            self.y = [self.yActual]
            self.co = [self.coActual]
            self.ysp = [self.yspActual]
            self.nDatosGrafica = round(self.tminGrafica / self.Ts)
            logging.info("Variables de estado de simulación inicializadas exitosamente.")
        except KeyError as e:
            logging.error(f"Error al inicializar variables de estado: Falta la clave {e} en el archivo de configuración.")
            showerror("Error", f"Error al inicializar variables de estado: Falta la clave {e} en el archivo de configuración.")

    # Actualizaciones de parámetros
    def _actualizar_parametro_gui(self, entrada_gui, nombre_parametro, tipo_dato, callback_actualizar, event=None):
        """Función auxiliar para validar y actualizar parámetros desde entradas de la GUI."""
        try:
            valor = tipo_dato(entrada_gui.get())
            callback_actualizar(valor)
            self.cambiosParametros = True
        except ValueError:
            showerror("Error", f"Ingrese un valor numérico válido para {nombre_parametro}.")
            entrada_gui.delete(0, "end")
            # Restaurar el valor anterior si la conversión falla
            if nombre_parametro == 'Kp':
                entrada_gui.insert(0, str(self.Kp))
            elif nombre_parametro == 'Tau':
                entrada_gui.insert(0, str(self.taup))
            elif nombre_parametro == 'td':
                entrada_gui.insert(0, str(self.td))
            elif nombre_parametro == 'Set point':
                 entrada_gui.insert(0, str(self.yspActual))
            elif nombre_parametro == 'CO':
                 entrada_gui.insert(0, str(self.coActual))
            # Para ganancias (Kc, Ki, Kd) se maneja en actualizar_ganancias

    def actualizar_kp(self, event=None):
        """Actualiza el valor de Kp basado en la entrada del usuario."""
        self._actualizar_parametro_gui(self.gui.entradaKp, 'Kp', float, lambda val: setattr(self, 'Kp', val), event)

    def actualizar_taup(self, event=None):
        """Actualiza el valor de Taup basado en la entrada del usuario."""
        self._actualizar_parametro_gui(self.gui.entradaTaup, 'Tau', float, lambda val: setattr(self, 'taup', val), event)

    def actualizar_td(self, event=None):
        """Actualiza el valor de Td basado en la entrada del usuario."""
        self._actualizar_parametro_gui(self.gui.entradaTd, 'td', float, lambda val: setattr(self, 'td', val), event)

    def actualizar_sp(self, event=None):
        """Actualiza el valor del set point (ysp) basado en la entrada del usuario."""
        try:
            self.yspActual = float(self.gui.entradaSetPoint.get())
            self.tstep = self.tActual
            self.co0 = self.co[-1]
            self.y0 = self.y[-1]
        except ValueError:
            showerror("Error", "Ingrese un valor numérico válido para el set point.")
            self.gui.entradaSetPoint.delete(0, "end")
            self.gui.entradaSetPoint.insert(0, str(self.yspActual))

    def actualizar_ganancias(self, event=None):
        """Actualiza las ganancias del controlador (Kc, Ki, Kd) basado en la entrada del usuario."""
        try:
            self.Kc = float(self.gui.entradaKc.get())
            self.Ki = float(self.gui.entradaKi.get())
            self.Kd = float(self.gui.entradaKd.get())
            self.cambiosParametros = True
            self.controller.set_controller_gains(self.Kc, self.Ki, self.Kd)
        except ValueError:
            showerror("Error", "Ingrese valores numéricos válidos para Kc, Ki y Kd.")
            self.gui.entradaKc.delete(0, "end")
            self.gui.entradaKc.insert(0, str(self.Kc))
            self.gui.entradaKi.delete(0, "end")
            self.gui.entradaKi.insert(0, str(self.Ki))
            self.gui.entradaKd.delete(0, "end")
            self.gui.entradaKd.insert(0, str(self.Kd))

    def actualizar_estado_control(self):
        """Actualiza el estado del control automático (encendido/apagado)."""
        self.controlAutomaticoEncendido = self.gui.controlAutomatico.get()
        self.controller.set_controller_status(self.controlAutomaticoEncendido)
        
        if self.controlAutomaticoEncendido:
            self.gui.labelCO.grid_forget()
            self.gui.entradaCO.grid_forget()
        else:
            self.gui.labelCO.grid(pady=5, row=17, column=0)
            self.gui.entradaCO.grid(padx=5, row=17, column=1)
            self.gui.entradaCO.delete(0, "end")
            self.coActual = self.co[-1]
            self.gui.entradaCO.insert(0, str(round(self.coActual, 1)))

    def actualizar_co(self, event=None):
        """Actualiza el valor de CO (Control Output) basado en la entrada del usuario."""
        try:
            self.coActual = float(self.gui.entradaCO.get())
            self.tstep = self.tActual
            self.co0 = self.co[-1]
            self.y0 = self.y[-1]
        except ValueError:
            showerror("Error", "Ingrese un valor numérico válido para CO.")
            self.gui.entradaCO.delete(0, "end")
            self.gui.entradaCO.insert(0, str(self.coActual))

    def actualizar_velocidad(self, event=None):
        """Actualiza la velocidad de simulación basada en el valor del slider."""
        nuevaVelocidad = round(self.gui.scaleVelocidad.get(), -1)
        if nuevaVelocidad < 1:
            self.gui.scaleVelocidad.set(1)
            self.tVel = 1
        else:
            self.gui.scaleVelocidad.set(nuevaVelocidad)
            self.tVel = int(nuevaVelocidad)

    def actualizar_estado_ruido(self):
        """Actualiza el estado de la simulación de ruido (encendido/apagado)."""
        self.ruidoSenalEncendido = self.gui.simularRuido.get()
    
    # Iniciar simulación
    def iniciar_simulacion(self):
        """Inicia la simulación del sistema."""
        logging.info("Iniciando simulación...")
        self.estadoSimulacion = True
        
        try:
            if float(self.gui.entradaKp.get()) != self.Kp:
                self.actualizar_kp()
            if float(self.gui.entradaTaup.get()) != self.taup:
                self.actualizar_taup()
            if float(self.gui.entradaTd.get()) != self.td:
                self.actualizar_td()
            
            self.gui.entradaKp.configure(state='disabled')
            self.gui.entradaTaup.configure(state='disabled')
            self.gui.entradaTd.configure(state='disabled')
            
            self.simulacion_pid()
            
            self.gui.boton_iniciar.configure(text='Detener', command=self.detener_simulacion, fg_color='red')
            
            logging.info("Simulación iniciada exitosamente.")
        except ValueError as e:
            logging.error(f"Error al iniciar la simulación: Ingrese un valor numérico válido para Kp, Tau y td. {e}")
            showerror("Error", "Ingrese un valor numérico válido para Kp, Tau y td.")
    
    # Detener los cálculos de la simulación
    def detener_simulacion(self):
        logging.info("Deteniendo simulación...")
        self.estadoSimulacion = False
        self.gui.entradaKp.configure(state='normal')
        self.gui.entradaTaup.configure(state='normal')
        self.gui.entradaTd.configure(state='normal')
        self.gui.boton_iniciar.configure(text='Iniciar', command=self.iniciar_simulacion, fg_color='green')
        logging.info("Simulación detenida exitosamente.")
    
    # Reiniciar los cálculos de la simulación
    def reiniciar_simulacion(self):
        logging.info("Reiniciando simulación...")
        self.tstep = 0
        self.tActual = 0
        self.yActual = self.y[-1]
        self.coActual = self.co[-1]
        self.yspActual = self.ysp[-1]
        self.t0 = self.tActual
        self.y0 = self.yActual
        self.co0 = self.coActual
        self.ysp0 = self.yspActual
        self.t = [self.tActual]
        self.y = [self.yActual]
        self.co = [self.coActual]
        self.ysp = [self.yspActual]
        self.gui.actualizar_grafica(self.t, self.y, self.ysp, self.co, self.tActual, self.tminGrafica, self.nDatosGrafica)
        logging.info("Simulación reiniciada exitosamente.")
    
    # Definición del modelo FOPDT
    def fopdt(self, t, y, co):
        """Define la ecuación diferencial del modelo FOPDT."""
        u = 0 if t < self.td + self.tstep else 1
        dydt = -(y - self.y0) / self.taup + self.Kp / self.taup * u * (co - self.co0)
        return dydt
    
    # Solución del modelo del lazo de control
    def simulacion_pid(self):
        """Simulación del lazo de control PID."""
        try:
            for i in range(self.tVel):
                self.tActual = self.t[-1] + self.Ts
                self.t.append(self.tActual)
                ts = [self.t[-2], self.t[-1]]  # Intervalo de tiempo
                self.ysp.append(self.yspActual)
                
                coAtrasado = self.co[-int(self.td/self.Ts)] if len(self.co) > int(self.td/self.Ts) else self.co0
                
                sol = solve_ivp(self.fopdt, ts, [self.y[-1]], method='RK45', t_eval=[self.tActual], args=(coAtrasado,))
                self.y.append(float(sol.y[0][-1]) * np.random.normal(1, np.sqrt(self.variance * self.ruidoSenalEncendido)))
                
                nuevoCO = self.controller.calculate_CO(self.y[-1], self.ysp[-1], self.co[-1] if self.controlAutomaticoEncendido else self.coActual)
                self.co.append(nuevoCO)
            
            self.gui.actualizar_grafica(self.t, self.y, self.ysp, self.co, self.tActual, self.tminGrafica, self.nDatosGrafica)
            
            if self.estadoSimulacion:
                self.gui.ventana.after(100, self.simulacion_pid)  # Actualización de la gráfica
        
        except Exception as e:
            logging.error(f"Error durante la simulación: {e}")
            showerror("Error", f"Error durante la simulación: {e}")
    
    # Exportar datos de la tendencia a CSV/XLSX
    def exportar_datos(self):
        ahora = datetime.datetime.now()
        nombreArchivo = ahora.strftime("data_%Y-%m-%d_%H_%M_%S")
        
        datos = {
            't': self.t,
            'CO': self.co,
            'y': self.y,
            'ysp': self.ysp
        }
        
        df = DataFrame(datos)
        formato = self.gui.formatoExportado.get()
        
        if formato == 'xlsx':
            df.to_excel(nombreArchivo + '.xlsx', index=False)
        elif formato == 'csv':
            df.to_csv(nombreArchivo + '.csv', decimal=',', sep=';', index=False)
        
        self.gui.labelStatus.configure(text=f'Exportado {ahora}')  # mensaje de confirmación del exportado
        logging.info(f"Datos exportados a {nombreArchivo}.{formato}")
    
    # Función para finalizar la aplicación
    def finalizar_aplicacion(self):
        """Finaliza la aplicación preguntando al usuario."""
        logging.info("Finalizando aplicación...")
        if askyesno(message='¿Desea salir del simulador?', title='Simulador de Lazos de Control by OF'):
            if self.cambiosParametros:
                if askyesno(message='¿Desea guardar la confirugación actual?', title='Simulador de Lazos de Control by OF'):
                    self.configuracion['Kp'] = float(self.gui.entradaKp.get())
                    self.configuracion['taup'] = float(self.gui.entradaTaup.get())
                    self.configuracion['td'] = float(self.gui.entradaTd.get())
                    self.configuracion['Kc'] = float(self.gui.entradaKc.get())
                    self.configuracion['Ki'] = float(self.gui.entradaKi.get())
                    self.configuracion['Kd'] = float(self.gui.entradaKd.get())
                    self.configuracion_manager.guardar_configuracion(self.configuracion)
            self.gui.ventana.quit()
            logging.info("Aplicación finalizada exitosamente.")
    
    # Ejecutar la aplicación
    def ejecutar(self):
        """Inicia el loop principal de la interfaz gráfica."""
        self.gui.ejecutar()

# Bloque de ejecución principal
if __name__ == '__main__':
    # Configurar logging a nivel INFO
    logging.basicConfig(level=logging.INFO)
    simulador = SimuladorControlador()
    simulador.ejecutar()
