"""
Nombre: simulador_controlador.py
Autor: Oscar Franco
Versión: 7 (2024-09-17)
Descripción: Aplicación para simular el comportamiento de un sistema según su función de transferencia
            en lazo abierto o aplicando un controlador PID.
"""

import json
import datetime
import numpy as np
from customtkinter import CTk, CTkButton, CTkEntry, CTkLabel, CTkFrame, CTkTabview, CTkSlider, CTkSwitch, CTkRadioButton, BooleanVar, StringVar, set_appearance_mode, set_default_color_theme
from tkinter.messagebox import showerror, askyesno
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.integrate import odeint, solve_ivp
from pandas import DataFrame

# Clase para la simulación del controlador PID
class SimuladorControlador:
    CONFIG_FILE = 'config.json'

    def __init__(self):
        self.configuracion = self.cargar_configuracion()
        self.inicializar_parametros()
        self.inicializar_estado_simulacion()
        self.crear_gui()

    # Cargar configuración desde archivo JSON
    def cargar_configuracion(self):
        try:
            with open(self.CONFIG_FILE) as file:
                return json.load(file)
        except FileNotFoundError:
            self.crear_configuracion_default()
            with open(self.CONFIG_FILE) as file:
                return json.load(file)

    # Crear configuración por defecto si no existe
    def crear_configuracion_default(self):
        config_default = {
            "mean": 1.0,
            "variance": 5e-09,
            "tVel": 50,
            "ruidoSenalEncendido": True,
            "Ts": 0.1,
            "controlAutomaticoEncendido": True,
            "tminGrafica": 120,
            "estadoSimulacion": True,
            "Kp": 4.59,
            "taup": 15.14,
            "td": 5.0,
            "Kc": 0.6058,
            "Ki": 0.0606,
            "Kd": 0.0,
            "t0": 0,
            "y0": 50,
            "co0": 50,
            "u0": 0,
            "ysp0": 50
        }
        with open(self.CONFIG_FILE, 'w') as file:
            json.dump(config_default, file, indent=4)

    # Inicialización de parámetros de simulación desde la configuración
    def inicializar_parametros(self):
        config = self.configuracion
        self.mean = config['mean']
        self.variance = config['variance']
        self.tVel = config['tVel']
        self.Ts = config['Ts']
        self.controlAutomaticoEncendido = config['controlAutomaticoEncendido']
        self.tminGrafica = config['tminGrafica']
        self.Kp = config['Kp']
        self.taup = config['taup']
        self.td = config['td']
        self.Kc = config['Kc']
        self.Ki = config['Ki']
        self.Kd = config['Kd']
        self.cambiosParametros=False

    # Inicializar variables de estado
    def inicializar_estado_simulacion(self):
        self.tstep = 0
        self.tActual = self.configuracion['t0']
        self.yActual = self.configuracion['y0']
        self.coActual = self.configuracion['co0']
        self.yspActual = self.configuracion['ysp0']
        self.estadoSimulacion = self.configuracion['estadoSimulacion']
        self.ruidoSenalEncendido = self.configuracion['ruidoSenalEncendido']

        self.t0 = self.configuracion['t0']
        self.y0 = self.configuracion['y0']
        self.co0 = self.configuracion['co0']
        self.u0 = self.configuracion['u0']
        self.ysp0 = self.configuracion['ysp0']

        self.Ek2 = 0
        self.Ek1 = 0
        self.Ek = 0

        self.t = [0]
        self.y = [self.yActual]
        self.co = [self.coActual]
        self.ysp = [self.yspActual]
        self.nDatosGrafica = round(self.tminGrafica / self.Ts)

    # GUI principal
    def crear_gui(self):
        set_appearance_mode("System")
        set_default_color_theme("blue")

        self.ventana = CTk()
        self.ventana.geometry("950x430")
        self.ventana.minsize(950, 430)
        self.ventana.title('Simulador de Lazos de Control by OF')
        self.ventana.protocol('WM_DELETE_WINDOW', self.finalizar_aplicacion)

        self.crear_grafica_tendencia()
        self.crear_comandos_gui()

    # Crear la gráfica de tendencias
    def crear_grafica_tendencia(self):
        self.fig, self.ax = plt.subplots(facecolor='grey')
        plt.title("Gráfica de Tendencia", color='black', size=16)

        self.ax.set_facecolor('black')
        self.ax.axhline(linewidth=2, color='w')
        self.ax.axvline(linewidth=2, color='w')
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
        self.twax.tick_params(direction='out', length=6, width=1, colors='purple') #AQUI

        self.frameGrafico = CTkFrame(self.ventana)
        self.frameGrafico.pack(side="left", expand=True, fill='both')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frameGrafico)
        self.canvas.get_tk_widget().pack(expand=True, padx=10, pady=10, fill='both')

    # Crear el área de comandos (controlador y simulación)
    def crear_comandos_gui(self):
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

        # finalización y barra de estado
        CTkButton(self.frameComandos, text='Reiniciar', width=20, command= self.reiniciar_simulacion).grid(column=0, row=22, padx=5, pady=5)
        CTkButton(self.frameComandos, text='Finalizar', width=20, command= self.finalizar_aplicacion).grid(column=2, row=22, padx=5, pady=5)
        self.labelStatus = CTkLabel(self.frameComandos, text='')
        self.labelStatus.grid(column=0, row= 24, columnspan=2)

    # Crear la pestaña de simulación
    def crear_tab_simulacion(self):
        CTkLabel(self.tabview.tab("Simulación"), text='PARÁMETROS DEL SISTEMA', font=('Verdana', 14, 'bold')).grid(padx=10, pady=10, row=0, column=0, columnspan=3)

        self.entradaKp = self.crear_parametro_input(self.tabview.tab("Simulación"), 'Kp', self.Kp, 1, self.actualizar_kp)
        self.entradaTaup = self.crear_parametro_input(self.tabview.tab("Simulación"), 'Tau', self.taup, 2, self.actualizar_taup)
        self.entradaTd = self.crear_parametro_input(self.tabview.tab("Simulación"), 'td', self.td, 3, self.actualizar_td)

        CTkButton(self.tabview.tab("Simulación"), text='Iniciar Simulación', width=200, command=self.iniciar_simulacion, fg_color='green').grid(padx=10, pady=10, row=4, column=0, columnspan=2)

        CTkLabel(self.tabview.tab("Simulación"), text='Velocidad de simulación:').grid(column=0, row=18)
        self.scaleVelocidad = CTkSlider(self.tabview.tab("Simulación"), from_=0, to=100, number_of_steps=10, command=self.actualizar_velocidad)
        self.scaleVelocidad.grid(padx=10, pady=10, row=19, column=0, columnspan=2)
        self.scaleVelocidad.set(self.tVel)

        self.simularRuido = BooleanVar(value=True)
        self.checkSimularRuido = CTkSwitch(self.tabview.tab("Simulación"), text='Simulación de Señal Ruidosa',
                                            variable=self.simularRuido, command=self.actualizar_estado_ruido)
        self.checkSimularRuido.grid(padx=10, pady=10, row=20, column=0, columnspan=2)

    # Crear la pestaña del controlador
    def crear_tab_controlador(self):
        self.entradaSetPoitn = self.crear_parametro_input(self.tabview.tab("Controlador"), 'Set point', self.yspActual, 7, self.actualizar_sp)
        CTkButton(self.tabview.tab("Controlador"), text='Actualizar SP', width=20, command=self.actualizar_sp).grid(padx=10, pady=10, row=8, column=0, columnspan=2)
        
        CTkLabel(self.tabview.tab("Controlador"), text='GANANCIAS DEL CONTROLADOR', font=('Verdana',14, 'bold')).grid(padx=10, pady=10, row=10, column=0, columnspan=2)
        
        self.entradaKc = self.crear_parametro_input(self.tabview.tab("Controlador"), 'Kc', self.Kc, 11, self.actualizar_kc)
        self.entradaKi = self.crear_parametro_input(self.tabview.tab("Controlador"), 'Ki', self.Ki, 13, self.actualizar_ki)
        self.entradaKd = self.crear_parametro_input(self.tabview.tab("Controlador"), 'Kd', self.Kd, 14, self.actualizar_kd)

        CTkButton(self.tabview.tab("Controlador"), text='Actualizar Ganancias', width=20, command=self.actualizar_ganancias).grid(padx=10, pady=10, row=15, column=0, columnspan=2)

        self.controlAutomatico = BooleanVar(value=True)
        self.checkControlAutomatico = CTkSwitch(self.tabview.tab("Controlador"), text='Control Automático Activo',
                                            variable=self.controlAutomatico, command=self.actualizar_estado_control)
        self.checkControlAutomatico.grid(padx=10, pady=10, row=16, column=0, columnspan=2)

        self.labelCO = CTkLabel(self.tabview.tab("Controlador"), text='CO: ')
        self.entradaCO = CTkEntry(self.tabview.tab("Controlador"), width=100)
        self.entradaCO.bind('<Return>', self.actualizar_co)

    # Crear la pestaña de exportación
    def crear_tab_exportado(self):
        CTkButton(self.tabview.tab("Exportado"), text='Exportar', width=20, command= self.exportar_datos).grid(column=0, row=4, padx=5, pady=5, columnspan=2)

        self.formatoExportado = StringVar(value='xlsx')
        CTkLabel(self.tabview.tab("Exportado"), text="Formato de exportado:").grid(row=0, column=0, columnspan=1, padx=10, pady=10, sticky="")
        CTkRadioButton(self.tabview.tab("Exportado"), variable=self.formatoExportado, value='xlsx', text='xlsx').grid(row=2, column=0, pady=10, padx=20)
        CTkRadioButton(self.tabview.tab("Exportado"), variable=self.formatoExportado, value='csv', text='csv').grid(row=2, column=1, pady=10, padx=20)

    # Función para crear inputs de parámetros en la GUI
    def crear_parametro_input(self, parent, label, def_value, row, command):
        CTkLabel(parent, text=f'{label}: ').grid(pady=5, row=row, column=0)
        entrada = CTkEntry(parent, width=100)
        entrada.insert(0, str(def_value))
        entrada.grid(padx=5, row=row, column=1)
        entrada.bind('<Return>', command)
        return entrada

    # Actualizaciones de parámetros
    def actualizar_kp(self, event=None):
        """Actualiza el valor de Kp basado en la entrada del usuario."""
        try:
            self.Kp = float(self.entradaKp.get())
            self.cambiosParametros = True
        except ValueError:
            showerror("Error", "Ingrese un valor numérico válido para Kp.")
            self.entradaKp.delete(0,"end")
            self.entradaKp.insert(0, str(self.Kp))

    def actualizar_taup(self, event=None):
        """Actualiza el valor de Taup basado en la entrada del usuario."""
        try:
            self.taup = float(self.entradaTaup.get())
            self.cambiosParametros = True
        except ValueError:
            showerror("Error", "Ingrese un valor numérico válido para Taup.")
            self.entradaTaup.delete(0,"end")
            self.entradaTaup.insert(0, str(self.taup))

    def actualizar_td(self, event=None):
        """Actualiza el valor de Td basado en la entrada del usuario."""
        try:
            self.td = float(self.entradaTd.get())
            self.cambiosParametros = True
        except ValueError:
            showerror("Error", "Ingrese un valor numérico válido para Td.")
            self.entradaTd.delete(0,"end")
            self.entradaTd.insert(0, str(self.td))

    def actualizar_sp(self, event=None):
        try:
            self.yspActual = float(self.entradaSetPoitn.get())
            self.tstep = self.tActual
            self.co0 = self.co[-1]
            self.y0 = self.y[-1]
        except:
            showerror("Error", "Ingrese un valor numérico válido para el set point.")
            self.entradaSetPoitn.delete(0,"end")
            self.entradaSetPoitn.insert(0, str(self.yspActual))

    def actualizar_kc(self, event=None):
        """Actualiza el valor de Kc basado en la entrada del usuario."""
        try:
            self.Kc = float(self.entradaKc.get())
            self.cambiosParametros = True
        except:
            showerror("Error", "Ingrese un valor numérico válido para Kc.")
            self.entradaKc.delete(0,"end")
            self.entradaKc.insert(0, str(self.Kc))

    def actualizar_ki(self, event=None):
        """Actualiza el valor de Ki basado en la entrada del usuario."""
        try:
            self.Ki = float(self.entradaKi.get())
            self.cambiosParametros = True
        except:
            showerror("Error", "Ingrese un valor numérico válido para Ki.")
            self.entradaKi.delete(0,"end")
            self.entradaKi.insert(0, str(self.Ki))

    def actualizar_kd(self, event=None):
        """Actualiza el valor de Kd basado en la entrada del usuario."""
        try:
            self.Kd = float(self.entradaKd.get())
            self.cambiosParametros = True
        except:
            showerror("Error", "Ingrese un valor numérico válido para Kd.")
            self.entradaKd.delete(0,"end")
            self.entradaKd.insert(0, str(self.Kd))

    def actualizar_ganancias(self):
        self.actualizar_kc()
        self.actualizar_ki()
        self.actualizar_kd()

    def actualizar_estado_control(self):
        self.controlAutomaticoEncendido = self.controlAutomatico.get()
        if self.controlAutomaticoEncendido:
            self.labelCO.grid_forget()
            self.entradaCO.grid_forget()
        else:
            self.labelCO.grid(pady=5, row=17, column=0)
            self.entradaCO.grid(padx=5, row=17, column=1)
            self.entradaCO.delete(0, "end")
            self.coActual = self.co[-1]
            self.entradaCO.insert(0, str(round(self.coActual,1)))

    def actualizar_co(self, event=None):
        """Actualiza el valor de CO basado en la entrada del usuario."""
        try:
            self.coActual = float(self.entradaCO.get())
            self.tstep = self.tActual
            self.co0 = self.co[-1]
            self.y0 = self.y[-1]
        except:
            showerror("Error", "Ingrese un valor numérico válido para CO.")
            self.entradaCO.delete(0,"end")
            self.entradaCO.insert(0, str(self.coActual))

    def guardar_configuracion(self):
        """Guarda los cambios de configuración en el archivo JSON."""
        with open(self.CONFIG_FILE, 'w') as file:
            json.dump(self.configuracion, file, indent=4)

    def actualizar_velocidad(self, event=None):
        nuevaVelocidad = round(self.scaleVelocidad.get(), -1)
        if nuevaVelocidad<1:
            self.scaleVelocidad.set(1)
            self.tVel = 1
        else:
            self.scaleVelocidad.set(nuevaVelocidad)
            self.tVel = int(nuevaVelocidad)

    def actualizar_estado_ruido(self):
        self.ruidoSenalEncendido = self.simularRuido.get()

    # Iniciar simulación
    def iniciar_simulacion(self):
        """Inicia la simulación del sistema."""

        self.estadoSimulacion = True
        try:
            if float(self.entradaKp.get()) != self.Kp:
                self.actualizar_kp()
            if float(self.entradaTaup.get()) != self.taup:
                self.actualizar_taup()
            if float(self.entradaTd.get()) != self.td:
                self.actualizar_td()

            self.entradaKp.configure(state='disabled')
            self.entradaTaup.configure(state='disabled')
            self.entradaTd.configure(state='disabled')
            self.simulacion_pid()
            CTkButton(self.tabview.tab("Simulación"), text='Detener Simulación', width=200, command=self.detener_simulacion, fg_color='red').grid(padx=10, pady=10, row=4, column=0, columnspan=2)
        except ValueError:
            showerror("Error", "Ingrese un valor numérico válido para Kp, Tau y td.")

    # Detener los cálculos de la simulación
    def detener_simulacion(self):
        self.estadoSimulacion = False
        self.entradaKp.configure(state='normal')
        self.entradaTaup.configure(state='normal')
        self.entradaTd.configure(state='normal')
        CTkButton(self.tabview.tab("Simulación"), text='Iniciar Simulación', width=200, command=self.iniciar_simulacion, fg_color='green').grid(padx=10, pady=10, row=4, column=0, columnspan=2)

    # Reiniciar los cálculos de la simulación
    def reiniciar_simulacion(self):
        self.tActual = 0
        self.y = [self.y[-1]]
        self.co = [self.co[-1]]
        self.ysp = [self.ysp[-1]]
        self.t = [self.tActual]

        self.actualizar_grafica()

    # Definición del modelo FOPDT
    def fopdt(self, t, y, co):
        u = 0 if t < self.td + self.tstep else 1
        dydt = -(y - self.y0) / self.taup + self.Kp / self.taup * (u - self.u0) * (co - self.co0)
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

                try:
                    coAtrasado = self.co[-int(self.td/self.Ts)]  # Control retrasado
                except:
                    coAtrasado = self.co0

                # # Usar odeint
                # def fopdt_odeint(y, t, co):
                #     return self.fopdt(t, y, co)  # La función se reordena según odeint
                # sol = odeint(fopdt_odeint,self.y[-1],ts,args=tuple([coAtrasado]))
                # self.y.append(float(sol[-1][0])*np.random.normal(self.mean,np.sqrt(self.variance*self.ruidoSenalEncendido)))

                # Usar solve_ivp
                sol = solve_ivp(self.fopdt, ts, [self.y[-1]], method='RK45', t_eval=[self.tActual], args=tuple([coAtrasado]))
                self.y.append(float(sol.y[0][-1]) * np.random.normal(self.mean, np.sqrt(self.variance * self.ruidoSenalEncendido)))

                if self.controlAutomaticoEncendido:
                    # PID
                    self.Ek2 = self.Ek1
                    self.Ek1 = self.Ek
                    self.Ek = self.ysp[-1] - self.y[-1]
                    q0 = self.Kc + self.Ts * self.Ki / 2 + self.Kd / self.Ts
                    q1 = self.Kc - self.Ts * self.Ki / 2 + 2 * self.Kd / self.Ts
                    q2 = self.Kd / self.Ts
                    deltaCO = q0 * self.Ek - q1 * self.Ek1 + q2 * self.Ek2
                    nuevoCO = self.co[-1] + deltaCO
                    self.co.append(max(0, min(100, nuevoCO)))  # Limitar entre 0 y 100
                else:
                    self.co.append(self.coActual)

            self.actualizar_grafica()

            if self.estadoSimulacion:
                self.ventana.after(100, self.simulacion_pid)  # Actualización de la gráfica

        except Exception as e:
            showerror("Error", f"Error durante la simulación: {e}")

    # Actualizar la gráfica en tiempo real
    def actualizar_grafica(self):
        """Actualiza la gráfica de tendencia con los nuevos valores."""
        self.ax.cla()
        self.twax.cla()

        self.ax.grid(axis='x', color='gray', linestyle='dashed')
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.yaxis.set_major_locator(MultipleLocator(1))
        self.ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        self.ax.xaxis.grid(which='minor', linestyle='dotted', color='gray')
        self.ax.tick_params(direction='out', colors='w', grid_color='w', grid_alpha=0.3)
        self.twax.yaxis.set_minor_locator(AutoMinorLocator(10))

        self.ax.plot(self.t, self.y, 'b', label='Y', linestyle='solid')
        self.ax.plot(self.t, self.ysp, color ='r', label='Y_sp', linestyle='solid')
        self.twax.plot(self.t, self.co, 'purple', label='CO', linestyle='solid')

        self.ax.set_xlabel("t [s]", color='black')
        self.ax.set_ylabel("y", color='blue')
        self.twax.set_ylabel('CO [%]', color='purple')

        if self.tActual<=self.tminGrafica:
            self.ax.set_xlim(0, self.tminGrafica)
        else:
            self.ax.set_xlim(self.tActual-self.tminGrafica, self.tActual)

        self.ax.set_ylim(np.amin([round(np.amin(self.y[-self.nDatosGrafica:])), round(np.amin(self.ysp[-self.nDatosGrafica:]))])*.95,
                np.amax([round(np.amax(self.y[-self.nDatosGrafica:])), round(np.amax(self.ysp[-self.nDatosGrafica:])), 1])*1.05) # ajusta el rango del eje y principal
        self.twax.set_ylim(0 if np.amin(self.co[-self.nDatosGrafica:])<0 else round(np.amin(self.co[-self.nDatosGrafica:]))*0.95,
                1 if np.amax(self.co[-self.nDatosGrafica:])<1 else round(np.amax(self.co[-self.nDatosGrafica:]))*1.05) # ajusta el rango del eje y secundari

        self.canvas.draw()

    # Exportar datos de la tendencia a CSV/XLSX
    def exportar_datos(self):
        ahora = datetime.datetime.now()
        nombreArchivo = f'data_{ahora}'.replace(':','_')

        datos = {
            't': self.t,
            'CO': self.co,
            'y': self.y,
            'ysp': self.ysp
        }

        df = DataFrame(datos)

        formato = self.formatoExportado.get()

        # exportar datos a excel
        if formato=='xlsx':
            df.to_excel(nombreArchivo + '.xlsx', index=False)

        # exportar datos a csv
        if formato=='csv':
            df.to_csv(nombreArchivo + '.csv', decimal=',', sep=';', index=False)

        self.labelStatus.configure(text= f'Exportado {ahora}') # mensaje de confirmación del exportado

    # Función para finalizar la aplicación
    def finalizar_aplicacion(self):
        """Finaliza la aplicación preguntando al usuario."""
        if askyesno(message='¿Desea salir del simulador?', title='Simulador de Lazos de Control by OF'):
            if self.cambiosParametros:
                if askyesno(message='¿Desea guardar la confirugación actual?', title='Simulador de Lazos de Control by OF'):
                    self.configuracion['Kp'] = float(self.entradaKp.get())
                    self.configuracion['taup'] = float(self.entradaTaup.get())
                    self.configuracion['td'] = float(self.entradaTd.get())
                    self.configuracion['Kc'] = float(self.entradaKc.get())
                    self.configuracion['Ki'] = float(self.entradaKi.get())
                    self.configuracion['Kd'] = float(self.entradaKd.get())

                    self.guardar_configuracion()  

            self.ventana.quit()
            # self.ventana.destroy()

    # Ejecutar la aplicación
    def ejecutar(self):
        """Inicia el loop principal de la interfaz."""
        self.ventana.mainloop()


# Bloque de ejecución principal
if __name__ == '__main__':
    simulador = SimuladorControlador()
    simulador.ejecutar()
