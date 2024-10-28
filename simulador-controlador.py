"""
Nombre: simulador-controlador.py
Autor: Oscar Franco
Versión: 5 (2023-01-28)
Descripción: Aplicacióin para simular el comportamiento de un sistema según su función de transferencia
            en lazo abierto o al aplicar un controlador PID

"""

import customtkinter as ctk
from customtkinter import *
from tkinter.messagebox import showerror, askyesno
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from numpy import amax, amin
from scipy.integrate import odeint
from random import gauss
from math import sqrt
import datetime
import json
from pandas import DataFrame

# cargue de datos desde config.json
try:
    with open('config.json') as file:
        configuracion = json.load(file)
except:
    # crea la configuración por defecto si no encuentra el archivo config.json
    configJson = """{   
        "mean": 1.0,
        "variance": 1e-8,
        "tVel": 10,
        "ruidoSenalEncendido": 1,
        "Ts": 0.1,
        "controlAutomaticoEncendido": 1,
        "tminGrafica": 120,
        "estadoSimulacion": 1,
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
    }"""

    with open('config.json', 'w+') as file:
        file.write(configJson)

    with open('config.json') as file:
        configuracion = json.load(file)

# parámetros del randomizador de señales
mean = configuracion['mean']
variance = configuracion['variance']

tVel = configuracion['tVel'] # velocidad de simulación
Ts = configuracion['Ts'] # tiempo de muestreo
controlAutomaticoEncendido = bool(configuracion['controlAutomaticoEncendido']) # estado del controlador (encendido/apagado)
tminGrafica = configuracion['tminGrafica'] # ventana de tiempo visible en la gráfica de tendencia
estadoSimulacion = bool(configuracion['estadoSimulacion']) # estado de calculos de la simulación
ruidoSenalEncendido = bool(configuracion['ruidoSenalEncendido']) # estado del cálculo del ruido en la señal de la variable medida

Kp = configuracion['Kp'] # ganancia del proceso
taup = configuracion['taup'] # constante de tiempo del proceso
td = configuracion['td'] # tiempo muerto del proceso

Kc = configuracion['Kc'] # ganancia proporcional del controlador
Ki = configuracion['Ki'] # ganancia integral del controlador
Kd = configuracion['Kd'] # ganancia derivativa del controlador

# condiciones inciales
t0 = configuracion['t0']
y0 = configuracion['y0']
co0 = configuracion['co0']
u0 = configuracion['u0']
ysp0 = configuracion['ysp0']

cambiosParametros = False

# inicializamos las variables de error
Ek2 = 0
Ek1 = 0
Ek = 0

# inicialización de los datos actuales
tstep = 0
tActual = t0
yActual = y0
coActual = co0
yspActual = ysp0

# inicialización de las listas de datos de las tendencias
t = [0]
y = [y0]
co = [co0]
ysp = [ysp0]
nDatosGrafica = round(tminGrafica/Ts)

# modelo a simular
def fopdt(y,t,co):

    u = 0 if t<td+tstep else 1

    # calcular la derivada
    dydt = -(y-y0)/taup + Kp/taup * (u-u0) * (co-co0)

    return dydt

def restaurar_parametros_sistema():
    entradaKp.delete(0,"end")
    entradaKp.insert(0, str(Kp))
    entradaTaup.delete(0,"end")
    entradaTaup.insert(0, str(taup))
    entradaTd.delete(0,"end")
    entradaTd.insert(0, str(td))

# arranca la simulación
def iniciar_simulacion():
    global estadoSimulacion
    estadoSimulacion = True
    if float(entradaKp.get()) != Kp:
        actualizar_kp()
    if float(entradaTaup.get()) != taup:
        actualizar_taup()
    if float(entradaTd.get()) != td:
        actualizar_td()
    entradaKp.configure(state='disabled')
    entradaTaup.configure(state='disabled')
    entradaTd.configure(state='disabled')
    simular_sistema()
    CTkButton(tabview.tab("Simulación"), text='Detener Simulación', width=200, command=detener_simulacion, fg_color='red').grid(padx=10, pady=10, row=4, column=0, columnspan=2)

# detiene los cálculos de la simulación
def detener_simulacion():
    global estadoSimulacion
    estadoSimulacion = False
    entradaKp.configure(state='normal')
    entradaTaup.configure(state='normal')
    entradaTd.configure(state='normal')
    CTkButton(tabview.tab("Simulación"), text='Iniciar Simulación', width=200, command=iniciar_simulacion, fg_color='green').grid(padx=10, pady=10, row=4, column=0, columnspan=2)

# actualiza el valor de Kp desde el campo de entrada
def actualizar_kp(event=None):
    global Kp, cambiosParametros
    kpEntry = entradaKp.get()
    try:
        Kp = float(kpEntry)
        cambiosParametros = True
    except:
        showerror(message=f'El valor Kp={kpEntry} no es un número válido', title='Simulador de Lazos de Control by OF')
        entradaKp.delete(0,"end")
        entradaKp.insert(0, str(Kp))

# actualiza el valor de taup desde el campo de entrada
def actualizar_taup(event=None):
    global taup, cambiosParametros
    taupEntry = entradaTaup.get()
    try:
        taup = float(taupEntry)
        cambiosParametros = True
    except:
        showerror(message=f'El valor tau={taupEntry} no es un número válido', title='Simulador de Lazos de Control by OF')
        entradaTaup.delete(0,"end")
        entradaTaup.insert(0, str(taup))

# actualiza el valor de td desde el campo de entrada
def actualizar_td(event=None):
    global td, cambiosParametros
    tdEntry = entradaTd.get()
    try:
        td = float(tdEntry)
        cambiosParametros = True
    except:
        showerror(message=f'El valor Kp={tdEntry} no es un número válido', title='Simulador de Lazos de Control by OF')
        entradaTd.delete(0,"end")
        entradaTd.insert(0, str(td))

# actualiza el valor de ysp desde el campo de entrada
def actualizar_sp(event=None):
    global yspActual, tstep, co0, y0
    yspEntry = entradaSetPoitn.get()
    try:
        yspActual = float(yspEntry)
        tstep = tActual
        co0 = co[-1]
        y0 = y[-1]
    except:
        showerror(message=f'El valor {yspEntry} no es un número válido', title='Simulador de Lazos de Control by OF')
        entradaSetPoitn.delete(0,"end")
        entradaSetPoitn.insert(0, str(yspActual))

# actualiza el valor de Kc desde el campo de entrada
def actualizar_kc(event=None):
    global Kc, cambiosParametros
    kcEntry = entradaKc.get()
    try:
        Kc = float(kcEntry)
        cambiosParametros = True
    except:
        showerror(message=f'El valor Kc={kcEntry} no es un número válido', title='Simulador de Lazos de Control by OF')
        entradaKc.delete(0,"end")
        entradaKc.insert(0, str(Kc))

# actualiza el valor de Ki desde el campo de entrada
def actualizar_ki(event=None):
    global Ki, cambiosParametros
    kiEntry = entradaKi.get()
    try:
        Ki = float(kiEntry)
        cambiosParametros = True
    except:
        showerror(message=f'El valor Ki={kiEntry} no es un número válido', title='Simulador de Lazos de Control by OF')
        entradaKi.delete(0,"end")
        entradaKi.insert(0, str(Ki))

# actualiza el valor de Kd desde el campo de entrada
def actualizar_kd(event=None):
    global Kd, cambiosParametros
    kdEntry = entradaKd.get()
    try:
        Kd = float(kdEntry)
        cambiosParametros = True
    except:
        showerror(message=f'El valor Kd={kdEntry} no es un número válido', title='Simulador de Lazos de Control by OF')
        entradaKd.delete(0,"end")
        entradaKd.insert(0, str(Kd))

# actualiza todas las ganancias desde los campos de entrada
def actualizar_ganancias():
    actualizar_kc()
    actualizar_ki()
    actualizar_kd()

# enciende/apaga el controlador
def actualizar_estado_control():
    global controlAutomaticoEncendido, coActual
    controlAutomaticoEncendido = controlAutomatico.get()
    if controlAutomaticoEncendido:
        labelCO.grid_forget()
        entradaCO.grid_forget()
    else:
        labelCO.grid(pady=5, row=17, column=0)
        entradaCO.grid(padx=5, row=17, column=1)
        entradaCO.delete(0, "end")
        coActual = co[-1]
        entradaCO.insert(0, str(round(coActual,4)))

# cambiar la salida del controlador cuando se ejecuta en manual
def actualizar_co(event=None):
    global coActual, tstep, co0, y0
    coEntry = entradaCO.get()
    try:
        coActual = float(coEntry)
        tstep = tActual
        co0 = co[-1]
        y0 = y[-1]
    except:
        showerror(message=f'El valor CO={coEntry} no es un número válido', title='Simulador de Lazos de Control by OF')
        entradaCO.delete(0,"end")
        entradaCO.insert(0, str(coActual))

# cambia la velocidad de ejecución de la simulación
def actualizar_velocidad(event=None):
    global tVel
    nuevaVelocidad = round(scaleVelocidad.get(), -1)
    if nuevaVelocidad<1:
        scaleVelocidad.set(1)
        tVel = 1
    else:
        scaleVelocidad.set(nuevaVelocidad)
        tVel = int(nuevaVelocidad)

def actualizar_estado_ruido():
    global ruidoSenalEncendido
    ruidoSenalEncendido = simularRuido.get()
        
# realiza una iteración de la simulación del modelo
def simular_sistema():
    global tActual, t, y, ysp, Ek, Ek1, Ek2, co

    for i in range(tVel):

        tActual = t[-1] + Ts
        t.append(tActual)

        ts = [t[-2],t[-1]]

        ysp.append(yspActual)

        try:
            coAtrasado = co[-int(td/Ts)]
        except:
            coAtrasado = co0
        y1 = odeint(fopdt,y[-1],ts,args=tuple([coAtrasado]))
        y.append(float(y1[-1])*gauss(mean,sqrt(variance*ruidoSenalEncendido)))

        if controlAutomaticoEncendido:
            ## PID
            Ek2 = Ek1
            Ek1 = Ek
            Ek = ysp[-1] - y[-1]
            q0 = Kc + Ts*Ki/2 + Kd/Ts
            q1 = Kc - Ts*Ki/2 + 2*Kd/Ts
            q2 = Kd/Ts
            deltaCO = q0*Ek - q1*Ek1 + q2*Ek2
            if co[-1] + deltaCO < 0:
                co.append(0)
            elif co[-1] + deltaCO > 100:
                co.append(100)           
            else:
                co.append(co[-1] + deltaCO)
        else:
            co.append(coActual)

    if tActual<=tminGrafica:
        ax.set_xlim(0, tminGrafica)
    else:
        ax.set_xlim(tActual-tminGrafica, tActual)


    ax.set_ylim(amin([round(amin(y[-nDatosGrafica:])), round(amin(ysp[-nDatosGrafica:]))])*.95,
            amax([round(amax(y[-nDatosGrafica:])), round(amax(ysp[-nDatosGrafica:])), 1])*1.05) # ajusta el rango del eje y principal
    twax.set_ylim(0 if amin(co[-nDatosGrafica:])<0 else round(amin(co[-nDatosGrafica:]))*0.95,
            1 if amax(co[-nDatosGrafica:])<1 else round(amax(co[-nDatosGrafica:]))*1.05) # ajusta el rango del eje y secundario

    lineCO, = twax.plot(t, co, color ='purple', linestyle='solid') # crea la línea con los datos
    lineSP, = ax.plot(t, ysp, color ='r', linestyle='solid') # crea la línea con los datos
    lineY, = ax.plot(t, y, color ='b', linestyle='solid') # crea la línea con los datos
        
    canvas.draw() # graficar

    lineCO.set_ydata([1e6]*len(t))
    lineSP.set_ydata([1e6]*len(t))
    lineY.set_ydata([1e6]*len(t))

    if estadoSimulacion:
        ventana.after(100, simular_sistema) # tiempo de actualización de la gráfica

# exportar datos de la tendencia a CSV
def  exportar_datos():
    global t, co, y, ysp
    
    ahora = datetime.datetime.now()
    nombreArchivo = f'data_{ahora}'.replace(':','_')

    datos = {
        't [s]': t,
        'CO [%]': co,
        'y': y,
        'ysp': ysp
    }

    df = DataFrame(datos)

    formato = formatoExportado.get()

    # exportar datos a excel
    if formato=='xlsx':
        df.to_excel(nombreArchivo + '.xlsx', index=False)

    # exportar datos a csv
    if formato=='csv':
        df.to_csv(nombreArchivo + '.csv', decimal=',', sep=';', index=False)

    labelStatus.configure(text= f'Exportado {ahora}') # mensaje de confirmación del exportado

def finalizar_aplicacion():
    global configuracion

    salir = askyesno(message='¿Desea salir del simulador?', title='Simulador de Lazos de Control by OF')
    if salir:
        if cambiosParametros:

            seleccion = askyesno(message='¿Desea guardar la confirugación actual?', title='Simulador de Lazos de Control by OF')

            if seleccion:
                configuracion['Kp'] = float(entradaKp.get())
                configuracion['taup'] = float(entradaTaup.get())
                configuracion['td'] = float(entradaTd.get())
                configuracion['Kc'] = float(entradaKc.get())
                configuracion['Ki'] = float(entradaKi.get())
                configuracion['Kd'] = float(entradaKd.get())

                configJson = json.dumps(configuracion)
                with open('config.json', 'w+') as file:
                    file.write(configJson)    

        ventana.quit()

if __name__ == '__main__':

    ############## GRÁFICA DE TENDENCIA

    fig, ax = plt.subplots(figsize=(8,5),facecolor='grey') # creación de figura
    plt.title("Gráfica de Tendencia",color='black',size=16, family="Arial") # asigna el título de la gráfica

    ax.set_facecolor('black') # asignación del fondo de la gráfica

    ax.axhline(linewidth=2, color='w') # ajuste de propiedades del eje x
    ax.set_xlabel("t [s]", color='black')
    ax.axvline(linewidth=2, color='w') # ajuste de propiedades del eje y
    ax.set_ylabel("y", color='blue')
    ax.grid(axis = 'x', color = 'gray', linestyle = 'dashed')
    ax.yaxis.set_minor_locator(AutoMinorLocator(5)) # subdivisiones de eje automáticas
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(AutoMinorLocator(10)) # subdivisiones de eje automáticas
    ax.xaxis.grid(which='minor', linestyle='dotted', color='gray')
    ax.tick_params(direction='out', colors='w', grid_color='w', grid_alpha=0.3) # parámetros de las marcaciones de los ejes

    twax = ax.twinx() # creación del eje y secundario
    twax.set_ylabel('CO [%]',color='purple')
    twax.tick_params(direction='out', length=6, width=1, colors='purple', grid_color='w', grid_alpha=0.5) # parámetros de las marcaciones de los ejes
    twax.set_ylim(0, 100)
    twax.yaxis.set_minor_locator(AutoMinorLocator(10)) # subdivisiones de eje automáticas

    ############## GUI

    ctk.set_appearance_mode("System")  # Modes: system (default), light, dark
    ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

    ventana = CTk()
    ventana.title('Simulador de Lazos de Control by OF')
    ventana.resizable(height=0, width=0)
    ventana.protocol('WM_DELETE_WINDOW', finalizar_aplicacion)

    # frame de gráfica
    frameGrafico = CTkFrame(ventana)
    frameGrafico.grid(column=0,row=0, columnspan=4)
    canvas = FigureCanvasTkAgg(fig, master = frameGrafico)
    canvas.get_tk_widget().grid(column=0, row=0, padx=5, pady =5, sticky='nw')

    # frame de comandos con tabs de opciones
    frameComandos = CTkFrame(ventana)
    frameComandos.grid(column=4,row=0)
    tabview = CTkTabview(frameComandos)
    tabview.grid(row=0, column=0, columnspan=3, sticky='n')
    tabview.add("Simulación")
    tabview.add("Controlador")
    tabview.add("Exportado")

    # tab de simulación
    CTkLabel(tabview.tab("Simulación"), text='PARÁMETROS DEL SISTEMA', font=('Verdana',14, 'bold')).grid(padx=10, pady=10, row=0, column=0, columnspan=3)
    
    CTkLabel(tabview.tab("Simulación"), text='Kp: ').grid(pady=5, row=1, column=0)
    entradaKp = CTkEntry(tabview.tab("Simulación"), width=100)
    entradaKp.insert(0, str(Kp))
    entradaKp.grid(padx=5, row=1, column=1)
    entradaKp.bind('<Return>', actualizar_kp)

    CTkLabel(tabview.tab("Simulación"), text='Tau: ').grid(pady=5, row=2, column=0)
    entradaTaup = CTkEntry(tabview.tab("Simulación"), width=100)
    entradaTaup.insert(0, str(taup))
    entradaTaup.grid(padx=5, row=2, column=1)
    entradaTaup.bind('<Return>', actualizar_taup)

    CTkLabel(tabview.tab("Simulación"), text='td: ').grid(pady=5, row=3, column=0)
    entradaTd = CTkEntry(tabview.tab("Simulación"), width=100)
    entradaTd.insert(0, str(td))
    entradaTd.grid(padx=5, row=3, column=1)
    entradaTd.bind('<Return>', actualizar_td)

    CTkButton(tabview.tab("Simulación"), text='Iniciar Simulación', width=200, command=iniciar_simulacion, fg_color='green').grid(padx=10, pady=10, row=4, column=0, columnspan=2)

    CTkLabel(tabview.tab("Simulación"), text='Velocidad de simulación:').grid(column=0, row=18)
    scaleVelocidad = CTkSlider(tabview.tab("Simulación"), from_=0, to=100, number_of_steps=10, command=actualizar_velocidad)
    scaleVelocidad.grid(padx=10, pady=10, row=19, column=0, columnspan=2)
    scaleVelocidad.set(tVel)

    simularRuido = BooleanVar(value=True)
    checkSimularRuido = CTkSwitch(tabview.tab("Simulación"), text='Simulación de Señal Ruidosa',
                                        variable=simularRuido, command=actualizar_estado_ruido)
    checkSimularRuido.grid(padx=10, pady=10, row=20, column=0, columnspan=2)

    # tab de comandos del controlador
    CTkLabel(tabview.tab("Controlador"), text='Set point:').grid(pady=5, row=7, column=0)
    entradaSetPoitn = CTkEntry(tabview.tab("Controlador"), width=100)
    entradaSetPoitn.insert(0, str(yspActual))
    entradaSetPoitn.grid(padx=5, row=7, column=1)
    entradaSetPoitn.bind('<Return>', actualizar_sp)
    CTkButton(tabview.tab("Controlador"), text='Actualizar SP', width=20, command=actualizar_sp).grid(padx=10, pady=10, row=8, column=0, columnspan=2)
    
    CTkLabel(tabview.tab("Controlador"), text='GANANCIAS DEL CONTROLADOR', font=('Verdana',14, 'bold')).grid(padx=10, pady=10, row=10, column=0, columnspan=2)
    
    CTkLabel(tabview.tab("Controlador"), text='Kc: ').grid(pady=5, row=11, column=0)
    entradaKc = CTkEntry(tabview.tab("Controlador"), width=100)
    entradaKc.insert(0, str(Kc))
    entradaKc.grid(padx=5, row=11, column=1)
    entradaKc.bind('<Return>', actualizar_kc)

    CTkLabel(tabview.tab("Controlador"), text='Ki: ').grid(pady=5, row=13, column=0)
    entradaKi = CTkEntry(tabview.tab("Controlador"), width=100)
    entradaKi.insert(0, str(Ki))
    entradaKi.grid(padx=5, row=13, column=1)
    entradaKi.bind('<Return>', actualizar_ki)

    CTkLabel(tabview.tab("Controlador"), text='Kd: ').grid(pady=5, row=14, column=0)
    entradaKd = CTkEntry(tabview.tab("Controlador"), width=100)
    entradaKd.insert(0, str(Kd))
    entradaKd.grid(padx=5, row=14, column=1)
    entradaKd.bind('<Return>', actualizar_kd)

    CTkButton(tabview.tab("Controlador"), text='Actualizar Ganancias', width=20, command=actualizar_ganancias).grid(padx=10, pady=10, row=15, column=0, columnspan=2)

    controlAutomatico = BooleanVar(value=True)
    checkControlAutomatico = CTkSwitch(tabview.tab("Controlador"), text='Control Automático Activo',
                                        variable=controlAutomatico, command=actualizar_estado_control)
    checkControlAutomatico.grid(padx=10, pady=10, row=16, column=0, columnspan=2)

    labelCO = CTkLabel(tabview.tab("Controlador"), text='CO: ')
    entradaCO = CTkEntry(tabview.tab("Controlador"), width=100)
    entradaCO.bind('<Return>', actualizar_co)

    # tab de comandos para exportar los datos de la simulación
    CTkButton(tabview.tab("Exportado"), text='Exportar', width=20, command= exportar_datos).grid(column=0, row=4, padx=5, pady=5, columnspan=2)

    formatoExportado = StringVar(value='xlsx')
    labelFormatoExportado = CTkLabel(tabview.tab("Exportado"), text="Formato de exportado:")
    labelFormatoExportado.grid(row=0, column=0, columnspan=1, padx=10, pady=10, sticky="")
    radio_button_1 = CTkRadioButton(tabview.tab("Exportado"), variable=formatoExportado, value='xlsx', text='xlsx')
    radio_button_1.grid(row=2, column=0, pady=10, padx=20)
    radio_button_2 = CTkRadioButton(tabview.tab("Exportado"), variable=formatoExportado, value='csv', text='csv')
    radio_button_2.grid(row=2, column=1, pady=10, padx=20)

    # finalización y barra de estado
    CTkButton(frameComandos, text='Finalizar', width=20, command= finalizar_aplicacion).grid(column=1, row=22, padx=5, pady=5)
    labelStatus = CTkLabel(frameComandos, text='')
    labelStatus.grid(column=0, row= 24, columnspan=2)

    ventana.mainloop()