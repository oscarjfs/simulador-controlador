from tkinter import *
from tkinter.messagebox import showerror
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from numpy import amax, amin
from scipy.integrate import odeint
from random import gauss
from math import sqrt
import datetime
import csv

# parámetros del randomizador de señales
mean = 1
variance = 2e-8

tVel = 10 # velocidad de simulación
Ts = 0.1 # tiempo de muestreo
controlAutomaticoEncendido = True # estado del controlador (encendido/apagado)
tminGrafica = 120 # ventana de tiempo visible en la gráfica de tendencia
estadoSimulacion = True # estado de calculos de la simulación

Kp = 4.5926 # ganancia del proceso
taup = 15.1423 # constante de tiempo del proceso
td = 5.0410 # tiempo muerto del proceso

Kc = 0.6058 # ganancia proporcional del controlador
Ki = 0.0606 # ganancia integral del controlador
Kd = 0.0 # ganancia derivativa del controlador

# condiciones inciales
t0 = 0
y0 = 50
co0 = 50
u0 = 0
ysp0 = 50
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

# modelo a simular
def fopdt(y,t,co):

    u = 0 if t<td+tstep else 1

    # calcular la derivada
    dydt = -(y-y0)/taup + Kp/taup * (u-u0) * (co-co0)

    return dydt

# arranca la simulación
def iniciar_simulacion():
    global estadoSimulacion
    estadoSimulacion = True
    actualizar_kp()
    actualizar_taup()
    actualizar_td()
    entradaKp.config(state='disabled')
    entradaTaup.config(state='disabled')
    entradaTd.config(state='disabled')
    simular_sistema()
    Button(frameComandos, text='Detener Simulación', width=50, command=detener_simulacion, bg='red').grid(padx=10, pady=10, row=4, column=0, columnspan=2)

# detiene los cálculos de la simulación
def detener_simulacion():
    global estadoSimulacion
    estadoSimulacion = False
    entradaKp.config(state='normal')
    entradaTaup.config(state='normal')
    entradaTd.config(state='normal')
    Button(frameComandos, text='Iniciar Simulación', width=50, command=iniciar_simulacion, bg='green').grid(padx=10, pady=10, row=4, column=0, columnspan=2)

# actualiza el valor de Kp desde el campo de entrada
def actualizar_kp(event=None):
    global Kp
    kpEntry = entradaKp.get()
    try:
        Kp = float(kpEntry)
    except:
        showerror(message=f'El valor Kp={kpEntry} no es un número válido', title='Simulador de Lazos de Control')
        entradaKp.delete(0,"end")
        entradaKp.insert(0, str(Kp))

# actualiza el valor de taup desde el campo de entrada
def actualizar_taup(event=None):
    global taup
    taupEntry = entradaTaup.get()
    try:
        taup = float(taupEntry)
    except:
        showerror(message=f'El valor tau={taupEntry} no es un número válido', title='Simulador de Lazos de Control')
        entradaTaup.delete(0,"end")
        entradaTaup.insert(0, str(taup))

# actualiza el valor de td desde el campo de entrada
def actualizar_td(event=None):
    global td
    tdEntry = entradaTd.get()
    try:
        td = float(tdEntry)
    except:
        showerror(message=f'El valor Kp={tdEntry} no es un número válido', title='Simulador de Lazos de Control')
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
        showerror(message=f'El valor {yspEntry} no es un número válido', title='Simulador de Lazos de Control')
        entradaSetPoitn.delete(0,"end")
        entradaSetPoitn.insert(0, str(yspActual))

# actualiza el valor de Kc desde el campo de entrada
def actualizar_kc(event=None):
    global Kc
    kcEntry = entradaKc.get()
    try:
        Kc = float(kcEntry)
    except:
        showerror(message=f'El valor Kc={kcEntry} no es un número válido', title='Simulador de Lazos de Control')
        entradaKc.delete(0,"end")
        entradaKc.insert(0, str(Kc))

# actualiza el valor de Ki desde el campo de entrada
def actualizar_ki(event=None):
    global Ki
    kiEntry = entradaKi.get()
    try:
        Ki = float(kiEntry)
    except:
        showerror(message=f'El valor Ki={kiEntry} no es un número válido', title='Simulador de Lazos de Control')
        entradaKi.delete(0,"end")
        entradaKi.insert(0, str(Ki))

# actualiza el valor de Kd desde el campo de entrada
def actualizar_kd(event=None):
    global Kd
    kdEntry = entradaKd.get()
    try:
        Kd = float(kdEntry)
    except:
        showerror(message=f'El valor Kd={kdEntry} no es un número válido', title='Simulador de Lazos de Control')
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
        showerror(message=f'El valor CO={coEntry} no es un número válido', title='Simulador de Lazos de Control')
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
        tVel = nuevaVelocidad
        
# realiza una iteración de la simulación del modelo
def simular_sistema():
    global tActual, t, y, ysp, co0, y0, Ek, Ek1, Ek2, co, tstep, coActual

    for i in range(tVel):

        tActual = t[-1] + Ts
        t.append(tActual)

        ts = [t[-2],t[-1]]

        ysp.append(yspActual)

        y1 = odeint(fopdt,y[-1],ts,args=tuple([co[-1]]))
        y.append(float(y1[-1])*gauss(mean,sqrt(variance)))

        if controlAutomaticoEncendido:
            ## PID
            Ek2 = Ek1
            Ek1 = Ek
            Ek = ysp[-1] - y[-1]
            deltaCO = (Kc+Ki*Ts+Kd/Ts)*Ek - (Kc+2*Kd/Ts)*Ek1 + (Kd/Ts)*Ek2
            if co[-1] + deltaCO < 0:
                co.append(0)
            else:
                co.append(co[-1] + deltaCO)
        else:
            co.append(coActual)

    if tActual<=tminGrafica:
        ax.set_xlim(0, tminGrafica)
    else:
        ax.set_xlim(tActual-tminGrafica, tActual)
    
    # ax.set_ylim(amax([round(amin(y[-tminGrafica:])), round(amin(y[-tminGrafica:])), 0])*.95,
    #         amax([round(amax(y[-tminGrafica:])), round(amax(ysp[-tminGrafica:])), 1])*1.05) # ajusta el rango del eje y principal
    ax.set_ylim(amax([round(amin(y)), round(amin(y)), 0])*.95, amax([round(amax(y)), round(amax(ysp)), 1])*1.05) # ajusta el rango del eje y principal
    twax.set_ylim(0 if amin(co)<0 else round(amin(co))*0.95, 1 if amax(co)<1 else round(amax(co))*1.05) # ajusta el rango del eje y secundario

        ############# CARGUE DE DATOS

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

    csv_file = f'data_{ahora}'.replace(':','_')
    csv_file += '.csv'

    with open(csv_file, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['Tiempo[s]', 'CO', 'y', 'y_sp'])

        for i in range(len(t)):
            fila = []
            fila.append(str(t[i]).replace('.',','))
            fila.append(str(co[i]).replace('.',','))
            fila.append(str(y[i]).replace('.',','))
            fila.append(str(ysp[i]).replace('.',','))
            spamwriter.writerow(fila)

    labelStatus.config(text= f'Exportado {csv_file}') # mensaje de confirmación del exportado

if __name__ == '__main__':

    ############## GRÁFICA DE TENDENCIA

    fig, ax = plt.subplots(figsize=(10,7),facecolor='grey') # creación de figura
    plt.title("Gráfica de Tendencia",color='black',size=16, family="Arial") # asigna el título de la gráfica

    #plt.xlim(0, 60) # ajusta el rango del eje x
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
    twax.set_ylabel('CO',color='purple')
    twax.tick_params(direction='out', length=6, width=1, colors='purple', grid_color='w', grid_alpha=0.5) # parámetros de las marcaciones de los ejes
    twax.set_ylim(0, 100)
    twax.yaxis.set_minor_locator(AutoMinorLocator(10)) # subdivisiones de eje automáticas

    ############## GUI

    ventana = Tk()
    ventana.configure(bg='gray22')
    ventana.wm_title('Simulador de Lazos de Control')

    frameGrafico = Frame(ventana, bg='gray22',bd=3)
    frameGrafico.grid(column=0,row=0, columnspan=4, rowspan=6)

    frameComandos = Frame(ventana, bg='gray22',bd=3)
    frameComandos.grid(column=4,row=0, rowspan=6)

    canvas = FigureCanvasTkAgg(fig, master = frameGrafico)
    canvas.get_tk_widget().grid(column=0, row=0, padx=5, pady =5)

    Label(frameComandos, text='PARÁMETROS DEL SISTEMA', font='Helvetica 14 bold', bg='gray22',fg='white').grid(padx=10, pady=10, row=0, column=0, columnspan=3)

    Label(frameComandos, text='Kp: ', bg='gray22',fg='white').grid(pady=5, row=1, column=0)
    entradaKp = Entry(frameComandos, width=30)
    entradaKp.insert(0, str(Kp))
    entradaKp.grid(padx=5, row=1, column=1)
    entradaKp.bind('<Return>', actualizar_kp)

    Label(frameComandos, text='Tau: ', bg='gray22', fg='white').grid(pady=5, row=2, column=0)
    entradaTaup = Entry(frameComandos, width=30)
    entradaTaup.insert(0, str(taup))
    entradaTaup.grid(padx=5, row=2, column=1)
    entradaTaup.bind('<Return>', actualizar_taup)

    Label(frameComandos, text='td: ', bg='gray22', fg='white').grid(pady=5, row=3, column=0)
    entradaTd = Entry(frameComandos, width=30)
    entradaTd.insert(0, str(td))
    entradaTd.grid(padx=5, row=3, column=1)
    entradaTd.bind('<Return>', actualizar_td)

    Button(frameComandos, text='Iniciar Simulación', width=20, command=iniciar_simulacion, bg='green').grid(padx=10, pady=10, row=4, column=0, columnspan=2)

    Label(frameComandos, text='Set point:', bg='gray22',fg='white').grid(pady=5, row=7, column=0)
    entradaSetPoitn = Entry(frameComandos, width=30)
    entradaSetPoitn.insert(0, str(yspActual))
    entradaSetPoitn.grid(padx=5, row=7, column=1)
    entradaSetPoitn.bind('<Return>', actualizar_sp)

    Button(frameComandos, text='Actualizar SP', width=20, command=actualizar_sp).grid(padx=10, pady=10, row=8, column=0, columnspan=2)

    Label(frameComandos, text='GANANCIAS DEL CONTROLADOR', font='Helvetica 14 bold', bg='gray22',fg='white').grid(padx=10, pady=10, row=10, column=0, columnspan=2)

    Label(frameComandos, text='Kc: ', bg='gray22',fg='white').grid(pady=5, row=11, column=0)
    entradaKc = Entry(frameComandos, width=30)
    entradaKc.insert(0, str(Kc))
    entradaKc.grid(padx=5, row=11, column=1)
    entradaKc.bind('<Return>', actualizar_kc)

    Label(frameComandos, text='Ki: ', bg='gray22', fg='white').grid(pady=5, row=13, column=0)
    entradaKi = Entry(frameComandos, width=30)
    entradaKi.insert(0, str(Ki))
    entradaKi.grid(padx=5, row=13, column=1)
    entradaKi.bind('<Return>', actualizar_ki)

    Label(frameComandos, text='Kd: ', bg='gray22', fg='white').grid(pady=5, row=14, column=0)
    entradaKd = Entry(frameComandos, width=30)
    entradaKd.insert(0, str(Kd))
    entradaKd.grid(padx=5, row=14, column=1)
    entradaKd.bind('<Return>', actualizar_kd)

    Button(frameComandos, text='Actualizar Ganancias', width=20, command=actualizar_ganancias).grid(padx=10, pady=10, row=15, column=0, columnspan=2)

    controlAutomatico = BooleanVar(frameComandos)
    controlAutomatico.set(True)
    checkControlAutomatico = Checkbutton(frameComandos, text='Control Automático Activo', bg="gray22", fg='white', selectcolor='gray',
                                        variable=controlAutomatico, command=actualizar_estado_control)
    checkControlAutomatico.grid(padx=10, pady=10, row=16, column=0, columnspan=2)

    labelCO = Label(frameComandos, text='CO: ', bg='gray22', fg='white')
    entradaCO = Entry(frameComandos, width=30)
    entradaCO.bind('<Return>', actualizar_co)

    scaleVelocidad = Scale(frameComandos, label='Velocidad de simulación:', length=400, bg="gray22", fg='white', 
                        from_=0, to=100, tickinterval=10, orient='horizontal', command=actualizar_velocidad)
    scaleVelocidad.grid(padx=10, pady=10, row=18, column=0, columnspan=2)
    scaleVelocidad.set(tVel)

    Button(frameComandos, text='Exportar', width=20, command= exportar_datos).grid(column=0, row=20, padx=5, pady=5)
    Button(frameComandos, text='Finalizar', width=20, command= ventana.quit).grid(column=1, row=20, padx=5, pady=5)

    labelStatus = Label(frameComandos, bg='gray22', fg='white')
    labelStatus.grid(column=0, row= 22, columnspan=2)

    ventana.mainloop()