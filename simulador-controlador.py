import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from numpy import amax, amin
from scipy.integrate import odeint
from random import gauss
from math import sqrt


# parámetros del randomizador de señales
mean = 1
variance = 2e-8

Ts = 0.1 # tiempo de muestreo
controlAutomaticoEncendido = True

Kp = 4.5926 # ganancia del proceso
taup = 15.1423 # constante de tiempo del proceso
td = 5.0450 # tiempo muerto del proceso

Kc = 0.6058 # ganancia proporcional del controlador
Ki = 0.0606 # ganancia integral del controlador
Kd = 0.0 # ganancia derivativa del controlador

tfinal = 200

# condiciones inciales
t0 = 0
y0 = 50
co0 = 50
u0 = 0
ysp0 = 50
Ek2 = 0
Ek1 = 0
Ek = 0

####### cambios en escalón en el tiempo
tstep1 = 5
yspstep1 = 52
coStep1 = 51
tstep2 = 100
yspstep2 = 50
coStep2 = 50
########################

tstep = tstep1
tActual = t0
yActual = y0
coActual = co0
yspActual = ysp0

ndatos = int(tfinal/Ts+1)
t = [0]
y = [y0]
co = [co0]
ysp = [ysp0]

def fopdt(y,t,co):

    u = 0 if t<td+tstep else 1

    # calcular la derivada
    dydt = -(y-y0)/taup + Kp/taup * (u-u0) * (co-co0)

    return dydt

# actualiza el valor de ysp desde el campo de entrada
def actualizar_sp(yspNuevo):
    global yspActual, tstep, co0, y0
    yspActual = float(yspNuevo)
    tstep = tActual
    co0 = co[-1]
    y0 = y[-1]

# cambiar la salida del controlador cuando se ejecuta en manual
def actualizar_co(coNuevo):
    global coActual, tstep, co0, y0
    coActual = float(coNuevo)
    tstep = tActual
    co0 = co[-1]
    y0 = y[-1]

# solución del modelo para cada paso o intervalo de tiempo (discretización)
for i in range(0,ndatos-1):

    tActual = t[-1] + Ts
    t.append(tActual)

    ts = [t[-2],t[-1]]

    coActual = co[-1]

    # perturbación en sp con controlador automático
    # controlAutomaticoEncendido = True
    # if tActual>=tstep1 and tActual<=tstep1+Ts:
    #     actualizar_sp(yspstep1)
    # elif tActual>=tstep2 and tActual<=tstep2+Ts:
    #     actualizar_sp(yspstep2)

    # perturbación en el elemento final de control con controlador en manual
    controlAutomaticoEncendido = False
    if tActual>=tstep1 and tActual<=tstep1+Ts:
        actualizar_co(coStep1)
    elif tActual>=tstep2 and tActual<=tstep2+Ts:
        actualizar_co(coStep2)
    
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
        if co[-1] + deltaCO > 100:
            co.append(100)           
        else:
            co.append(co[-1] + deltaCO)
    else:
        co.append(coActual)

############## GRÁFICA DE TENDENCIA

fig, ax = plt.subplots(figsize=(13,8),facecolor='grey') # creación de figura
plt.title("Gráfica de Tendencia",color='black',size=16, family="Arial") # asigna el título de la gráfica

plt.xlim(0, tfinal) # ajusta el rango del eje x
ax.set_facecolor('black') # asignación del fondo de la gráfica

ax.axhline(linewidth=2, color='w') # ajuste de propiedades del eje x
ax.set_xlabel("t [s]", color='black')
ax.axvline(linewidth=2, color='w') # ajuste de propiedades del eje y
ax.set_ylabel("y", color='blue')
ax.grid(axis = 'x', color = 'gray', linestyle = 'dashed')
ax.xaxis.set_minor_locator(AutoMinorLocator(10)) # subdivisiones de eje automáticas
ax.xaxis.grid(which='minor', linestyle='dotted', color='gray')
ax.tick_params(direction='out', colors='w', grid_color='w', grid_alpha=0.3) # parámetros de las marcaciones de los ejes

twax = ax.twinx() # creación del eje y secundario
twax.set_ylabel('CO',color='purple')
twax.tick_params(direction='out', length=6, width=1, colors='purple', grid_color='w', grid_alpha=0.5) # parámetros de las marcaciones de los ejes

############# CARGUE DE DATOS

ax.set_ylim(amin([round(amin(y)), round(amin(ysp))])*.95,
            amax([round(amax(y)), round(amax(ysp)), 1])*1.05) # ajusta el rango del eje y principal
twax.set_ylim(0 if amin(co)<0 else round(amin(co))*0.95,
        1 if amax(co)<1 else round(amax(co))*1.05) # ajusta el rango del eje y secundario


lineCO, = twax.plot(t, co, color ='purple', linestyle='solid') # crea la línea con los datos
lineSP, = ax.plot(t, ysp, color ='r', linestyle='solid') # crea la línea con los datos
lineY, = ax.plot(t, y, color ='b', linestyle='solid') # crea la línea con los datos

plt.show()