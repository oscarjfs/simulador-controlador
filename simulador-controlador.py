import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy.integrate import odeint
from random import gauss
from math import sqrt


# parámetros del randomizador de señales
mean = 1
variance = 1e-5

Ts = 0.1 # tiempo de muestreo
controlAutomaticoEncendido = True

Kp = 4.592591493136898 # ganancia del proceso
taup = 15.142301049593728 # constante de tiempo del proceso
td = 5.049971526325448 # tiempo muerto del proceso

Kc = 0.6057530836821651 # ganancia proporcional del controlador
Ki = 0.06063277987348371 # ganancia integral del controlador
Kd = 0 # ganancia derivativa del controlador

tfinal = 100

# condiciones inciales
t0 = 0
y0 = 0
co0 = 0
u0 = 0
ysp0 = 0
Ek2 = 0
Ek1 = 0
Ek = 0

####### cambios en el sp
tstep1 = 5
yspstep1 = 0.5
tstep2 = 50
yspstep2 = 1.5
########################

tstep = tstep1
tActual = t0
yActual = y0
coActual = co0
yspActual = ysp0

ndatos = int(tfinal/Ts+1)
t = [ti*Ts for ti in range(t0,ndatos)]

y = [y0]
co = [co0]
ysp = [ysp0]

def fopdt(y,t,co):

    u = 0 if t<td+tstep else 1

    # calcular la derivada
    dydt = -(y-y0)/taup + Kp/taup * (u-u0) * (co-co0)

    return dydt

# solución del modelo para cada paso o intervalo de tiempo (discretización)
for i in range(0,ndatos-1):
    ts = [t[i],t[i+1]]

    tActual = t[i+1]
    coActual = co[-1]

    if tActual<tstep1:
        ysp.append(ysp0)
    elif tActual<tstep2:
        ysp.append(yspstep1)
    else:
        ysp.append(yspstep2)

    if ysp[-2] != ysp[-1]:
        tstep = tActual
        co0 = co[-1]
        y0 = y[-1]

    if (not controlAutomaticoEncendido) and (co[-2] != co[-1]):
        tstep = tActual

    y1 = odeint(fopdt,y[-1],ts,args=tuple([co[-1]]))
    y.append(float(y1[-1])*gauss(mean,sqrt(variance)))

    if controlAutomaticoEncendido:
        ## PID
        Ek2 = Ek1
        Ek1 = Ek
        Ek = ysp[-1] - y[-1]
        deltaCO = (Kc+Ki*Ts+Kd/Ts)*Ek - (Kc+2*Kd/Ts)*Ek1 + (Kd/Ts)*Ek2
        co.append(co[-1] + deltaCO)
    else:
        co.append(coActual)

############## GRÁFICA DE TENDENCIA

fig, ax = plt.subplots(figsize=(13,8),facecolor='grey') # creación de figura
plt.title("Gráfica de Tendencia",color='black',size=16, family="Arial") # asigna el título de la gráfica

plt.xlim(0, tfinal) # ajusta el rango del eje x
plt.ylim(round(np.amin(y)*.9), round(np.amax(y))*1.1 if np.amax(y)>np.amax(ysp) else round(np.amax(ysp))*1.1) # ajusta el rango del eje y
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
twax.set_ylim(0, 1 if np.amax(co)<1 else round(np.amax(co))*1.1)

############# CARGUE DE DATOS

lineCO, = twax.plot(t, co, color ='purple', linestyle='solid') # crea la línea con los datos
lineSP, = ax.plot(t, ysp, color ='r', linestyle='solid') # crea la línea con los datos
lineY, = ax.plot(t, y, color ='b', linestyle='solid') # crea la línea con los datos

plt.show()