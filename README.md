# LABORATORIO 2: 
## CONVOLUCIÓN, CORRELACIÓN Y TRANSFORMADA DE FOURIER
### Objetivo
En este laboratorio se pretende comprender y aplicar los conceptos fundamentales del procesamiento de señales, utilizando la convolución para determinar la respuesta de un sistema discreto, la correlación como medida de similitud entre señales y la Transformada de Fourier como herramienta de análisis en el dominio de la frecuencia.
# PARTE A

Se realizo la convolución de las señales **$x[n]$** y **$h[n]$**, tanto a mano como en python, de cada uno de los integrantes del grupo.
<img width="336" height="1280" alt="image" src="https://github.com/user-attachments/assets/0ac85b41-078e-4aca-bfb9-71c84910b7fb" />

## LIBRERIAS
Las librerias que implementamos fueron las siguientes:

+ **Importación de librerias**
```phyton
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from scipy.signal import correlate
from numpy.fft import fft, fftfreq
```
## A mano
+ **Karen**
### CONVOLUCIÓN
<img width="1000" height="1390" alt="image" src="https://github.com/user-attachments/assets/39e440d0-d69c-41cc-934b-e63bd866082a" />

### GRÁFICA
<img width="1000" height="764" alt="image" src="https://github.com/user-attachments/assets/3cc04951-d661-4bc8-b684-101044d458c9" />

+ **Alissia**
### CONVOLUCIÓN 
![img5](https://github.com/user-attachments/assets/f7b66f0b-dff7-43e4-8338-f408b29190e7)

### GRÁFICA
![img11](https://github.com/user-attachments/assets/a5a13979-f34e-4bfe-83e3-b5b7e26d17fc)

+ **Raúl**
### CONVOLUCIÓN
![img15](https://github.com/user-attachments/assets/650571a2-26b8-4c34-9915-b5651f1a5550)

### GRÁFICA
![img19](https://github.com/user-attachments/assets/8371adad-c301-443c-8681-25e25ca1ac3b)

## En Python
+ **Karen**
```python
import numpy as np
import matplotlib.pyplot as plt

# Señales
h_karen = [5,6,0,0,8,7,1]
x_karen = [1,0,3,1,6,4,8,6,8,5]

# Convolución
y_karen = np.convolve(x_karen, h_karen)
n_karen = np.arange(len(y_karen))

# Resultados
print("=== Karen ===")
print("x[n] =", x_karen)
print("h[n] =", h_karen)
print("y[n] =", y_karen.tolist())

# Gráfico
plt.figure(figsize=(10,4))
plt.stem(n_karen, y_karen, basefmt="k-")
plt.title("Convolución y[n] para Karen")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()
```
### GRÁFICA
<img width="852" height="437" alt="image" src="https://github.com/user-attachments/assets/382f5793-1779-45d7-b3bf-57beafa23c1c" />

+ **Alissia**
```python
import numpy as np
import matplotlib.pyplot as plt

# Señales
h_alissia = [5,6,0,0,7,2,4]
x_alissia = [1,0,0,3,8,6,5,6,3,5]

# Convolución
y_alissia = np.convolve(x_alissia, h_alissia)
n_alissia = np.arange(len(y_alissia))

# Resultados
print("=== Alissia ===")
print("x[n] =", x_alissia)
print("h[n] =", h_alissia)
print("y[n] =", y_alissia.tolist())

# Gráfico
plt.figure(figsize=(10,4))
plt.stem(n_alissia, y_alissia, basefmt="k-")
plt.title("Convolución y[n] para Alissia")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()
```
### GRÁFICA
<img width="942" height="439" alt="image" src="https://github.com/user-attachments/assets/6e69c88e-c5ee-4079-b55d-b915515170f3" />

+ **Raúl**
```python
import numpy as np
import matplotlib.pyplot as plt

# Señales
h_raul = [5,6,0,0,6,8,7]
x_raul = [1,1,9,3,5,1,9,6,8,5]

# Convolución
y_raul = np.convolve(x_raul, h_raul)
n_raul = np.arange(len(y_raul))

# Resultados
print("=== Raul ===")
print("x[n] =", x_raul)
print("h[n] =", h_raul)
print("y[n] =", y_raul.tolist())

# Gráfico
plt.figure(figsize=(10,4))
plt.stem(n_raul, y_raul, basefmt="k-")
plt.title("Convolución y[n] para Raul")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()
```
### GRÁFICA
<img width="902" height="435" alt="image" src="https://github.com/user-attachments/assets/5ba1b078-41b9-4f00-b45c-14ff6aaf6f55" />

Se realizo el cálculo de la convolución de dos señales discretas, donde la función que se utilizó para esta tarea fue `np.convolve` de la librería **Numpy**. Esta función recibe como entrada la señal **$x[n]$** y la respuesta al impulso **$h[n]$** y devuelve como salida **$y[n]$**, que corresponde a una convolución líneal.

La convolución es importante porque describe como una señal se modifica cuando pasa por un sistema, esta operación se utiliza para aplicar filtros, mejorar o atenuar ciertas frecuencias, eliminar ruido, o extraer información relevante de una señal.

# PARTE B

Se realizo la correlación cruzada de dos señales dadas y se determino su importancia en el procesamiento digital de señales.
<img width="363" height="1280" alt="image" src="https://github.com/user-attachments/assets/bb076aa0-c52d-4c2b-83f3-44843574d6e1" />

Las señales que nos dan son las siguientes:

- $x_1[nT_s] = \cos(2\pi 100nT_s), \quad 0 \leq n < 9$
- $x_2[nT_s] = \sin(2\pi 100nT_s), \quad 0 \leq n < 9$

A continuación se realiza la correlación cruzada:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

# ===========================
# Parámetros y señales
# ===========================
Ts = 1.25e-3   # periodo de muestreo [s]
fs = 1/Ts      # frecuencia de muestreo [Hz]
f0 = 100       # frecuencia de la señal [Hz]
n = np.arange(0, 9)   # índices 0..8
t = n * Ts

# Definición de señales
x1 = np.cos(2*np.pi*f0*t)
x2 = np.sin(2*np.pi*f0*t)

# ===========================
# Correlación cruzada
# ===========================
r12 = correlate(x1, x2, mode='full')          # no normalizada
lags = correlation_lags(len(x1), len(x2), mode='full')

# Normalización
norm_factor = np.sqrt(np.dot(x1, x1) * np.dot(x2, x2))
r12_norm = r12 / norm_factor

# ===========================
# Gráfica 1: Señales originales
# ===========================
plt.figure(figsize=(9,3))
plt.stem(t, x1, basefmt="k-", linefmt="C0-", markerfmt="C0o", label="x1[n] = cos(2π100nTs)")
plt.stem(t, x2, basefmt="k-", linefmt="C1-", markerfmt="C1s", label="x2[n] = sin(2π100nTs)")
plt.title("Señales discretas x1[n] y x2[n]")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.show()

# ===========================
# Gráfica 2: Correlación no normalizada
# ===========================
plt.figure(figsize=(9,3))
plt.stem(lags, r12, basefmt="k-", linefmt="m-", markerfmt="mo")
plt.title("Correlación cruzada r_{x1,x2}[lag] (no normalizada)")
plt.xlabel("lag [muestras]")
plt.ylabel("r[lag]")
plt.grid(True)
plt.show()

# ===========================
# Gráfica 3: Correlación normalizada
# ===========================
plt.figure(figsize=(9,3))
plt.stem(lags, r12_norm, basefmt="k-", linefmt="m-", markerfmt="mo")
plt.title("Correlación cruzada normalizada r_{x1,x2}[lag]")
plt.xlabel("lag [muestras]")
plt.ylabel("r_norm[lag]")
plt.ylim(-1.1, 1.1)  # resaltar rango [-1,1]
plt.grid(True)
plt.show()
```
### RESULTADOS OBTENIDOS
<img width="699" height="282" alt="image" src="https://github.com/user-attachments/assets/56c24895-af4a-4f19-be29-31f99be26749" />
<img width="685" height="278" alt="image" src="https://github.com/user-attachments/assets/b60eaa3c-48bc-4344-bb6e-98f4efcf0601" />
<img width="698" height="283" alt="image" src="https://github.com/user-attachments/assets/d898bac9-6778-4fc4-a28c-ee7358416879" />

La correlacion cruzada es una herramienta que nos dice que tan parecidas son dos señales, en diferentes posiciones, mide el grado de similitud entre ellas a medida que una se va desplazando (desfasando) con respecto a la otra.

Teniendo en cuenta esto, lo que podemos describir de la secuencia resultante es:
+ **Presenta valores elevados en las posiciones donde las dos señales coinciden en forma y desplazamiento, mientras que en los otros desplazamientos los valores disminuyen o tieden a cero**
+ **Como el seno y el coseno estan desfasados 90°, la correlación presenta valores negativos y positivos, alternados, sin alcanzar un máximo de coincidencia en `lag=0`**.
+ **Dado que coseno y seno son ortogonales en un período completo, la correlación promedio tenderá a ser cercana a cero, confirmando que son señales diferentes pero relacionadas.**
+ **Los picos positivos y negativos de la correlación muestran los puntos de máxima similitud y máxima oposición según el corrimiento aplicado.**

### ¿En qué situaciones resulta útil aplicar la correlación cruzada en el procesamiento digital de señales? 

1. **Detección de patrones: Se usa para identificar si una señal contiene una señal conocida.**
2. **Estimación de retardo (delay): Permite calcular cuánto está desfasada una señal con respecto a otra, útil en sistemas de sincronización.**
3. **Procesamiento de audio y voz: Se aplica para reconocimiento de voz, cancelación de eco y para sincronizar audio grabado con señales de referencia.**
4. **Procesamiento Biomédico: mide la similitud entre señales, que también revela la existencia de retardos fisiológicos, sincronización o descoordinación entre distintas regiones del cuerpo.**
5. **Sistemas de posicionamiento y comunicaciones inalámbricas: Se utiliza para sincronizar la señal recibida con una señal piloto enviada, ayudando a estimar distancias y tiempos de llegada.**

# PARTE C
Con ayuda del generador de señales fisiológicas, se obtuvo la señal de un EOG, a la cual se le hallo las medidas estadísticas y la Transformada de Fourier.

<img width="237" height="1280" alt="image" src="https://github.com/user-attachments/assets/d4c50846-9153-42b5-ab81-0b37731a6af9" />

Se extrajo la señal del generador de señales fisiológicas por medio de un DAQ y se graficó con el siguiente código:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from numpy.fft import fft, fftfreq

df = pd.read_csv("Señal_EOG.txt", sep="\t")

print(df.head())

# Graficar la señal
plt.figure(figsize=(12,4))
plt.plot(df["time_s"], df["EOG_mV"], linewidth=1)
plt.xlabel("Tiempo (s)")
plt.ylabel("EOG (mV)")
plt.title("Señal EOG simulada")
plt.tight_layout()
plt.show()
```
Obteniendo la siguiente gráfica:
<img width="1258" height="468" alt="image" src="https://github.com/user-attachments/assets/20eb013d-a820-4f57-a8bd-0043281650f3" />

A continuación determinamos la frecuencia de Nyquist y digitalizamos la señal utilizando una frecuencia de muestreo que sea 4 veces la frecuencia de Nyquist y el código quedó de la siguiente manera:

```python
# 1. Cargamos la señal
df = pd.read_csv("Señal_EOG.txt", sep="\t")

# Señal original
t = df["time_s"].values
x = df["EOG_mV"].values

# Frecuencia de muestreo
fs = 1 / np.mean(np.diff(t))
print(f"Frecuencia de muestreo: {fs:.2f} Hz")

# 2. Frecuencia de Nyquist
f_nyquist = fs / 2
print(f"Frecuencia de Nyquist: {f_nyquist:.2f} Hz")

# 3. Nueva frecuencia de muestreo (4 veces Nyquist)
fs_new = 4 * f_nyquist
print(f"Nueva frecuencia de muestreo: {fs_new:.2f} Hz")

# 4. Número de muestras en la nueva señal
dur = t[-1] - t[0]
n_new = int(dur * fs_new)

# 5. Digitalización con nueva Fs
from scipy.signal import resample
x_resampled = resample(x, n_new)
t_resampled = np.linspace(t[0], t[-1], n_new)

# 6. Graficar comparación
plt.figure(figsize=(12,5))
plt.plot(t, x, label="Original (250 Hz)", alpha=0.7)
plt.plot(t_resampled, x_resampled, label=f"Resampleado ({fs_new:.0f} Hz)", linewidth=1)
plt.xlabel("Tiempo (s)")
plt.ylabel("EOG (mV)")
plt.title("Comparación: Señal original vs. Digitalizada")
plt.legend()
plt.tight_layout()
plt.show()
```
Obteniendo los siguientes resultados:
+ **Frecuencia de muestreo: 250.00 Hz**
+ **Frecuencia de Nyquist: 125.00 Hz**
+ **Nueva frecuencia de muestreo: 500.00 Hz**

<img width="1067" height="437" alt="image" src="https://github.com/user-attachments/assets/7b4bae10-351d-4f49-a487-14cf5a20aee5" />

Luego caracterizamos la señal según las medidas estadísticas:
```python
media = np.mean(x_resampled)
mediana = np.median(x_resampled)
desv_std = np.std(x_resampled)
maximo = np.max(x_resampled)
minimo = np.min(x_resampled)

print("===== Características estadísticas =====")
print(f"Media: {media:.4f}")
print(f"Mediana: {mediana:.4f}")
print(f"Desviación estándar: {desv_std:.4f}")
print(f"Máximo: {maximo:.4f}")
print(f"Mínimo: {minimo:.4f}")
```
### Resultados
+ **Media: 0.0227**
+ **Mediana: 0.0901**
+ **Desviación estándar: 0.6214**
+ **Máximo: 2.2859**
+ **Mínimo: -1.2537**

### Transformada de Fourier

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

# ---------- 1) Preprocesado ----------
# Quitar componente DC (media)
x_no_dc = x_resampled - np.mean(x_resampled)

# Filtro pasa-altas Butterworth (corte en 0.05 Hz)
fc = 0.05  # Hz
b, a = butter(N=4, Wn=fc/(fs_new/2), btype='highpass')
x_hp = filtfilt(b, a, x_no_dc)

# ---------- 2) Función FFT (amplitud normalizada) ----------
def fft_pos_amp(x, fs):
    N = len(x)
    X = fft(x)
    f = fftfreq(N, 1/fs)
    pos = slice(0, N//2)
    amp = np.abs(X[pos]) / N  # amplitud por Hz
    return f[pos], amp

f_orig, A_orig = fft_pos_amp(x_resampled, fs_new)
f_hp,   A_hp   = fft_pos_amp(x_hp, fs_new)

# ---------- 3) Gráfica comparativa ----------
plt.figure(figsize=(9,4))
plt.semilogx(f_orig, A_orig, label="Original", color="C0")
plt.semilogx(f_hp,   A_hp,   label=f"Filtrada (HP > {fc} Hz)", color="C1")
plt.title("Transformada de Fourier (comparación)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [mV/Hz]")  # ajusta unidad si aplica
plt.xlim(1e-2, 0.5)   # rango 0.01 a 0.5 Hz
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()
```
### GRÁFICA
<img width="801" height="349" alt="image" src="https://github.com/user-attachments/assets/0295eb31-c662-4043-ba3e-57587af37ff9" />

Y clasificamos la señal:
+ **Determinística o aleatoria: Es una señal Aleatoria, ya que incluye ruido y eventos transitorios con tiempos y amplitudes no deterministas, es decir que no hay parpadeos.**
+ **Periódica o aperiódica: La señal es Aperiódica, ya que no hay repetición regular; la correlación no muestra picos repetidos a intervalos fijos y la FFT no tiene líneas discretas separadas.**
+ **Analógica o digital: Es una señal Digital, la señal está muestreada y representada por valores discretos en el tiempo. Físicamente modela un proceso analógico EOG, pero es registro es digital.**

A continuacion se realizó el cálculo de la densidad espectral de potencia y se analizó los estadísticos en el dominio de la frecuencia de la señal:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Suponemos que ya tienes x_resampled y fs_new
x = x_resampled
fs = fs_new

# 1) Calcular PSD
freqs, Pxx = welch(x, fs=fs_new, nperseg=1024)

# Normalizar PSD para usar como "probabilidad" en histograma
PSD_norm = PSD / np.sum(PSD)

# 2) Gráfica de la PSD
plt.figure(figsize=(9,4))
plt.semilogy(freqs, Pxx, color='magenta')
plt.title("Densidad Espectral de Potencia (PSD)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [mV²/Hz]")  # ajusta unidad
plt.xlim(0, 5)  # enfocarse en bajas frecuencias
plt.grid(True, which="both")
plt.show()


# 3) Histograma de frecuencias
plt.figure(figsize=(8,4))
plt.hist(freqs, bins=30, weights=PSD_norm, color='#40E0D0', edgecolor='black')
plt.title("Distribución de Potencia en Frecuencia")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Probabilidad relativa")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4) Estadísticos en frecuencia
f_mean = np.sum(freqs * PSD_norm)
cdf = np.cumsum(PSD_norm)
f_median = freqs[np.searchsorted(cdf, 0.5)]
f_std = np.sqrt(np.sum((freqs - f_mean)**2 * PSD_norm))

print("\n Estadísticos en el dominio de la frecuencia")
print("-------------------------------------------------")
print(f"Frecuencia media:   {f_mean:.3f} Hz")
print(f"Frecuencia mediana: {f_median:.3f} Hz")
print(f"Desviación estándar:{f_std:.3f} Hz")
```
### RESULTADOS
+ **DENSIDAD ESPECTRAL DE POTENCIA**
<img width="709" height="352" alt="image" src="https://github.com/user-attachments/assets/41467175-4e23-400a-b6b4-cd808c88a815" />

+ **HISTOGRAMA DE FRECUENCIAS**
<img width="709" height="353" alt="image" src="https://github.com/user-attachments/assets/cd860505-a975-4000-92a8-548738c77c07" />

### ESTADISTICOS EN EL DOMINIO DE LA FRECUENCIA
+ **Frecuencia media:   4.062 Hz**
+ **Frecuencia mediana: 1.953 Hz**
+ **Desviación estándar:9.536 Hz**

La señal EOG procesada muestra un dominio espectral cargado en frecuencias muy bajas (<0.5 Hz), lo cual es característico de este tipo de biopotenciales oculares. Esto explica el “casco” o pico fuerte observado al inicio del espectro.
En el dominio temporal, la señal exhibe eventos no periódicos asociados a parpadeos y movimientos oculares, más un componente aleatorio (ruido instrumental y fisiológico).

En consecuencia, la señal no puede considerarse determinística ni periódica, sino que se clasifica como aleatoria, aperiódica y digital, reflejando tanto la naturaleza fisiológica del EOG como el proceso de adquisición por computadora.

### Análisis de los Datos Estadísticos

+ **Frecuencia Media: la energía espectral de la señal EOG se concentra en torno a los 4 Hz, esto tiene mucho sentido ya que los movimientos oculares tienen componentes de baja frecuencia, asociadas al parpadeo y a los desplazamientos oculares.**
+ **Frecuencia Mediana: La mediana representa el punto donde se concentra el 50% de la energía total de la señal, esta se encuentra en las frecuencias muy bajas, lo que podría deberse a movimientos muy lentos de los parpadeos y esta predomina sobre los datos de mayor frecuencia.**
+ **Desviación Estándar: aunque la mayor parte de la energía está en bajas frecuencias, existe dispersión significativa hacia frecuencias más altas, esto puede deberse a ruido (por ejemplo, actividad muscular o interferencia eléctrica de 50/60 Hz), o a eventos rápidos en la señal (como micro-movimientos oculares).**










