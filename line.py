import numpy as np
import matplotlib.pyplot as plt
from hsv_plot import HSV

plt.style.use('seaborn')
plt.ion()

x = np.linspace(0, 6*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = 1.5*np.cos(x+0.2)
y4 = 2*np.cos(x+0.35)
y5 = 0.7*np.cos(x+1)

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y1) 
line2, = ax.plot(x, y2) 
line3, = ax.plot(x, y3) 
line4, = ax.plot(x, y4) 

HSV(fig, [line1, line2, line3, line4])