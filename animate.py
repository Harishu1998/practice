import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    graph_data = open('plot.txt','r').read()
    lines = graph_data.split('\n')
    X = []
    S = []
    E = []
    T = []
    for line in lines:
        if len(line) > 1:
            t, x, s, e = line.split(',')
            T.append(float(t))
            X.append(float(x))
            S.append(float(s))
            E.append(float(e))
    ax1.clear()
    ax1.plot(T,X)
    ax1.plot(T,S)
    ax1.plot(T,E)

ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()