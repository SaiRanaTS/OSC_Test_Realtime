import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use("seaborn-dark")
for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#16171c'

for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.9'  # very light grey



x_vals = []
y_vals = []

index = count()


def animate(i):
    data = pd.read_csv('Data_Generated/CRI.csv')
    x = data['Time']
    y1 = data['TS1_CRIW']
    y2 = data['TS2_CRIW']
    y3 = data['TS3_CRIW']
    y4 = data['TS4_CRIW']
    y5 = data['TS5_CRIW']

    plt.cla()

    plt.plot(x, y1, label='TS1', linewidth=0.6)
    plt.plot(x, y2, label='TS2', linewidth=0.6)
    plt.plot(x, y3, label='TS3', linewidth=0.6)
    plt.plot(x, y4, label='TS4', linewidth=0.6)
    plt.plot(x, y5, label='TS5', linewidth=0.6)

    plt.legend(loc='upper right')
    plt.xlabel("Simulation Time Interval")
    plt.ylabel("CRI Index")
    plt.title("CRI Live Plot")
    plt.grid(color='w', linestyle='-', linewidth=0.1,alpha=0.3)
    #plt.grid()
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()