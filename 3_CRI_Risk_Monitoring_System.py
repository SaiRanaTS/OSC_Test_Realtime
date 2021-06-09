import matplotlib.pyplot as plt
from itertools import count
import pandas as pd
from matplotlib.animation import FuncAnimation

plt.style.use("seaborn-dark")
for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#16171c'

for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.9'  # very light grey

x_vals = []
y_vals = []

index = count()

fig = plt.figure()
fig.set_size_inches(7, 3)

ax1 = fig.add_subplot(611)
ax2 = fig.add_subplot(612)
ax3 = fig.add_subplot(613)
ax4 = fig.add_subplot(614)
ax5 = fig.add_subplot(615)
ax6 = fig.add_subplot(616)


def vsl_colr(crix):
    if crix <= 0.1:
        cl = 'midnightblue'
        return cl
    elif crix > 0.1 and crix <= 0.17:
        cl = 'blue'
        return cl
    elif crix > 0.17 and crix <= 0.25:
        cl = 'mediumblue'
        return cl
    elif crix > 0.25 and crix <= 0.6:
        cl = 'green'
        return cl
    elif crix > 0.6 and crix <= 0.7:
        cl = 'yellowgreen'
        return cl
    elif crix > 0.7 and crix <= 0.85:
        cl = 'red'
        return cl
    else:
        cl = 'yellow'
        return cl


def animate(i):
    data = pd.read_csv('Data_Generated/CRI.csv')
    x = data['Time']
    y1 = data['TS1_CRIW']
    y2 = data['TS2_CRIW']
    y3 = data['TS3_CRIW']
    y4 = data['TS4_CRIW']
    y5 = data['TS5_CRIW']

    plt.cla()
    numberiii = len(x) - 1
    colorz1 = vsl_colr(y1[numberiii])
    colorz2 = vsl_colr(y2[numberiii])
    colorz3 = vsl_colr(y3[numberiii])
    colorz4 = vsl_colr(y4[numberiii])
    colorz5 = vsl_colr(y5[numberiii])

    colorz = 'r'

    plt.plot(x, y1, label='TS1', linewidth=0.6)
    plt.plot(x, y2, label='TS2', linewidth=0.6)
    plt.plot(x, y3, label='TS3', linewidth=0.6)
    plt.plot(x, y4, label='TS4', linewidth=0.6)
    plt.plot(x, y5, label='TS5', linewidth=0.6)

    ax1.cla()
    ax1.plot(x, y1, label='TS1', color=colorz1, linewidth=0.6)
    ax1.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set(ylim=(0, 1))

    ax2.cla()
    ax2.plot(x, y2, label='TS2', color=colorz2, linewidth=0.6)
    ax2.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set(ylim=(0, 1))

    ax3.cla()
    ax3.plot(x, y3, label='TS3', color=colorz3, linewidth=0.6)
    ax3.legend(loc='upper right')
    ax3.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
    ax3.set(ylim=(0, 1))

    ax4.cla()
    ax4.plot(x, y4, label='TS4', color=colorz4, linewidth=0.6)
    ax4.legend(loc='upper right')
    ax4.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
    ax4.set(ylim=(0, 1))

    ax5.cla()
    ax5.plot(x, y5, label='TS5', color=colorz5, linewidth=0.6)
    ax5.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
    ax5.legend(loc='upper right')
    ax5.set(ylim=(0, 1))

    ax6.legend(loc='upper right')
    ax6.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
    # plt.grid()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()