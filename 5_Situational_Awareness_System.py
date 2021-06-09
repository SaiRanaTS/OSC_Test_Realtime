import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
import numpy as np
from matplotlib.patches import Circle

plt.style.use("seaborn-dark")
for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#000000'

for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.9'  # very light grey


fig = plt.figure()
fig.set_size_inches(5, 5)
ax = fig.add_subplot(111, polar=True)

ax.set_theta_zero_location("N")
ax.set_rticks([10, 10, 15, 20])
ax.set_theta_direction(-1)


x_vals = []
y_vals = []

index = count()
ang1 = [0]
d1 = [0]

ang2 = [0]
d2 = [0]

ang3 = [0]
d3 = [0]

ang4 = [0]
d4 = [0]

ang5 = [0]
d5 = [0]

def animate(i):




    data = pd.read_csv('Data_Generated/Radar.csv')
    x = data['Time']
    theta1 = data['T1_Alpha_OT']
    theta2 = data['T2_Alpha_OT']
    theta3 = data['T3_Alpha_OT']
    theta4 = data['T4_Alpha_OT']
    theta5 = data['T5_Alpha_OT']

    r1 = data['T1_Distance']
    r2 = data['T2_Distance']
    r3 = data['T3_Distance']
    r4 = data['T4_Distance']
    r5 = data['T5_Distance']

    numberiii = len(x) - 1
    plt.cla()
    d1.append(r1[numberiii])
    ang1.append(theta1[numberiii])


    d2.append(r2[numberiii])
    ang2.append(theta2[numberiii])

    d3.append(r3[numberiii])
    ang3.append(theta3[numberiii])

    d4.append(r4[numberiii])
    ang4.append(theta4[numberiii])

    d5.append(r5[numberiii])
    ang5.append(theta5[numberiii])


    ax.cla()
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    ax.set_ylim(0, 2)

    ax.set_rmax(2)




    ax.scatter(math.radians(ang1[i]), d1[i], c='r', s=40, cmap='hsv', alpha=0.75)
    ax.text(math.radians(ang1[i]), d1[i],f'T1: Hannara',fontsize=10, color='white')

    ax.scatter(math.radians(ang2[i]), d2[i], c='y', s=40, cmap='hsv', alpha=0.75)
    ax.text(math.radians(ang2[i]), d2[i], f'T2: Stavangerfjord', fontsize=10, color='white')

    ax.scatter(math.radians(ang3[i]), d3[i], c='g', s=40, cmap='hsv', alpha=0.75)
    ax.text(math.radians(ang3[i]), d3[i], f'T3: UNI', fontsize=10, color='white')

    ax.scatter(math.radians(ang4[i]), d4[i], c='b', s=40, cmap='hsv', alpha=0.75)
    ax.text(math.radians(ang4[i]), d4[i], f'T4: Frigat 1', fontsize=10, color='white')

    ax.scatter(math.radians(ang5[i]), d5[i], c='w', s=40, cmap='hsv', alpha=0.75)
    ax.text(math.radians(ang5[i]), d5[i], f'T5: Frigat 2', fontsize=10, color='white')


    # Do your plotting

    # for object
    theta1 = np.deg2rad([5, 5])
    R1 = [0, 2]

    ax.fill_between(
        np.linspace(math.radians(0), math.radians(5), 100),  # Go from 0 to pi/2
        0,  # Fill from radius 0
        5,  # To radius 1
        color ='r',
        alpha = 0.1,
    )

    # for sunrise
    theta2 = np.deg2rad([67.5, 67.5])
    R2 = [0, 2]

    ax.fill_between(
        np.linspace(math.radians(5), math.radians(67.5), 100),  # Go from 0 to pi/2
        0,  # Fill from radius 0
        5,  # To radius 1
        color ='green',
        alpha = 0.1,
    )


    # for sunset
    theta3 = np.deg2rad([112.5, 112.5])
    R3 = [0, 5]

    ax.fill_between(
        np.linspace(math.radians(5), math.radians(112.5), 100),  # Go from 0 to pi/2
        0,  # Fill from radius 0
        5,  # To radius 1
        color ='yellow',
        alpha = 0.1,
    )


    # for midday
    theta4 = np.deg2rad([210, 210])
    R4 = [0, 2]

    ax.fill_between(
        np.linspace(math.radians(112.5), math.radians(210), 100),  # Go from 0 to pi/2
        0,  # Fill from radius 0
        5,  # To radius 1
        color='blue',
        alpha=0.1,
    )

    theta5 = np.deg2rad([247.5, 247.5])
    R5 = [0, 2]

    ax.fill_between(
        np.linspace(math.radians(210), math.radians(247.5), 100),  # Go from 0 to pi/2
        0,  # Fill from radius 0
        5,  # To radius 1
        color='orange',
        alpha=0.1,
    )


    ax.fill_between(
        np.linspace(math.radians(247.5), math.radians(355), 100),  # Go from 0 to pi/2
        0,  # Fill from radius 0
        5,  # To radius 1
        color='lime',
        alpha=0.1,
    )



    theta6 = np.deg2rad([355, 355])
    R6 = [0, 2]

    ax.fill_between(
        np.linspace(math.radians(355), math.radians(360), 100),  # Go from 0 to pi/2
        0,  # Fill from radius 0
        5,  # To radius 1
        color='red',
        alpha=0.1,
    )




    ax.plot(theta1, R1, theta2, R2, theta3, R3, theta4, R4,theta5, R5,theta6, R6, color = 'w',  lw=1)

    def_marker = Path([[-0.005, -0.02], [0.005, -0.02], [0.005, 0.01], [0, 0.02], [-0.005, 0.01], [0, 0], ],
                      [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])

    ax.scatter(0, 0, marker=def_marker, color='white', s=1000, alpha=0.9)

    # plt.fill_between(theta1, theta2, R1, alpha=0.2) #does not work
    circle0 = Circle([2, 2], 2)
    cir_path0 = circle0.get_path()
    ax.scatter(0, 0, marker=cir_path0, facecolors='none', edgecolors='w', s=30000, alpha=0.7)



    ax.grid(False)

    ax.set_title("Radar Plot", va='bottom')

   #ax.scatter(ang1[i], d1[i], c='r', s=40, cmap='hsv', alpha=0.75)

    # c2 = ax.scatter(theta2, r2, c='g', s=40, cmap='hsv', alpha=0.75)
    # ax.cla()
    # c3 = ax.scatter(theta3, r3, c='y', s=40, cmap='hsv', alpha=0.75)
    # ax.cla()
    # c4 = ax.scatter(theta4, r4, c='b', s=40, cmap='hsv', alpha=0.75)
    # ax.cla()
    # c5 = ax.scatter(theta5, r5, c='black', s=40, cmap='hsv', alpha=0.75)
    # ax.cla()






ani = FuncAnimation(plt.gcf(), animate, interval=1000)


ax.cla()
ax.set_thetamin(0)
ax.set_thetamax(360)
plt.show()