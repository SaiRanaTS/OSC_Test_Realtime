import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from itertools import count
import pandas as pd
from matplotlib.animation import FuncAnimation
from PIL import Image
import math
import numpy as np

plt.style.use("seaborn-dark")
for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#16171c'

for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.9'  # very light grey

index = count()
img = plt.imread('ji.jpg')
fig, ax = plt.subplots()
fig.set_size_inches(17, 6)
plt.subplots_adjust(left=0.05, right=0.5, top=0.5, bottom=0.05)
ax.imshow(img, alpha=0.96)

plt.grid(alpha=0.2)
plt.axis('off')


ax1 = fig.add_subplot(251)
ax2 = fig.add_subplot(252)

ax3 = fig.add_subplot(253)
ax4 = fig.add_subplot(254)
ax5 = fig.add_subplot(255)


ax6 = fig.add_subplot(256)
ax7 = fig.add_subplot(257)
ax8 = fig.add_subplot(258)
ax9 = fig.add_subplot(259)
ax10 = fig.add_subplot(2,5,10)








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


n = 0
nx = 0
nx2 = 0
nx3 = 0
nx4 = 0
nx5 = 0
def animate(i):
    global nx
    global nx2
    global nx3
    global nx4
    global nx5
    n = 0
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

    if nx == 0:
        print(nx)

        # Target Ship 1 img
        l1 = 150
        h1 = 25
        img1 = plt.imread('Vsl_Type/TYP1S1.png')
        im1 = Image.open('Vsl_Type/TYP1S1.png')
        width1, height1 = im1.size
        nii1 = range(1, width1, round(width1 / 8))
        l1_cap = range(0, l1, round(l1 / 8))
        h1_cap = range(h1, 0, -round(h1 / 5))
        mii1 = range(1, height1, round(height1 / 5))
        # ---------------------Plot TS1-------------------
        ax1.cla()
        # plt.axis('off')
        ax1.imshow(img1)
        ax1.set_xticks(list(nii1))
        ax1.set_yticks(list(mii1))
        ax1.set_xticklabels(list(l1_cap))
        ax1.set_yticklabels(list(h1_cap))
        ax1.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax1.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        # ax2.axis('off')
        nx += 1

    elif nx == 1:

        # Target Ship 1 img
        h1 = 25
        widz = 40
        img1 = plt.imread('Vsl_Type/TYP1F.png')
        im1 = Image.open('Vsl_Type/TYP1F.png')
        width1, height1 = im1.size
        nii1 = range(1, width1, round(width1 / 8))
        print(list(nii1))
        l1_cap = range(-widz, widz, round(widz / 4))
        print(list(l1_cap))
        h1_cap = range(h1, 0, -round(h1 / 5))
        mii1 = range(1, height1, round(height1 / 5))
        # ---------------------Plot TS1-------------------
        ax1.cla()
        # plt.axis('off')
        ax1.imshow(img1)
        ax1.set_xticks(list(nii1))
        ax1.set_yticks(list(mii1))
        ax1.set_xticklabels(list(l1_cap))
        ax1.set_yticklabels(list(h1_cap))
        ax1.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax1.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        # ax2.axis('off')
        nx += 1


    elif nx == 2:

        # Target Ship 1 img
        l1 = 150
        h1 = 25
        img1 = plt.imread('Vsl_Type/TYP1S2.png')
        im1 = Image.open('Vsl_Type/TYP1S2.png')
        width1, height1 = im1.size
        nii1 = range(1, width1, round(width1 / 8))
        l1_cap = range(0, l1, round(l1 / 8))
        h1_cap = range(h1, 0, -round(h1 / 5))
        mii1 = range(1, height1, round(height1 / 5))
        # ---------------------Plot TS1-------------------
        ax1.cla()
        # plt.axis('off')
        ax1.imshow(img1)
        ax1.set_xticks(list(nii1))
        ax1.set_yticks(list(mii1))
        ax1.set_xticklabels(list(l1_cap))
        ax1.set_yticklabels(list(h1_cap))
        ax1.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax1.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        # ax2.axis('off')
        nx += 1

    elif nx == 3:

        # Target Ship 1 img
        h1 = 25
        widz = 40
        img1 = plt.imread('Vsl_Type/TYP1B.png')
        im1 = Image.open('Vsl_Type/TYP1B.png')
        width1, height1 = im1.size
        nii1 = range(1, width1, round(width1 / 8))
        print(list(nii1))
        l1_cap = range(-widz, widz, round(widz / 4))
        print(list(l1_cap))
        h1_cap = range(h1, 0, -round(h1 / 5))
        mii1 = range(1, height1, round(height1 / 5))
        # ---------------------Plot TS1-------------------
        ax1.cla()
        # plt.axis('off')
        ax1.imshow(img1)
        ax1.set_xticks(list(nii1))
        ax1.set_yticks(list(mii1))
        ax1.set_xticklabels(list(l1_cap))
        ax1.set_yticklabels(list(h1_cap))
        ax1.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax1.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        nx = 0








    if nx2 == 0:
        print(nx)

        # Target Ship 2 img
        l2 = 150
        h2 = 25
        img2 = plt.imread('Vsl_Type/TYP2S1.png')
        im2 = Image.open('Vsl_Type/TYP2S1.png')
        width2, height2 = im2.size
        nii2 = range(1, width2, round(width2 / 8))
        l2_cap = range(0, l2, round(l2 / 8))
        h2_cap = range(h2, 0, -round(h2 / 5))
        mii2 = range(1, height2, round(height2 / 5))
        # ---------------------Plot TS1-------------------
        ax2.cla()
        # plt.axis('off')
        ax2.imshow(img2)
        ax2.set_xticks(list(nii2))
        ax2.set_yticks(list(mii2))
        ax2.set_xticklabels(list(l2_cap))
        ax2.set_yticklabels(list(h2_cap))
        ax2.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax2.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        # ax2.axis('off')
        nx2 += 1

    elif nx2 == 1:

        # Target Ship 1 img
        h2 = 25
        widz2 = 40
        img2 = plt.imread('Vsl_Type/TYP2F.png')
        im2 = Image.open('Vsl_Type/TYP2F.png')
        width2, height2 = im2.size
        nii2 = range(1, width2, round(width2 / 8))
        print(list(nii2))
        l2_cap = range(-widz2, widz2, round(widz2 / 4))
        print(list(l2_cap))
        h2_cap = range(h2, 0, -round(h2 / 5))
        mii2 = range(1, height2, round(height2 / 5))
        # ---------------------Plot TS1-------------------
        ax2.cla()
        # plt.axis('off')
        ax2.imshow(img2)
        ax2.set_xticks(list(nii2))
        ax2.set_yticks(list(mii2))
        ax2.set_xticklabels(list(l2_cap))
        ax2.set_yticklabels(list(h2_cap))
        ax2.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax2.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        # ax2.axis('off')
        nx2 += 1


    elif nx2 == 2:

        # Target Ship 1 img
        l2 = 150
        h2 = 25
        img2 = plt.imread('Vsl_Type/TYP2S2.png')
        im2 = Image.open('Vsl_Type/TYP2S2.png')
        width2, height2 = im2.size
        nii2 = range(1, width2, round(width2 / 8))
        l2_cap = range(0, l2, round(l2 / 8))
        h2_cap = range(h2, 0, -round(h2 / 5))
        mii2 = range(1, height2, round(height2 / 5))
        # ---------------------Plot TS1-------------------
        ax2.cla()
        # plt.axis('off')
        ax2.imshow(img2)
        ax2.set_xticks(list(nii2))
        ax2.set_yticks(list(mii2))
        ax2.set_xticklabels(list(l2_cap))
        ax2.set_yticklabels(list(h2_cap))
        ax2.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax2.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        # ax2.axis('off')
        nx2 += 1

    elif nx2 == 3:

        # Target Ship 1 img
        h2 = 25
        widz2 = 40
        img2 = plt.imread('Vsl_Type/TYP2B.png')
        im2 = Image.open('Vsl_Type/TYP2B.png')
        width2, height2 = im2.size
        nii2 = range(1, width2, round(width2 / 8))
        print(list(nii2))
        l2_cap = range(-widz2, widz2, round(widz2 / 4))
        print(list(l2_cap))
        h2_cap = range(h2, 0, -round(h2 / 5))
        mii2 = range(1, height2, round(height2 / 5))
        # ---------------------Plot TS1-------------------
        ax2.cla()
        # plt.axis('off')
        ax2.imshow(img2)
        ax2.set_xticks(list(nii2))
        ax2.set_yticks(list(mii2))
        ax2.set_xticklabels(list(l2_cap))
        ax2.set_yticklabels(list(h2_cap))
        ax2.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax2.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        nx2 = 0




#++++++++++++++++++++++++++

    if nx3 == 0:


        # Target Ship 3 img
        l3 = 150
        h3 = 25
        img3 = plt.imread('Vsl_Type/TYP3S1.png')
        im3 = Image.open('Vsl_Type/TYP3S1.png')
        width3, height3 = im3.size
        nii3 = range(1, width3, round(width3 / 8))
        l3_cap = range(0, l3, round(l3 / 8))
        h3_cap = range(h3, 0, -round(h3 / 5))
        mii3 = range(1, height3, round(height3 / 5))
        # ---------------------Plot TS1-------------------
        ax3.cla()
        # plt.axis('off')
        ax3.imshow(img3)
        ax3.set_xticks(list(nii3))
        ax3.set_yticks(list(mii3))
        ax3.set_xticklabels(list(l3_cap))
        ax3.set_yticklabels(list(h3_cap))
        ax3.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax3.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        # ax2.axis('off')
        nx3 += 1

    elif nx3 == 1:

        # Target Ship 1 img
        h3 = 25
        widz3 = 40
        img3 = plt.imread('Vsl_Type/TYP3F.png')
        im3 = Image.open('Vsl_Type/TYP3F.png')
        width3, height3 = im3.size
        nii3 = range(1, width3, round(width3 / 8))
        print(list(nii3))
        l3_cap = range(-widz3, widz3, round(widz3 / 4))
        print(list(l3_cap))
        h3_cap = range(h3, 0, -round(h3 / 5))
        mii3 = range(1, height3, round(height3 / 5))
        # ---------------------Plot TS1-------------------
        ax3.cla()
        # plt.axis('off')
        ax3.imshow(img3)
        ax3.set_xticks(list(nii3))
        ax3.set_yticks(list(mii3))
        ax3.set_xticklabels(list(l3_cap))
        ax3.set_yticklabels(list(h3_cap))
        ax3.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax3.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        # ax2.axis('off')
        nx3 += 1


    elif nx3 == 2:

        # Target Ship 1 img
        l3 = 150
        h3 = 25
        img3 = plt.imread('Vsl_Type/TYP3S2.png')
        im3 = Image.open('Vsl_Type/TYP3S2.png')
        width3, height3 = im3.size
        nii3 = range(1, width3, round(width3 / 8))
        l3_cap = range(0, l3, round(l3 / 8))
        h3_cap = range(h3, 0, -round(h3 / 5))
        mii3 = range(1, height3, round(height3 / 5))
        # ---------------------Plot TS1-------------------
        ax3.cla()
        # plt.axis('off')
        ax3.imshow(img3)
        ax3.set_xticks(list(nii3))
        ax3.set_yticks(list(mii3))
        ax3.set_xticklabels(list(l3_cap))
        ax3.set_yticklabels(list(h3_cap))
        ax3.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax3.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        # ax2.axis('off')
        nx3 += 1

    elif nx3 == 3:

        # Target Ship 1 img
        h3 = 25
        widz3 = 40
        img3 = plt.imread('Vsl_Type/TYP3B.png')
        im3 = Image.open('Vsl_Type/TYP3B.png')
        width3, height3 = im3.size
        nii3 = range(1, width3, round(width3 / 8))
        print(list(nii3))
        l3_cap = range(-widz3, widz3, round(widz3 / 4))
        print(list(l3_cap))
        h3_cap = range(h3, 0, -round(h3 / 5))
        mii3 = range(1, height3, round(height3 / 5))
        # ---------------------Plot TS1-------------------
        ax3.cla()
        # plt.axis('off')
        ax3.imshow(img3)
        ax3.set_xticks(list(nii3))
        ax3.set_yticks(list(mii3))
        ax3.set_xticklabels(list(l3_cap))
        ax3.set_yticklabels(list(h3_cap))
        ax3.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax3.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        nx3 = 0





#++++++++++++++++++++++++++


#++++++++++++++++++++++++++

    if nx4 == 0:


        # Target Ship 3 img
        l4 = 150
        h4 = 25
        img4 = plt.imread('Vsl_Type/TYP4S1.png')
        im4 = Image.open('Vsl_Type/TYP4S1.png')
        width4, height4 = im4.size
        nii4 = range(1, width4, round(width4 / 8))
        l4_cap = range(0, l4, round(l4 / 8))
        h4_cap = range(h4, 0, -round(h4 / 5))
        mii4 = range(1, height4, round(height4 / 5))
        # ---------------------Plot TS1-------------------
        ax4.cla()
        # plt.axis('off')
        ax4.imshow(img4)
        ax4.set_xticks(list(nii4))
        ax4.set_yticks(list(mii4))
        ax4.set_xticklabels(list(l4_cap))
        ax4.set_yticklabels(list(h4_cap))
        ax4.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax4.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        # ax2.axis('off')
        nx4 += 1

    elif nx4 == 1:

        # Target Ship 1 img
        h4 = 25
        widz4 = 40
        img4 = plt.imread('Vsl_Type/TYP4F.png')
        im4 = Image.open('Vsl_Type/TYP4F.png')
        width4, height4 = im4.size
        nii4 = range(1, width4, round(width4 / 8))
        print(list(nii4))
        l4_cap = range(-widz4, widz4, round(widz4 / 4))
        print(list(l4_cap))
        h4_cap = range(h4, 0, -round(h4 / 5))
        mii4 = range(1, height4, round(height4 / 5))
        # ---------------------Plot TS1-------------------
        ax4.cla()
        # plt.axis('off')
        ax4.imshow(img4)
        ax4.set_xticks(list(nii4))
        ax4.set_yticks(list(mii4))
        ax4.set_xticklabels(list(l4_cap))
        ax4.set_yticklabels(list(h4_cap))
        ax4.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax4.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        # ax2.axis('off')
        nx4 += 1


    elif nx4 == 2:

        # Target Ship 1 img
        l4 = 150
        h4 = 25
        img4 = plt.imread('Vsl_Type/TYP4S2.png')
        im4 = Image.open('Vsl_Type/TYP4S2.png')
        width4, height4 = im4.size
        nii4 = range(1, width4, round(width4 / 8))
        l4_cap = range(0, l4, round(l4 / 8))
        h4_cap = range(h4, 0, -round(h4 / 5))
        mii4 = range(1, height4, round(height4 / 5))
        # ---------------------Plot TS1-------------------
        ax4.cla()
        # plt.axis('off')
        ax4.imshow(img4)
        ax4.set_xticks(list(nii4))
        ax4.set_yticks(list(mii4))
        ax4.set_xticklabels(list(l4_cap))
        ax4.set_yticklabels(list(h4_cap))
        ax4.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax4.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        # ax2.axis('off')
        nx4 += 1

    elif nx4 == 3:

        # Target Ship 1 img
        h4 = 25
        widz4 = 40
        img4 = plt.imread('Vsl_Type/TYP4B.png')
        im4 = Image.open('Vsl_Type/TYP4B.png')
        width4, height4 = im4.size
        nii4 = range(1, width4, round(width4 / 8))
        print(list(nii4))
        l4_cap = range(-widz4, widz4, round(widz4 / 4))
        print(list(l4_cap))
        h4_cap = range(h4, 0, -round(h4 / 5))
        mii4 = range(1, height4, round(height4 / 5))
        # ---------------------Plot TS1-------------------
        ax4.cla()
        # plt.axis('off')
        ax4.imshow(img4)
        ax4.set_xticks(list(nii4))
        ax4.set_yticks(list(mii4))
        ax4.set_xticklabels(list(l4_cap))
        ax4.set_yticklabels(list(h4_cap))
        ax4.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax4.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        nx4 = 0




#++++++++++++++++++++++++++
#5555555555555555555

#++++++++++++++++++++++++++

    if nx5 == 0:


        # Target Ship 3 img
        l5 = 150
        h5 = 25
        img5 = plt.imread('Vsl_Type/TYP5S1.png')
        im5 = Image.open('Vsl_Type/TYP5S1.png')
        width5, height5 = im5.size
        nii5 = range(1, width5, round(width5 / 8))
        l5_cap = range(0, l5, round(l5 / 8))
        h5_cap = range(h5, 0, -round(h5 / 5))
        mii5 = range(1, height5, round(height5 / 5))
        # ---------------------Plot TS1-------------------
        ax5.cla()
        # plt.axis('off')
        ax5.imshow(img5)
        ax5.set_xticks(list(nii5))
        ax5.set_yticks(list(mii5))
        ax5.set_xticklabels(list(l5_cap))
        ax5.set_yticklabels(list(h5_cap))
        ax5.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax5.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        # ax2.axis('off')
        nx5 += 1

    elif nx5 == 1:

        # Target Ship 1 img
        h5 = 25
        widz5 = 40
        img5 = plt.imread('Vsl_Type/TYP5F.png')
        im5 = Image.open('Vsl_Type/TYP5F.png')
        width5, height5 = im5.size
        nii5 = range(1, width5, round(width5 / 8))
        print(list(nii5))
        l5_cap = range(-widz5, widz5, round(widz5 / 4))
        print(list(l5_cap))
        h5_cap = range(h5, 0, -round(h5 / 5))
        mii5 = range(1, height5, round(height5 / 5))
        # ---------------------Plot TS1-------------------
        ax5.cla()
        # plt.axis('off')
        ax5.imshow(img5)
        ax5.set_xticks(list(nii5))
        ax5.set_yticks(list(mii5))
        ax5.set_xticklabels(list(l5_cap))
        ax5.set_yticklabels(list(h5_cap))
        ax5.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax5.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        # ax2.axis('off')
        nx5 += 1


    elif nx5 == 2:

        # Target Ship 1 img
        l5 = 150
        h5 = 25
        img5 = plt.imread('Vsl_Type/TYP5S2.png')
        im5 = Image.open('Vsl_Type/TYP5S2.png')
        width5, height5 = im5.size
        nii5 = range(1, width5, round(width5 / 8))
        l5_cap = range(0, l5, round(l5 / 8))
        h5_cap = range(h5, 0, -round(h5 / 5))
        mii5 = range(1, height5, round(height5 / 5))
        # ---------------------Plot TS1-------------------
        ax5.cla()
        # plt.axis('off')
        ax5.imshow(img5)
        ax5.set_xticks(list(nii5))
        ax5.set_yticks(list(mii5))
        ax5.set_xticklabels(list(l5_cap))
        ax5.set_yticklabels(list(h5_cap))
        ax5.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax5.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        # ax2.axis('off')
        nx5 += 1

    elif nx5 == 3:

        # Target Ship 1 img
        h5 = 25
        widz5 = 40
        img5 = plt.imread('Vsl_Type/TYP5B.png')
        im5 = Image.open('Vsl_Type/TYP5B.png')
        width5, height5 = im5.size
        nii5 = range(1, width5, round(width5 / 8))
        print(list(nii5))
        l5_cap = range(-widz5, widz5, round(widz5 / 4))
        print(list(l5_cap))
        h5_cap = range(h5, 0, -round(h5 / 5))
        mii5 = range(1, height5, round(height5 / 5))
        # ---------------------Plot TS1-------------------
        ax5.cla()
        # plt.axis('off')
        ax5.imshow(img5)
        ax5.set_xticks(list(nii5))
        ax5.set_yticks(list(mii5))
        ax5.set_xticklabels(list(l5_cap))
        ax5.set_yticklabels(list(h5_cap))
        ax5.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
        ax5.text(400, 0, f' TS1: Fast Petrol Vessel\n IMO Number: 10023', fontsize=10, color='white')
        nx5 = 0









    plt.plot(x, y1, label='TS1', linewidth=0.6)
    plt.plot(x, y2, label='TS2', linewidth=0.6)
    plt.plot(x, y3, label='TS3', linewidth=0.6)
    plt.plot(x, y4, label='TS4', linewidth=0.6)
    plt.plot(x, y5, label='TS5', linewidth=0.6)

    ax6.cla()
    ax6.set_title('Axis [0, 0]')
    ax6.set(xlabel='x-label', ylabel='y-label')
    ax6.plot(x, y1, label='TS1', color=colorz1, linewidth=0.6)
    ax6.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
    ax6.legend(loc='upper right')
    ax6.set(ylim=(0, 1))

    ax7.cla()
    ax7.plot(x, y2, label='TS2', color=colorz2, linewidth=0.6)
    ax7.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
    ax7.legend(loc='upper right')
    ax7.set(ylim=(0, 1))

    ax8.cla()
    ax8.plot(x, y3, label='TS3', color=colorz3, linewidth=0.6)
    ax8.legend(loc='upper right')
    ax8.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
    ax8.set(ylim=(0, 1))

    ax9.cla()
    ax9.plot(x, y4, label='TS4', color=colorz4, linewidth=0.6)
    ax9.legend(loc='upper right')
    ax9.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
    ax9.set(ylim=(0, 1))

    ax10.cla()
    ax10.plot(x, y5, label='TS5', color=colorz5, linewidth=0.6)
    ax10.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
    ax10.legend(loc='upper right')
    ax10.set(ylim=(0, 1))

    ax10.legend(loc='upper right')
    ax10.grid(color='w', linestyle='-', linewidth=0.1, alpha=0.3)
    # plt.grid()





ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.legend(loc='upper right')
plt.tight_layout()
# fig.suptitle('Vertically stacked subplots')
plt.axis('off')
plt.show()





