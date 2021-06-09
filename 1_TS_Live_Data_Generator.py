# This Module Generate Target Ship Data for Live Simulation



import csv
import random
import time
import math

num = 0



t1x = [0]
t1y = [0]

t2x = [0]
t2y = [0]

t3x = [0]
t3y = [0]

t4x = [0]
t4y = [0]

t5x = [0]
t5y = [0]



T1y_start = 6919744
T1x_start = 362000
T1_v = 10
T1_h = 0


T2y_start = 6921814
T2x_start = 364341
T2_v = 12
T2_h = 0

T3y_start = 6921094
T3x_start = 364006
T3_v = 15
T3_h = 0

T4y_start = 6920900
T4x_start = 357164
T4_v = 16
T4_h = 0

T5y_start = 6919080
T5x_start = 355318
T5_v = 18
T5_h = 0


fieldnames = ["number","T1y","T1x","T1v","T1h","T2y","T2x","T2v","T2h","T3y","T3x","T3v","T3h","T4y","T4x","T4v","T4h","T5y","T5x","T5v","T5h"]

with open('Data_Input/TS_Data.csv','w') as csv_file:
    csv_writer = csv.DictWriter(csv_file,fieldnames=fieldnames)
    csv_writer.writeheader()

while True:
    with open('Data_Input/TS_Data.csv','a') as csv_file:
        csv_writer = csv.DictWriter(csv_file,fieldnames=fieldnames)

        info = {
            "number": num,
            "T1y": T1y_start,
            "T1x": T1x_start,
            "T1v": T1_v,
            "T1h": T1_h,
            "T2y": T2y_start,
            "T2x": T2x_start,
            "T2v": T2_v,
            "T2h": T2_h,
            "T3y": T3y_start,
            "T3x": T3x_start,
            "T3v": T3_v,
            "T3h": T3_h,
            "T4y": T4y_start,
            "T4x": T4x_start,
            "T4v": T4_v,
            "T4h": T4_h,
            "T5y": T5y_start,
            "T5x": T5x_start,
            "T5v": T5_v,
            "T5h": T5_h,


        }
        csv_writer.writerow(info)
        print(num,T1y_start,T1x_start,T1_v,T1_h,T2y_start,T2x_start,T2_v,T2_h,T3y_start,T3x_start,T3_v,T3_h,T4y_start,T4x_start,T4_v,T4_h,T5y_start,T5x_start,T5_v,T5_h)
        #---------------------------------------------------TS1 --------------------------------------------------
        num += 1
        t1dx = t1x[num-1]-t1x[num-2]
        if t1dx == 0:
            t1dx = 0.001
        t1dy = t1y[num - 1] - t1y[num - 2]

        t1_ang = 220 + round(math.degrees(math.atan(t1dy/t1dx)),3)
        #print('Ts1 ang: ',t1_ang)

        T1_h = t1_ang


        T1y_start = T1y_start - 3
        t1y.append(T1y_start)
        T1x_start = T1x_start - 5
        t1x.append(T1x_start)

        # ---------------------------------------------------TS2 --------------------------------------------------

        t2dx = t2x[num-1]-t2x[num-2]
        if t2dx == 0:
            t2dx = 0.001
        t2dy = t2y[num - 1] - t2y[num - 2]

        t2_ang = 220 +  round(math.degrees(math.atan(t2dy/t2dx)),3)
        #print('Ts2 ang: ',t2_ang)

        T2_h = t2_ang


        T2y_start = T2y_start - 2
        t2y.append(T2y_start)
        T2x_start = T2x_start - 5
        t2x.append(T2x_start)

        # ---------------------------------------------------TS3 --------------------------------------------------
        t3dx = t3x[num-1]-t3x[num-2]
        if t3dx == 0:
            t3dx = 0.001
        t3dy = t3y[num - 1] - t3y[num - 2]

        t3_ang = 220 +  round(math.degrees(math.atan(t3dy/t3dx)),3)
        #print('Ts3 ang: ',t3_ang )

        T3_h = t3_ang


        T3y_start = T3y_start - 2
        t3y.append(T3y_start)
        T3x_start = T3x_start - 5
        t3x.append(T3x_start)

        # ---------------------------------------------------TS4 --------------------------------------------------

        t4dx = t4x[num-1]-t4x[num-2]
        if t4dx == 0:
            t4dx = 0.001
        t4dy = t4y[num - 1] - t4y[num - 2]

        t4_ang =  90 + round(math.degrees(math.atan(t4dy/t4dx)),3)
        #print('Ts4 ang: ',t4_ang)
        T4_h = t4_ang


        T4y_start = T4y_start - 0
        t4y.append(T4y_start)
        T4x_start = T4x_start + 5
        t4x.append(T4x_start)


        # ---------------------------------------------------TS5 --------------------------------------------------

        t5dx = t5x[num - 1] - t5x[num - 2]
        if t5dx == 0:
            t5dx = 0.001
        t5dy = t5y[num - 1] - t5y[num - 2]

        t5_ang =  90 + round(math.degrees(math.atan(t5dy / t5dx)), 3)
        #print('Ts5 ang: ', t5_ang)

        T5_h = t5_ang

        T5y_start = T5y_start + 3
        t5y.append(T4y_start)
        T5x_start = T5x_start + 8
        t5x.append(T5x_start)



    time.sleep(0.1)