import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import pandas as pd
import csv
from matplotlib.path import Path
from matplotlib import transforms
sys.path.append("../../PathPlanning/CubicSpline/")


try:
    import Path_Tracker_Support_Files.cubic_spline_planner
except:
    raise


k = 0.5  # control gain
Kp = 1.0  # speed proportional gain
dt = 0.1  # [s] time difference
L = 60  # [m] Length of the ship
max_steer = np.radians(20.0)  # [rad] max steering angle

show_animation = True

fieldnames = ["Time", "Rad_Ang", "Velo"]

with open('APS_Data_Gen/Rad_RPM.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

fieldnameAu = ["Time", "AY", "AX", "AV", "AH"]

with open('AutoPilot_Data_Gen/Auto_OWN.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnameAu)
    csv_writer.writeheader()



class State(object):
    """
    Class representing the state of a vehicle.
    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0,r=0.0, v=0.0,rho=1014,g=9.80665,L=68,U_norm=8,xg=-3.38,num=0):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.r = r
        self.rho = rho
        self.g = g
        self.L = L
        self.U_norm = U_norm
        self.xg = xg
        self.m = 634.9*10**(-5) * (0.5*rho*L**3)
        self.Izz = 2.63*10**(-5) * (0.5*rho*L**5)
        self.num = num


    def update(self, acceleration, delta):
        """
        Update the state of the vehicle.
        Stanley Control uses bicycle model.
        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)

        with open('APS_Data_Gen/Rad_RPM.csv', 'a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            info = {
                "Time": self.num,
                "Rad_Ang": round(math.degrees(delta),1),
                "Velo": round(self.v,2),
            }
            csv_writer.writerow(info)

        with open('AutoPilot_Data_Gen/Auto_OWN.csv', 'a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnameAu)

            info = {
                "Time": self.num,
                "AY": 6918869 + self.y,
                "AX": 362024 + self.x,
                "AV": round(self.v,2),
                "AH": 90 - round(math.degrees(self.yaw),1),
            }
            csv_writer.writerow(info)




        #print(math.degrees(delta))
        #print(self.num)


        self.x += self.v * np.cos(self.yaw) * dt  #u x t
        self.y += self.v * np.sin(self.yaw) * dt  #v x t
        self.yaw += self.v / L * np.tan(delta) * dt #psi
        #self.yaw = normalize_angle(self.yaw)
        self.v += acceleration * dt

        u1 = self.v * np.cos(self.yaw)  # surge vel.
        v1 = self.v * np.sin(self.yaw)  # sway vel.
        r1 = self.v / L * np.tan(delta)  # yaw vel.

        U = np.sqrt(u1 ** 2 + v1 ** 2)  # speed

        # X - Coefficients
        Xu_dot = -31.0323 * 10 ** (-5) * (0.5 * self.rho * self.L ** 3)
        Xuu = -167.7891 * 10 ** (-5) * (0.5 * self.rho * self.L ** 2)
        Xvr = 209.5232 * 10 ** (-5) * (0.5 * self.rho * self.L ** 3)
        Xdel = -2.382 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 2) * (U ** 2))
        Xdd = -242.1647 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 2) * (U ** 2))
        # N - Coefficients
        Nv_dot = 19.989 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 4))
        Nr_dot = -29.956 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 5))
        Nuv = -164.080 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 3))
        Nur = -175.104 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 4))
        Nrr = -156.364 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 5))
        Nrv = -579.631 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 4))
        Ndel = -166.365 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 3) * (U ** 2))
        # Y - Coefficients
        Yv_dot = -700.907 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 3))
        Yr_dot = -52.018 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 4))
        Yuv = -1010.163 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 2))
        Yur = 233.635 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 3))
        Yvv = -316.746 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 2))
        Yvr = -1416.083 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 3))
        Yrv = -324.593 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 3))
        Ydel = 370.6 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 2) * (U ** 2))

        # Hydrodynamic
        Ta = -Xuu * self.U_norm * self.U_norm  # Assumption: constant
        Xhyd = Xuu * u1 * np.abs(u1) + Xvr * v1 * r1 + Ta
        Yhyd = Yuv * np.abs(u1) * v1 + Yur * u1 * r1 + Yvv * v1 * np.abs(v1) + Yvr * v1 * np.abs(r1) \
               + Yrv * r1 * np.abs(v1)
        Nhyd = Nuv * np.abs(u1) * v1 + Nur * np.abs(u1) * r1 + Nrr * r1 * np.abs(r1) \
               + Nrv * r1 * np.abs(v1)

        Xrudder = 0  # Xdd * rudder_angle * rudder_angle + Xdel * rudder_angle
        Yrudder = Ydel * delta
        Nrudder = Ndel * delta

        H = np.array([[self.m - Xu_dot, 0, 0, 0],
                      [0, self.m - Yv_dot, self.m * self.xg - Yr_dot, 0],
                      [0, self.m * self.xg - Nv_dot, self.Izz - Nr_dot, 0],
                      [0, 0, 0, 1]])
        f = np.array([Xhyd + self.m * (v1 * r1 + self.xg * r1 ** 2) + Xrudder,
                      Yhyd - self.m * u1 * r1 + Yrudder,
                      Nhyd - self.m * self.xg * u1 * r1 + Nrudder,
                      r1])
        # output = np.matmul(np.linalg.inv(H), f).reshape((4))
        output = np.linalg.solve(H, f)

        u0 = u1 + output[0] * dt
        v0 = v1 + output[1] * dt

        r = r1 + output[2] * 180. / math.pi * dt

        self.v = np.sqrt(u0 ** 2 + v0 ** 2)
        self.r = r

        self.num += 1


        # position[2] = position[2] + velocity[2] * dt
        #
        # if position[2] > 180:
        #     position[2] = position[2] - 360
        #
        # if position[2] < -180:
        #     position[2] = position[2] + 360

        # rot_matrix = np.array(
        #     [[math.cos(position[2] * math.pi / 180),
        #       -math.sin(position[2] * math.pi / 180)],
        #      [math.sin(position[2] * math.pi / 180),
        #       math.cos(position[2] * math.pi / 180)]])

        # pdot = np.array([[velocity[0]], [velocity[1]]])
        # pdot = pdot.reshape(2,1)
        # pdot = np.array([velocity[0], velocity[1]])
        # XYvel = np.dot(rot_matrix, pdot)
        # position[0] += XYvel[0] * dt
        # position[1] += XYvel[1] * dt



def pid_control(target, current):
    """
    Proportional control for the speed.
    :param target: (float)
    :param current: (float)
    :return: (float)
    """
    return Kp * (target - current)


def stanley_control(state, cx, cy, cyaw, last_target_idx):
    """
    Stanley steering control.
    :param state: (State object)
    :param cx: ([float])
    :param cy: ([float])
    :param cyaw: ([float])
    :param last_target_idx: (int)
    :return: (float, int)
    """
    current_target_idx, error_front_axle = calc_target_index(state, cx, cy)

    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx


    # theta_e corrects the heading error
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw)
    # theta_d corrects the cross track error
    theta_d = np.arctan2(k * error_front_axle, state.v)
    # Steering control
    delta = theta_e + theta_d

    return delta, current_target_idx


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def calc_target_index(state, cx, cy):
    """
    Compute index in the trajectory list of the target.
    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :return: (int, float)
    """
    # Calc front axle position
    fx = state.x + L * np.cos(state.yaw)
    fy = state.y + L * np.sin(state.yaw)

    # Search nearest point index
    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]
    d = np.hypot(dx, dy)
    target_idx = np.argmin(d)

    # Project RMS error onto front axle vector
    front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                      -np.sin(state.yaw + np.pi / 2)]
    error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

    return target_idx, error_front_axle


def main():
    """Plot an example of Stanley steering control on a cubic spline."""
    #  target course
    data = pd.read_csv('APS_Data_Gen/Path_Gen.csv')
    x89 = data['Time']
    numberiii = len(x89)-1

    X_csv = []
    Y_csv = []

    for i in range(numberiii):
        X_csv.append(data['x'][x89][i])
        Y_csv.append(data['y'][x89][i])

    print('CSV X:',X_csv)
    print('CSV Y:', Y_csv)

    ts1gen_x = data['x'][x89][numberiii]
    ts1gen_y = data['y'][x89][numberiii]

    print('cvx',ts1gen_x)

    pos_path = 'Pos2.xlsx'
    df = pd.read_excel(pos_path)

    x = []
    y = []
    vp = []

    split_x = X_csv
    split_y = Y_csv
    split_v = df['P'].unique()



    v_len = len(split_v)

    print(v_len)
    for i in range(v_len):
        vp.append(round(split_v[i]))



    ttl = len(split_x)
    leng = round(ttl / 15)
    print(leng)

    for i in range(leng):
        x.append(round(split_x[i * 15]))
        y.append(round(split_y[i * 15]))


    print(x)
    print(y)
    print(vp)

    plt.scatter(x, y,label="Way Points")
    plt.xlabel("East(m)")
    plt.ylabel("North(m)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


    ax = x
    ay = y

    cx, cy, cyaw, ck, s = Path_Tracker_Support_Files.cubic_spline_planner.calc_spline_course(
        ax, ay, ds=0.1)

    target_speed = 100  # [m/s]

    max_simulation_time = 2000

    # Initial state
    state = State(x=0, y=0, yaw=np.radians(00.0), v=0.0)

    last_idx = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_idx, _ = calc_target_index(state, cx, cy)
    i = 0
    ppj = 0
    img = plt.imread('Path_Planner_Support_Files/Autop_blk.jpg')
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)

    while max_simulation_time >= time and last_idx > target_idx:
       # ppj = round(i)
        #print(ppj)
        #target_speeds = vp[ppj]
        #print(target_speeds)
        ai = 0.6*pid_control(target_speed, state.v)
        di, target_idx = stanley_control(state, cx, cy, cyaw, target_idx)
        i+=1


        #print(target_idx)
        # if target_idx > 1800:
        #     ai =
        state.update(ai, di)

        time += dt

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)



        if show_animation:  # pragma: no cover
            plt.cla()

            # for stopping simulation with the esc key.



            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            ax.imshow(img, extent=[-4122, 6028, -400, 4675], alpha=0.96)

            plt.gcf().canvas.mpl_connect('key_release_event',lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, "-r", label="Ship Course Plotted")
            plt.plot(x, y, "-g", label="Trajectory Traced")
            #plt.scatter(x, y, color='g', s=2, alpha=0.9)
            #print(90-math.degrees(yaw[-1]))


            ov_mk_rtt = transforms.Affine2D().rotate_deg(-(90-math.degrees(yaw[-1])))

            def_marker_comp = Path(
                [[-0.005, -0.02], [0.005, -0.02], [0.005, 0.01], [0, 0.02], [-0.005, 0.01], [0, 0], ],
                [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
            com_mkr = def_marker_comp.transformed(ov_mk_rtt)
            plt.scatter(x[-1], y[-1], marker=com_mkr, color='w', s=1000, alpha=0.9)

            plt.scatter(4630, 2600, marker=com_mkr, color='w', s=3000, alpha=0.9)

            plt.scatter(-200, 2600, marker=def_marker_comp, color='Black', s=100, alpha=0.9, label="Ship")

            dltaa = np.clip(di, -max_steer, max_steer)

            plt.text(3750, 800,
                     f'Own Ship: Ulstein \nVelocity = {round(v[-1]*0.2, 1)}Knots\nHeading = {round(90-math.degrees(yaw[-1]),1)}\nRudder Angle = {round(math.degrees(dltaa))}',
                     fontsize=10, color='white')


            # corc = 90-math.degrees(yaw[-1])
            # sine_degt1 = math.sin(math.radians(corc))
            # cos_degt1 = math.cos(math.radians(corc))
            # z1 = 10 * sine_degt1
            # p1 = 10 * cos_degt1
            # plt.quiver(2750, 550, z1, p1, scale=100, color='black', pivot='middle')

            plt.plot(cx[target_idx], cy[target_idx], "xg", label="Follow Marker")
            plt.axis("equal")
            plt.grid(False)
            plt.title("Speed[Knots]:" + str(state.v * 0.2)[:4])
            plt.legend()
            plt.pause(0.001)



    # Test
    assert last_idx >= target_idx, "Cannot reach goal"

    if show_animation:  # pragma: no cover



        plt.plot(cx, cy, "-r")
        plt.plot(x, y, ".b")
        plt.legend()
        plt.xlabel("East[m]")
        plt.ylabel("North[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(t, [iv * 1 for iv in v], "-r")
        plt.xlabel("Simulation Time[s]")
        plt.ylabel("Speed[Knots]")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()