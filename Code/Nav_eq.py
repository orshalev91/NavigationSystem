import numpy as np
from pyquaternion import quaternion as q


# function for calculation of the rotation matrix for rotaion from tangent frame to body frame.

def rt2b(roll, pitch, yaw):
    cr = np.cos(roll)
    sr = np.sin(roll)

    cp = np.cos(pitch)
    sp = np.sin(pitch)

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    r = np.zeros((3,3))
    r[0] = [cy*cp, sy*cp, -sp]
    r[1] = [-sy*cr+cy*sp*sr, cy*cr+sy*sp*sr, cp*sr]
    r[2] = [sy*sr+cy*sp*cr, -cy*sr+sy*sp*cr, cp*cr]
    return r


# function that converts DCM to quaternions based on the matlab site equations

def dcm2q(rnb):
    q8 = np.zeros(4)
    q8[3] = 0.5 * np.sqrt(np.abs(1 + np.sum(np.diag(rnb))))
    q8[0] = (rnb[2, 1] - rnb[1, 2]) / (4 * q8[3])
    q8[1] = (rnb[0, 2] - rnb[2, 0]) / (4 * q8[3])
    q8[2] = (rnb[1, 0] - rnb[0, 1]) / (4 * q8[3])
    return q.Quaternion(q8)


# function that calculates gravity according to your location on earth, default value, you are in beer sheva lat

def gravity(latitude=31.26, h=0):
    lambda1 = np.pi/180*latitude  # convert to radians
    gamma = 9.780327*(1+0.0053024 * (np.sin(lambda1) ** 2)-0.0000058 * np.sin(2*lambda1) ** 2)

    g = gamma-(3.0877e-6-0.004e-6 * np.sin(lambda1) ** 2) * h+0.072e-12 * h ** 2
    g = np.transpose(np.array([0, 0, -g]))
    return g


# function that converts quaternios to DCM based on the matlab site equations

def q2dcm(qua: q.Quaternion):
    p = np.zeros((6, 1))
    R = np.zeros((3, 3))
    p[0] = qua[0]**2
    p[1] = qua[1]**2
    p[2] = qua[2]**2
    p[3] = qua[3]**2

    p[4] = p[1] + p[2]

    if p[0] + p[3] + p[4] != 0:
        p[5] = 2 / (p[0] + p[3] + p[4])
    else:
        p[5] = 0

    R[0, 0] = 1 - p[5] * p[4]
    R[1, 1] = 1 - p[5] * [p[0] + p[2]]
    R[2, 2] = 1 - p[5] * [p[0] + p[1]]

    p[0] = p[5] * qua[0]
    p[1] = p[5] * qua[1]
    p[4] = p[5] * qua[2] * qua[3]
    p[5] = p[0] * qua[1]

    R[0, 1] = p[5] - p[4]
    R[1, 0] = p[5] + p[4]

    p[4] = p[1] * qua[3]
    p[5] = p[0] * qua[2]

    R[0, 2] = p[5] + p[4]
    R[2, 0] = p[5] - p[4]

    p[4] = p[0] * qua[3]
    p[5] = p[1] * qua[2]

    R[1, 2] = p[5] - p[4]
    R[2, 1] = p[5] + p[4]
    return R


# Function that implements the navigation equations of an INS.
#
# Inputs:
# x         Current navigation state [position (NED), velocity (NED), attitude (Quaternion)]
# u         Measured inertial quantities [Specific force (m/s^2), Angular velocity (rad/s)]
# Ts        Sampling period (s)
#
# Output:
# x         Updated navigation state [position (NED), velocity (NED), attitude (Quaternion)]

def nav_eq(x, u, ts):
    g_t = np.array(gravity(), ndmin=2).transpose()
    x_q = q.Quaternion(x[6], x[7], x[8], x[9])
    x_main = np.array(x[0:6], ndmin=2)
    f_t = np.matmul(q2dcm(x_q), u[0:3])
    acc_t = f_t - g_t

    # state space model matrices
    a = np.eye(6)
    a[0, 3] = ts
    a[1, 4] = ts
    a[2, 5] = ts

    # matrix form of kinematic equations used to calculate progress
    b_1 = np.append(np.eye(3), np.zeros((3, 3)), axis=0) * (ts ** 2 / 2)
    b_2 = np.append(np.zeros((3, 3)), np.eye(3), axis=0) * ts
    b = b_1 + b_2
    x_1 = np.array(np.matmul(a, x_main))
    x_2 = np.matmul(b, acc_t)
    x[0:6] = x_1 + x_2
    w_tb = u[3:6]

    P = w_tb[0] * ts
    Q = w_tb[1] * ts
    R = w_tb[2] * ts

    # OMEGA matrix definition as described in reference and final report
    omega = np.zeros((4, 4))
    omega[0, 0:4] = [0, R*0.5, -Q*0.5, P*0.5]
    omega[1, 0:4] = [-R*0.5, 0, P*0.5, Q*0.5]
    omega[2, 0:4] = [Q*0.5, -P*0.5, 0, R*0.5]
    omega[3, 0:4] = [-P*0.5, -Q*0.5, -R*0.5, 0]

    v = np.linalg.norm(w_tb) * ts

    if v != 0:
        x[6:10] = np.matmul((np.cos(v/2)*np.eye(4) + 2/v*np.sin(v/2)*omega), x[6:10])
    return x
