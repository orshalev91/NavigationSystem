import numpy as np
import Nav_eq
from numpy import linalg as la
from scipy.linalg import block_diag
import navpy
import Classes
import copy
from pyquaternion import quaternion as q

# function call: out_data=GPSaidedINS(in_data,settings)
#
# This function integrates GNSS and IMU data. The data fusion
# is based upon a loosely-coupled feedback GNSS-aided INS approach.
#
# Input struct:
# in_data.GNSS.pos_ned      GNSS-receiver position estimates in NED
#                           coordinates [m]
# in_data.GNSS.t            Time of GNSS measurements [s]
# (in_data.GNSS.HDOP        GNSS Horizontal Dilution of Precision [-])
# (in_data.GNSS.VDOP        GNSS Vertical Dilution of Precision [-])
# in_data.IMU.acc           Accelerometer measurements [m/s^2]
# in_data.IMU.gyro          Gyroscope measurements [rad/s]
# in_data.IMU.t             Time of IMU measurements [s]
#
# Output struct:
# out_data.x_h              Estimated navigation state vector [position; velocity; attitude]
# out_data.delta_u_h        Estimated IMU biases [accelerometers; gyroscopes]
# out_data.diag_P           Diagonal elements of the Kalman filter state
#                           covariance matrix.


def sensors_data_read(data, in_data: Classes.InData, settings: Classes.Settings, out_data: Classes.OutData, kalman_filter :Classes.Filter):
    if len(in_data.imu) < 101:  # gathering 100 IMU readings to find average bias
        if data[4] != "":  # check if its IMU data or GPS - data[4] is IMU
            # append accelerometer and gyroscope data
            in_data.imu.append(np.array([float(data[5]), float(data[4]), float(data[6]),
                                         float(data[10]), float(data[11]), float(data[12])]))
            # append magnetometer data
            in_data.magnetometer.append(np.array([float(data[7]), float(data[8]), float(data[9])]))
            in_data.t.append((float(data[0])/100))  # append dt
        if data[16] != "":  # GPS data
            in_data.gnssref.append([float(data[16]), float(data[17])])
        if len(in_data.imu) > 100 and len(in_data.gnssref) != 0:  # after gathering 100 readings send the data to setup
            gps_aided_ins_setup(in_data, settings, out_data, kalman_filter)
    else:  # normal run after setup
        if data[4] != "":  # check if its IMU data
            in_data.imu.append(np.array([float(data[5]), float(data[4]), float(data[6]),
                                         float(data[10]), float(data[11]), float(data[12])]))
            in_data.magnetometer.append(np.array([float(data[7]), float(data[8]), float(data[9])]))
            in_data.t.append((float(data[0])/100))
            gps_aided_ins(in_data, out_data, kalman_filter, 1)  # run navigation step with IMU data input
        if data[16] != "":
            # append GPS data after converting long/lat to NED coordinate
            in_data.gnss.append(navpy.lla2ned(float(data[16]), float(data[17]), 0, in_data.gnssref[0][0], in_data.gnssref[0][1], 0))
            in_data.tgnss.append((float(data[0])/100))  # dt
            gps_aided_ins(in_data, out_data, kalman_filter, 0)  # run navigation step with GPS data input

    return


def gps_aided_ins_setup(in_data: Classes.InData, settings: Classes.Settings, out_data: Classes.OutData,
                        kalman_filter: Classes.Filter):

    #  Initialization state vector
    x_h = copy.deepcopy(init_navigation_state(in_data.imu, settings, in_data))

    # Initialize Kalman filter
    [kalman_filter.p, kalman_filter.q1, kalman_filter.q2, kalman_filter.r, kalman_filter.h] = init_filter(settings)

    # # Allocating memory for output data
    out_data.x_h.append(copy.deepcopy(np.array(x_h)))
    out_data.diag_p.append(copy.deepcopy(list(np.diag(kalman_filter.p))))
    out_data.delta_u_h.append(np.zeros((6, 1)))

    # Information fusion
    in_data.ctrl_gnss_data = 0


def gps_aided_ins(in_data: Classes.InData, out_data: Classes.OutData, kalman_filter: Classes.Filter, is_imu):
    if is_imu:  # if the system received IMU data
        # Sampling period
        ts = in_data.t[-1]-in_data.t[-2]
        # calibrate the sensor measurements using bias estimate
        delta_u_h = copy.deepcopy(np.array(out_data.delta_u_h[-1]))
        u_h = np.transpose(np.array(in_data.imu[-1], ndmin=2) + np.transpose(delta_u_h))
        xtt = copy.deepcopy(out_data.x_h[-1])
        x_h = copy.deepcopy(Nav_eq.nav_eq(xtt, u_h, ts))
        # get state space model matrices
        [f, g] = state_space_model(x_h, u_h, ts)
        # time update of the kalman filter state covariance
        kalman_filter.p = np.matmul(np.matmul(f, kalman_filter.p), np.transpose(f)) + np.matmul(np.matmul(g, block_diag(kalman_filter.q1, kalman_filter.q2)), np.transpose(g))

    else: # if the system received GPS data
        # calculate kalman filter gain
        k = np.matmul(np.matmul(kalman_filter.p, np.transpose(kalman_filter.h)),
                      np.linalg.inv((np.matmul(np.matmul(kalman_filter.h, kalman_filter.p),
                                               np.transpose(kalman_filter.h))+kalman_filter.r)))
        # update the pertubation state estimate
        x_h = np.array(out_data.x_h[-1]).transpose()[0]
        # xh3 = np.array(x_h[0:3]).transpose()[0]
        z = np.concatenate((np.zeros(9), copy.deepcopy(out_data.delta_u_h[-1])), axis=None) + \
            np.matmul(k, (in_data.gnss[-1] - x_h[0:3]))
        # correct the navigation state using current pertubation estimates
        x_h[0:6] = x_h[0:6] + z[0:6]
        x_h[6:10] = gamma(x_h[6:10], z[6:9])
        delta_u_h = z[9:15]
        x_h = np.array(x_h, ndmin=2).transpose()
        # update the kalman filter state covariance
        kalman_filter.p = np.matmul((np.eye(15)-np.matmul(k, kalman_filter.h)), kalman_filter.p)

    # save the data to the output data structure
    out_data.x_h.append(copy.deepcopy(x_h))
    out_data.diag_p.append(copy.deepcopy(list(np.diag(kalman_filter.p))))
    out_data.delta_u_h.append(copy.deepcopy(delta_u_h))


###### Sub Functions ######

# #  Init navigation state
def init_navigation_state(u, settings, in_data: Classes.InData):
    #  Calculate the roll and pitch
    f = np.array([[np.mean(u[0][0:100])], [np.mean(u[1][0:100])], [np.mean(u[2][0:100])], [np.mean(u[3][0:100])],
                  [np.mean(u[4][0:100])], [np.mean(u[5][0:100])]])
    roll = np.arctan2(-f[1], -f[2])
    pitch = np.arctan2(f[0], la.norm(f[1:3]))
    rb2t = np.transpose(Nav_eq.rt2b(roll, pitch, settings.init_heading))
    qua = Nav_eq.dcm2q(rb2t)
    x_h = np.zeros(10)
    x_h[6:10] = qua.elements

    return np.array(x_h, ndmin=2).transpose()


# Init filter
def init_filter(settings):
    # Kalman filter state matrix
    p = np.zeros((15, 15))
    p[0:3, 0:3] = settings.factp[0]**2*np.eye(3)
    p[3:6, 3:6] = settings.factp[1]**2*np.eye(3)
    p[6:9, 6:9] = np.power(np.diag(settings.factp[2:5]), 2)
    p[9:12, 9:12] = settings.factp[5]**2*np.eye(3)
    p[12:15, 12:15] = settings.factp[6] ** 2 * np.eye(3)
    # p[15:18, 15:18] = settings.factp[7] ** 2 * np.eye(3)

    # Process noise covariance
    q1 = np.zeros((6, 6))
    q1[0:3, 0:3] = np.power(settings.sigma_acc, 2)
    q1[3:6, 3:6] = np.power(settings.sigma_gyro, 2)

    q2 = np.zeros((6, 6))
    q2[0:3, 0:3] = np.dot(settings.sigma_acc_bias, settings.sigma_acc_bias)*np.eye(3)
    q2[3:6, 3:6] = np.dot(settings.sigma_gyro_bias, settings.sigma_gyro_bias)*np.eye(3)
    # GNSS- receiver position measurment noise
    r = np.dot(settings.sigma_gps, settings.sigma_gps)*np.eye(3)
    # Observation matrix
    h = np.concatenate((np.eye(3), np.zeros((3, 12))), axis=1)

    return np.array([p, q1, q2, r, h])


# State transition matrix
def state_space_model(x, u, ts):
    # Convert quaternion to DCM
    rb2t = Nav_eq.q2dcm(x[6:10])
    # Transform measured force to force in the the tangent plane coordinate system
    f_t = np.matmul(rb2t, u[0:3])
    st = np.array([[0, -f_t[2], f_t[1]], [f_t[2], 0, -f_t[0]], [-f_t[1], f_t[0], 0]])

    fc = np.zeros((15, 15))
    fc[0:3, 3:6] = np.eye(3)
    fc[3:6, 6:9] = st[:, :]
    fc[3:6, 9:12] = rb2t
    fc[6:9, 12:15] = -rb2t
    # Approximation of the discrete time state transition matrix
    f = np.eye(15)+ts*fc
    # Noise gain matrix
    g = np.zeros((15, 12))
    g[3:6, 0:3] = rb2t
    g[6:9, 3:6] = -rb2t
    g[9:12, 6:9] = np.eye(3)
    g[12:15, 9:12] = np.eye(3)
    return [f, g]


# Quaternion error correction
def gamma(q, epsilon):
    # Convert quaternion to DCM
    r = Nav_eq.q2dcm(q)
    # Construct skew symmetric matrix
    omega = np.array([[0, -epsilon[2], epsilon[1]], [epsilon[2], 0, -epsilon[0]], [-epsilon[1], epsilon[0], 0]])
    # Correct the DCM matrix
    r = np.matmul((np.eye(3) - omega), r)
    # Calculate the correct quaternions
    q = Nav_eq.dcm2q(r)
    return q
