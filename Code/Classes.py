import numpy as np


class Settings:
    #############################################
    ##              GENERAL PARAMETERS         ## 
    #############################################
    init_heading = -2.0
    A = np.matrix('0 1 0; 0 0 1')

    #############################################
    ##             FILTER PARAMETERS           ##
    #############################################

    # Process noise covariance (Q)
    sigma_acc = 0.05  # [m/s^2]
    sigma_gyro = 0.1*np.pi/180  # [rad/s]
    sigma_acc_bias = 0.0001     # [m/s^2]
    sigma_gyro_bias = 0.01*np.pi/180  # [rad/s]

    # GNSS position measurement noise covariance (R)
    sigma_gps = 3/np.sqrt(3)  # [m]

    # Initial Kalman filter uncertainties (standard deviantions)
    arr = np.array([1, 1, 20])
    factp = np.zeros(8)
    factp[0] = 0.5                                  # Position [m]
    factp[1] = 0.5                                  # Velocity [m/s]
    factp[2:5] = ((np.pi/180)*arr)               # Attitude (roll,pitch,yaw) [rad]
    factp[5] = 0.01                               # Accelerometer biases [m/s^2]
    factp[6] = (0.05*np.pi/180)                       # Gyro biases [rad/s]
    factp[7] = 0.00001


class OutData:
    x_h = []
    diag_p = []
    delta_u_h = []
    magnetometer = []


class Filter:
    p = []
    q1 = []
    q2 = []
    r = []
    h = []


class InData:
    imu = []
    magnetometer = []
    gnss = []
    gnssref = []
    t = []
    tgnss = []
    ctrl_gnss = 0
    name = ''
