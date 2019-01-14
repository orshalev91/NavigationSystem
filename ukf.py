import numpy as np
import csv
from scipy import linalg
import math
''' Based on https://www.mathworks.com/matlabcentral/fileexchange/18217-learning-the-unscented-kalman-filter '''


class UKF:
    NUMOFSTATES = 0
    def __init__(self, numofstates, f, h, Q, R):
        """

        :param numofstates:
        :param f: lambda function of the state transition
        :param h: lambda function of the measurement matrix
        :param Q: Covariance matrix of state transition uncertainty
        :param R: Covariance matrix of measurement noise
        """
        self.NUMOFSTATES = numofstates
        self.f = f
        self.h = h
        self.P = np.eye(numofstates)
        self.Q = Q
        self.R = R

    def estimate(self, x, z):
        """
        Runs the propagation and the measurement update stage of the UKF
        :param x: The current estimation
        :param z: The measurement
        :return: estimated location, xk+1|xk, zk+1
        """
        L = len(x)
        m = len(z)
        alpha = 0.001
        ki = 0
        beta = 2
        lamb = (alpha ** 2) * (L+ki) - L
        c = L + lamb
        a = np.array([[lamb / c]])
        b = 0.5/c + np.zeros((1, 2 * L))
        Wm = np.concatenate((a, b), axis=1)
        Wc = np.copy(Wm)
        Wc[0] += (1 - alpha ** 2 + beta)
        c = np.sqrt(c)
        X = self._sigmas(c, self.P, x)
        x1, X1, P1, X2 = self._ut(self.f, X, Wm, Wc, L, self.Q)
        z1, Z1, P2, Z2 = self._ut(self.h, X1, Wm, Wc, m, self.R)
        P12 = np.dot(np.dot(X2, np.diag(Wc[0, :])), Z2.transpose())
        K = np.dot(P12, linalg.inv(P2))
        xk1 = x1 + np.dot(K ,(z - z1))
        self.P = P1 - np.dot(K, P12.transpose())
        return xk1

    def reset_error_covariance(self):
        self.P = np.eye(self.NUMOFSTATES)

    def propagate(self, x):
        """
        Runs the propagation only of the UKF
        :param x: The current estimation
        :return: estimated location xk+1|xk
        """
        L = len(x)
        m = len(z)
        alpha = 0.001
        ki = 0
        beta = 2
        lamb = (alpha ** 2) * (L + ki) - L
        c = L + lamb
        a = np.array([[lamb / c]])
        b = 0.5 / c + np.zeros((1, 2 * L))
        Wm = np.concatenate((a, b), axis=1)
        Wc = np.copy(Wm)
        Wc[0, 0] += (1 - alpha ** 2 + beta)
        c = np.sqrt(c)
        X = self._sigmas(c, self.P, x)
        x1, X1, P1, X2 = self._ut(self.f, X, Wm, Wc, L, self.Q)
        return x1

    @staticmethod
    def _ut(f, X, Wm, Wc, n, R):
        """
        Unscented Transformation
        :param f: nonlinear map
        :param X: sigma points
        :param Wm: weights for mean
        :param Wc: weights for covraiance
        :param n: numer of outputs of f
        :param R: additive covariance
        :return:
            y: transformed mean
            Y: transformed smapling points
            P: transformed covariance
            Y1: transformed deviations
        """
        L = len(X[0])
        Y = np.zeros((n, L))
        y = np.zeros((n, 1))
        for k in range(L):
            Y[:, [k]] = f(X[:, [k]])
            v = Wm[0, k]
            y += v * Y[:, [k]]
        # Y = (np.array(map(f, X.transpose()))[:, :, 0]).transpose()
        # y = np.array(map(lambda v, y: v * y, Y, Wm))
        # y = np.sum(y, axis=0)
        # y = y.reshape((-1, 1))
        Y1 = Y - np.repeat(y, L, axis=1)
        a = np.dot(Y1, np.diag(Wc[0, :]))
        P = np.dot(a, Y1.transpose()) + R
        return y, Y, P, Y1

    @staticmethod
    def _sigmas(c, P, x):
        """
        Sigma points around reference point
        :param c: coefficient
               x: reference point
               P: covariance
        :return: X: Sigma points
        """
        A = c * np.linalg.cholesky(P).transpose()
        Y = np.repeat(x, x.size, axis=1)
        X1 = np.concatenate((x, Y+A), axis=1)
        X = np.concatenate((X1, Y-A), axis=1)
        return X


if __name__ == '__main__':
    # print UKF._sigmas(0.0017, np.eye(3), np.array([[0.2037], [-0.034], [0.9284]]))
    f = lambda x: np.array([[x[0, 0]+x[2,0]*dt +0.5*(dt**2)*(np.cos(theta)*x[4, 0])], [x[1, 0]+x[3, 0]*dt+0.5*(dt**2)*(np.sin(theta)*x[4, 0])], [x[2, 0]+(dt*np.cos(theta)*x[4, 0])], [x[3, 0]+(dt*np.sin(theta)*x[4, 0])], [x[4, 0]], [x[5, 0]]])
    h = lambda x: np.array([[x[4, 0]], [x[5, 0]]])  # np.array([[np.cos(x[0])], [x[1]]])

    estimator = UKF(6, f, h, Q=np.eye(6) * 0.1, R=np.eye(1) * 0.1) # X Y Vx Vy ax gyz
    # s = np.array([[.0], [.0]]) + 0.1 * np.random.randn(2, 1)
    theta = 0.0
    dt = 0.0
    sV = []
    zV = []
    xV = []
    tV = [] #time vector
    xk1k = []
    thetaV = [] # theta vector
    prevTime = 0.0
    # x = s + 0.1 * np.random.randn(2, 1)
    z = np.array([[.0], [.0]])
    x = np.array([[.0], [.0], [.0], [.0], [.0], [.0]])
    tV.append(dt)
    zV.append(z)
    xV.append(x)
    thetaV.append(theta)
    with open('measurements.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                a = float(row[1])  #assertign readings to z
                b = float(row[3])
                currTime = float(row[0])
                dt = float(currTime-prevTime)
                z = np.array([[a], [b]])
                zV.append(z)
                theta += b*dt
                # print(a)
                x = estimator.estimate(x, z)
                xV.append(x)

                # print(dt)# print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                line_count += 1
                prevTime = currTime

                tV.append(currTime)
                thetaV.append(theta)
        print(f'Processed {line_count} lines.')
        tR = np.array(tV)
        ar = np.array(xV)
        thetaR = np.array(thetaV)
    with open('results.csv', 'w', newline='') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(['dt', 'X', 'Y', 'V', 'Theta', 'Sin', 'Cos'])
        for x in range(len(tV)):
            results_writer.writerow([tR[x], ar[x, 0, 0], ar[x, 1, 0], math.sqrt(ar[x, 2, 0]**2+ar[x, 3, 0]**2),
                                     thetaR[x], math.sin(thetaR[x]), math.cos(thetaR[x])])



    # x = np.array([[.0], [.0], [.0], [.0], [1], [1]])
    # xV.append(x)
    #
    # z = np.array([[1], [1]])
    # zV.append(z)
    # x = estimator.estimate(x, z)
    # xV.append(x)
    # # x = f(x)
    # z = np.array([[1], [1]])
    # zV.append(z)
    # x = estimator.estimate(x, z)
    # xV.append(x)
    # # x = f(x)
    # z = np.array([[1], [1]])
    # zV.append(z)
    # x = estimator.estimate(x, z)
    # xV.append(x)
    # # x = f(x)
    # z = np.array([[1], [1]])
    # zV.append(z)
    # x = estimator.estimate(x, z)
    # xV.append(x)
    # x = f(x)

    # for k in range(200):
    #     z = h(s)#[:, :, 0]
    #     z += 0.1 * np.random.randn(1, 1)
    #     sV.append(s)
    #     zV.append(z)
    #     # xk1k.append(estimator.propagate(x))31
    #     x = estimator.estimate(x, z)
    #     xV.append(x)
    #     s = f(s)
    #     s = s + 0.1 * np.random.randn(2,1)
    import matplotlib.pyplot as plt


    plt.figure(0)
    plt.subplot(611)
    # plt.plot(np.array(sV)[:, 0, 0], 'b-')
    plt.plot(ar[:, 0, 0], 'r--')
    plt.subplot(612)
    # plt.plot(np.array(sV)[:, 1, 0], 'b-')
    plt.plot(ar[:, 1, 0], 'r--')
    plt.subplot(613)
    # plt.plot(np.array(sV)[:, 2, 0], 'b-')
    plt.plot(ar[:, 2, 0], 'r--')
    plt.subplot(614)
    # plt.plot(np.array(sV)[:, 2, 0], 'b-')
    plt.plot(ar[:, 3, 0], 'r--')
    plt.subplot(615)
    plt.plot(np.array(zV)[:, 0, 0], 'b-')
    plt.plot(ar[:, 4, 0], 'r--')
    plt.subplot(616)
    plt.plot(np.array(zV)[:, 1, 0], 'b-')
    plt.plot(ar[:, 5, 0], 'r--')
    # plt.show()
    plt.figure(1)
    plt.grid()
    plt.plot(ar[:, 0, 0], ar[:, 1, 0], 'ro')
    # plt.axis([0, 6, 0, 20])
    plt.show()