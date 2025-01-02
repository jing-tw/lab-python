# This kalman filter module was copied from https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python.
import numpy as np
import matplotlib.pyplot as plt
import csv
# import pandas as pd


class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
            (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def example():
    dt = 1.0/60
    F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([0.5]).reshape(1, 1)

    x = np.linspace(-10, 10, 100)
    measurements = - (x**2 + 2*x - 2)  + np.random.normal(0, 2, 100)

    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)
    predictions = []

    for z in measurements:
        predictions.append(np.dot(H,  kf.predict())[0])
        kf.update(z)

    plt.plot(range(len(measurements)), measurements, label = 'Measurements')
    plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
    plt.legend()
    plt.show()

def readData(strFilename, strSheetName, strColumn, bShow=False):
    df = pd.read_excel(strFilename, sheet_name=strSheetName, dtype = 'str')
    print('Column headings:')
    print(df.columns)

    lstSamples = []
    for i in df.index:
        dictSample = {}
        for key in df.columns:
            dictSample[key] = df[key][i]
        
        floatSample = float(dictSample[strColumn])
        print('floatSample[' + str(i) + '] = ', floatSample)
        lstSamples.append(floatSample)

    if bShow:
        plt.plot(range(len(lstSamples)), lstSamples, label = 'lstSamples')
        plt.legend()
        plt.show()
    return lstSamples

def example2(measurements):
    # dt = 1.0/60
    dt = 1.0/240
    F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([0.5]).reshape(1, 1)


    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)
    predictions = []

    for z in measurements:
        predictions.append(np.dot(H,  kf.predict())[0])
        kf.update(z)

    print('predictions = ', predictions)
    plt.plot(range(len(measurements)), measurements, label = 'Measurements')
    plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # test 1
    # lstSamples = readData('data.xlsx','1', 'Temp')
    # print('lstSamples = ', lstSamples)
    # example2(lstSamples)

    # test 2
    # lstSamples = readData('data.xlsx','2', 'Temp')
    # print('lstSamples = ', lstSamples)
    # example2(lstSamples)

    # test 3
    # lstSamples = readData('data.xlsx','4', 'Temp')
    # print('lstSamples = ', lstSamples)
    # example2(lstSamples)

    # read csv
    i = 0
    lstSamples = []
    with open('data_1.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            i = i + 1
            column1, column2 = row
            print(f"{i} Column 1: {column1}, Column 2: {column2}")
            lstSamples.append(float(column2))

    print('lstSamples = ', lstSamples)

    # run kalman filter
    example2(lstSamples)