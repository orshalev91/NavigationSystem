import scipy.io as sio
import Classes
import GPSaidedINS
import Plot_data
import numpy as np

## Load data
print('Loads data')
## initiate
out_data = Classes.OutData
kalman_filter = Classes.Filter
in_data = Classes.InData

## Load filter settings
print('Loads settings')
settings = Classes.Settings()

## Run the GNSS-aided INS
print('Runs the GNSS-aided INS')

# loop to read the measurements, wait for 100 reads to go to gnssaided setup
with open('../Input/drive3_rt-GPS.csv', 'r') as f:  ## select drive data file
    in_data.name = f.name
    next(f)
    for line in f:
        line = line.split(",")[:-1]
        # print line
        if len(line) > 1:
            GPSaidedINS.sensors_data_read(line, in_data, settings, out_data, kalman_filter) # send the line to decoding


## Plot the data
print('Plot data')
Plot_data.plot_data(in_data, out_data)

