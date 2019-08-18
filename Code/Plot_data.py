import navpy as nv
import matplotlib.pyplot as plt
import Classes
import csv


def plot_data(in_data: Classes.InData, out_data: Classes.OutData):
    pos_ned = in_data.gnss
    x_h = out_data.x_h
    [lat_ref, lon_ref] = [[el[0] for el in in_data.gnssref], [el[1] for el in in_data.gnssref]]
    [lat_ref, lon_ref] = [sum(lat_ref)/len(lat_ref), sum(lon_ref)/len(lon_ref)]
    x_ned = nv.ned2lla([[el[0][0] for el in x_h],
                        [el[1][0] for el in x_h],
                        [el[2][0] for el in x_h]],
                       lat_ref, lon_ref, 0)  # convert ned to lat lon for output to html
    x_ned_list = []
    for i in range(len(x_ned[0])):
        x_ned_list.append([x_ned[2][i], x_ned[0][i], x_ned[1][i]])  # append the lat long for the output format
    with open('../Output/output.csv', 'wt', newline='') as file:  # write the data as lat long to output file
        filewc = csv.writer(file, dialect='excel')
        filewc.writerows(x_ned_list)
    plt.figure(1)
    aided = plt.plot(
        [el[1] for el in x_h],
        [el[0] for el in x_h],
        'r-.',
        label='GNSS aided INS trajectory')  # plot the GPS aided INS trajectory

    x_ned_1 = nv.ned2lla([[el[0][0] for el in x_h],
                          [el[1][0] for el in x_h],
                          [el[2][0] for el in x_h]],
                         lat_ref, lon_ref, 0)
    x_ned_list_limited = []
    for i in range(len(x_ned_1[0])):
        x_ned_list_limited.append([x_ned_1[2][i], x_ned_1[0][i], x_ned_1[1][i]])
    # --------for file with GPS signal------------------------------------
    # for files with no GPS signals comment this section
    GPS = plt.plot([el[1] for el in pos_ned],
                   [el[0] for el in pos_ned],
                   'b--',
                   label='GNSS position estimate')  # plot the GPS signal
    start = plt.plot(pos_ned[0][1],
                     pos_ned[0][0],
                     'ks',
                     linewidth=4.0,
                     label='Start point')  # plot the start point
    #  ------------------------------------------
    plt.title('Trajectory')
    plt.ylabel('North [m]')
    plt.xlabel('East [m]')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()