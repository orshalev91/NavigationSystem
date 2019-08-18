# NavigationSystem
1.	Add data file with IMU and GPS to “Input” folder
Notes: data files should be CSV files and the data should be arranged in columns in this order
t[s];x[m];y[m];ax[m/s^2];ay[m/s^2];az[m/s^2];roll[rad];pitch[rad];yaw[rad];gy_x[rad/sec];gy_y[rad/sec];gy_z[rad/sec];_;_;_;lattitude;longtitude
2.	Edit “Main” file in “Code” folders on line 22 to select which data file to run
3.	Option: Comment/Uncomment lines 19-21 in file “Plot_data” in “Code” folder to enable/disable output file with the systems position coordinates.
4.	Note: If running a data file with no GPS signals comment the section in lines 36-57 in file “Plot_data” in “Code” folder
5.	Run file “Main”
6.	If chosen to enable an output file in step 3. After the run an output file will be in “Output” folder with the systems positions coordinates in the order:
Altitude, Latitude, Longitude.
