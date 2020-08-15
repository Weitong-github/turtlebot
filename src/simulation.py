#! /usr/bin/env python


###
'''
Turtlebot localization using line detection and kalman filter.
CORO M1
Group Project
July 2020

Associated prof: Gaetan Garcia

Andres Gutierrez
Weitong Ma

'''
###

import rospy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.fftpack import diff
import math
import rosbag
from std_msgs.msg import Float64, Int32
from geometry_msgs.msg import Pose2D
import RobotAndSensorDefinition as RD

#node name
rospy.init_node('simulation')

#Arrays creation
tq = []
xq = []
yq = []
thetaq = []
xOdo = []
yOdo = []
thetaOdo = []
qRight = []
qLeft = []        

#open previous bagfile
bag = rosbag.Bag('src/turtlebot_sim/bag_files/trajectory.bag')

for topic, msg, t in bag.read_messages(topics=['rwheel', 'trackGauge', 'tq', 'Poseq', 'PoseOdo', 'qRight', 'qLeft']):
	if topic == 'rwheel':
		rwheel = msg.data

	if topic == 'trackGauge':
		trackGauge = msg.data

	if topic == 'tq':
		tq.append(msg.data)

	if topic == 'Poseq':
		xq.append(msg.x)
		yq.append(msg.y)
		thetaq.append(msg.theta)

	if topic == 'PoseOdo':
		xOdo.append(msg.x)
		yOdo.append(msg.y)
		thetaOdo.append(msg.theta)

	if topic == 'qRight':
		qRight.append(msg.data)

	if topic == 'qLeft':
		qLeft.append(msg.data)
	
bag.close()

#Time rescaling
totalTime = tq[len(tq)-1] 	
nbSamples = math.floor(totalTime*RD.samplingFrequency)
treal = np.arange(0,nbSamples)*RD.samplingPeriod

#bring the data to the new time given by the sampling freq
qR_inter = interp1d(tq, qRight)
qR = qR_inter(treal)

qL_inter = interp1d(tq, qLeft)
qL = qL_inter(treal)

xr_inter = interp1d(tq, xq)
xreal = xr_inter(treal)

yr_inter = interp1d(tq, yq)
yreal = yr_inter(treal)

thetar_inter = interp1d(tq, thetaq)
thetareal = thetar_inter(treal)

xOdo_inter = interp1d(tq, xOdo)
xOdo = xOdo_inter(treal)

yOdo_inter = interp1d(tq, yOdo)
yOdo = yOdo_inter(treal)

thetaOdo_inter = interp1d(tq, thetaOdo)
thetaOdo = thetaOdo_inter(treal)

# Apply quantization noise on encoder values based on resolution
qR = qR*RD.rad2dots
qL = qL*RD.rad2dots
for i in range(0,len(qR)):
	qR[i] = round(qR[i])
	qL[i] = round(qL[i])
qR = qR*RD.dots2rad
qL = qL*RD.dots2rad


#Sensor state and coordinate arrays
sensorState = np.zeros((len(treal), RD.nbLineDetectors))
xs_array = np.zeros((len(treal), RD.nbLineDetectors))
ys_array = np.zeros((len(treal), RD.nbLineDetectors))


for i in range(0,int(nbSamples)):
	for j in range(0,RD.nbLineDetectors):
		
		oTm = np.array([[np.cos(thetareal[i]), -np.sin(thetareal[i]), xreal[i]],
						[np.sin(thetareal[i]), np.cos(thetareal[i]), yreal[i]],
						[0, 0, 1]])
		#Compute sensor coords on world frame
		oSensor = np.matmul(oTm,RD.mSensors[:,j])
		xs = oSensor[0]
		xs_array[i][j] = xs
		ys = oSensor[1]
		ys_array[i][j] = ys

		#Check if the sensor is over the lines
		if(xs%RD.xSpacing < RD.lineWidth/2) or (xs%RD.xSpacing > RD.xSpacing - RD.lineWidth/2):
			sensorState[i][j] = 1
		elif(ys%RD.ySpacing < RD.lineWidth/2) or (ys%RD.ySpacing > RD.ySpacing - RD.lineWidth/2):
			sensorState[i][j] = 1
		else:
			sensorState[i][j] = 0


#Plotting data arrays
x1_plot = []
y1_plot = []

x2_plot = []
y2_plot = []

#Search for plotting coordwhen sensors are 1
for i in range(0,np.shape(sensorState)[0]):

	if(sensorState[i][0] == 1):
		x1_plot.append(xs_array[i][0])
		y1_plot.append(ys_array[i][0])
	if(sensorState[i][1] == 1):
		x2_plot.append(xs_array[i][1])
		y2_plot.append(ys_array[i][1])


plt.figure(1)
plt.title("Robot Simulation")
for i in range(0,10):
	plt.plot([i*100, i*100], [0,1000], 'k')
	plt.plot([0,1000],[i*100, i*100], 'k')
plt.plot(xq,yq, '-g', label='Trajectory')
plt.plot(xOdo, yOdo, '-r', label ='Odometry')
plt.plot(x1_plot, y1_plot, 'xm', label='Measurements')
plt.plot(x2_plot, y2_plot, 'xm')
plt.legend(loc=1)
plt.show()


#Store the data
bag = rosbag.Bag('src/turtlebot_sim/bag_files/simulation.bag', 'w')

try:
	dots2rad_info = Float64()
	dots2rad_info.data = RD.dots2rad
	bag.write('dots2rad', dots2rad_info)

	rad2dots_info = Float64()
	rad2dots_info.data = RD.rad2dots
	bag.write('rad2dots', rad2dots_info)

	rwheel_info = Float64()
	rwheel_info.data = RD.rwheel
	bag.write('rwheel', rwheel_info)


	trackGauge_info = Float64()
	trackGauge_info.data = RD.trackGauge
	bag.write('trackGauge', trackGauge_info)

	topRobotSpeed_info = Float64()
	topRobotSpeed_info.data = RD.topRobotSpeed
	bag.write('topRobotSpeed', topRobotSpeed_info)

	xSpacing_info = Float64()
	xSpacing_info.data = RD.xSpacing
	bag.write('xSpacing', xSpacing_info)

	ySpacing_info = Float64()
	ySpacing_info.data = RD.ySpacing
	bag.write('ySpacing', ySpacing_info)

	for i in range(0,np.shape(treal)[0]):
		treal_info = Float64()
		treal_info.data = treal[i]
		bag.write('treal', treal_info)

		PoseReal_info = Pose2D()
		PoseReal_info.x = xreal[i]
		PoseReal_info.y = yreal[i]
		PoseReal_info.theta = thetareal[i]
		bag.write('PoseReal', PoseReal_info)

		PoseOdo_info = Pose2D()
		PoseOdo_info.x = xOdo[i]
		PoseOdo_info.y = yOdo[i]
		PoseOdo_info.theta = thetaOdo[i]
		bag.write('PoseOdo', PoseOdo_info)

		qR_info = Float64()
		qR_info.data = qR[i]
		bag.write('qR', qR_info)

		qL_info = Float64()
		qL_info.data = qL[i]
		bag.write('qL', qL_info)

		sensor1State_info = Float64()
		sensor1State_info.data = sensorState[i][0]
		bag.write('sensor1State', sensor1State_info)

		sensor2State_info = Float64()
		sensor2State_info.data = sensorState[i][1]
		bag.write('sensor2State', sensor2State_info)


finally:
    bag.close()
