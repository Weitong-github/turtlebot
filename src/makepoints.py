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
from scipy.interpolate import CubicSpline 

import rosbag
from std_msgs.msg import Float64, Int32
from geometry_msgs.msg import Pose2D
import RobotAndSensorDefinition as RD


#node name
rospy.init_node('makepoints')

rate = rospy.Rate(10)

#world plotting, for refence on the waypoints selection
plt.title('Waypoints Selection')
plt.figure(1)
for i in range(0,10):
	plt.plot([i*100, i*100], [0,1000], 'k')
	plt.plot([0,1000],[i*100, i*100], 'k')

#waypoints input
xy = plt.ginput(-1)
plt.close()
plt.show()


#storage of the x and y coords of the points
x = []
y = []

for n in range(0, np.shape(xy)[0]):
	x.append(xy[n][0])
	y.append(xy[n][1])


#time and number of sample definition for interpolation 
t = np.arange(0,np.shape(x)[0])
tq = np.arange(0,np.shape(x)[0]-1, 0.001)

ts = (max(x) - min(x)) / ((np.shape(x)[0]-1)*1000)

# calculate spline to connect way points
cs_x = CubicSpline(t,x)
cs_y = CubicSpline(t,y)
xq = cs_x(tq)
yq = cs_y(tq)


# check dimmentions
#print(np.shape(xq))
#print(np.shape(yq))
#print(np.shape(tq))

plt.figure(2)
for i in range(0,10):
	plt.plot([i*100, i*100], [0,1000], 'k')
	plt.plot([0,1000],[i*100, i*100], 'k')
plt.title('Generated Trajectory')
# Plot the waypoints
plt.plot(x,y, 'o', label= "data")
# Plot the interpolated curve.
plt.plot(xq, yq, 'g-',linewidth = 4, label= "spline")


#Velocity computation
v = []
a = [xq[1], yq[1]]
b = [xq[0], yq[0]]
c = np.subtract(a,b)
v.append(np.linalg.norm(c) / (tq[1]-tq[0]))
topSpeed = 0
for i in range(1,np.shape(tq)[0]-1):
	a = [xq[i+1], yq[i+1]]
	b = [xq[i], yq[i]]
	c = np.subtract(a,b)
	v.append(np.linalg.norm(c) / (tq[i+1]-tq[i]))
	topSpeed = max(topSpeed, v[i])
v.append(v[np.shape(tq)[0]-2])


# Scale time and velocity so the top speed has a given value.
speedFactor = RD.topRobotSpeed/topSpeed
tq = tq/speedFactor

for i in range(0,len(v)):
	v[i] = v[i]*speedFactor

# Calculate robot orientation at each point. Done numerically, but 
# can be done using the spline coefficients. 
thetaq = []
thetaq.append(np.arctan2(yq[1]-yq[0], xq[1]-xq[0]))
for i in range(1,np.shape(tq)[0]-1):
	thetaq.append(np.arctan2(yq[i+1]-yq[i], xq[i+1]-xq[i]))

	while thetaq[i]-thetaq[i-1] > np.pi:
		thetaq[i] = thetaq[i]-2*np.pi
	while thetaq[i]-thetaq[i-1] < -np.pi:
		thetaq[i] = thetaq[i]+2*np.pi
thetaq.append(thetaq[np.shape(tq)[0]-2])


# Calculate elementary travelled distance and elementary rotation of the
# robot between time instants.
deltaD = [0]
deltaTheta = [0]
for i in range(1, np.shape(tq)[0]):
	a = [xq[i], yq[i]]
	b = [xq[i-1], yq[i-1]]
	c = np.subtract(a,b)
	deltaD.append(np.linalg.norm(c))
	deltaTheta.append(thetaq[i] - thetaq[i-1])


# Calculate wheel rotation angle at each time instant (zero at t=0)
qRight = np.zeros(np.shape(tq)[0])
qLeft = np.zeros(np.shape(tq)[0])

for i in range(1, np.shape(tq)[0]):
	deltaq = [deltaD[i], deltaTheta[i]]
	delta = np.matmul(RD.cartesianToJoint,deltaq)

	qRight[i] = (qRight[i-1] + delta[0])
	qLeft[i] = (qLeft[i-1] + delta[1])


# Check results by recalculating path using odometry equations.
xOdo = np.zeros(np.shape(tq)[0])
yOdo = np.zeros(np.shape(tq)[0])
thetaOdo = np.zeros(np.shape(tq)[0])

xOdo[0] = xq[0]
yOdo[0] = yq[0]
thetaOdo[0] = thetaq[0]


for i in range(1, np.shape(tq)[0]):
	dq = [qRight[i]-qRight[i-1], qLeft[i]-qLeft[i-1]]
	dCart = np.matmul(RD.jointToCartesian_error,dq)
	xOdo[i] = xOdo[i-1] + dCart[0]*np.cos(thetaOdo[i-1])
	yOdo[i] = yOdo[i-1] + dCart[0]*np.sin(thetaOdo[i-1])
	thetaOdo[i] = thetaOdo[i-1] + dCart[1]

#Adding the odometry to the previous plot
plt.plot(xOdo,yOdo, 'r-', label = "odo")
plt.legend()

#Plot to see te velocity over the generated trajectory
plt.figure(3)
plt.title("Velocity")
plt.xlabel("time (seg)")
plt.plot(tq, v, 'r-')
plt.grid()
plt.show()


#Storage of data on a bag file
bag = rosbag.Bag('src/turtlebot_sim/bag_files/trajectory.bag', 'w')

try:
    rwheel_info = Float64()
    rwheel_info.data = RD.rwheel
    bag.write('rwheel', rwheel_info)


    trackGauge_info = Float64()
    trackGauge_info.data = RD.trackGauge
    bag.write('trackGauge', trackGauge_info)

    
    for i in range(0,np.shape(tq)[0]):
    	tq_info = Float64()
    	tq_info.data = tq[i]
    	bag.write('tq', tq_info)

    	Poseq_info = Pose2D()
    	Poseq_info.x = xq[i]
    	Poseq_info.y = yq[i]
    	Poseq_info.theta = thetaq[i]
    	bag.write('Poseq', Poseq_info)

    	PoseO_info = Pose2D()
    	PoseO_info.x = xOdo[i]
    	PoseO_info.y = yOdo[i]
    	PoseO_info.theta = thetaOdo[i]
    	bag.write('PoseOdo', PoseO_info)



    	qRight_info = Float64()
    	qRight_info.data = qRight[i]
    	bag.write('qRight', qRight_info)

    	qLeft_info = Float64()
    	qLeft_info.data = qLeft[i]
    	bag.write('qLeft', qLeft_info)


finally:
    bag.close()
