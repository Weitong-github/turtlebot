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
import defineVariances as DV

#node name
rospy.init_node('Kalman_Filter')


#Arrays declaration
treal = []
xreal = []
yreal = []
thetareal = []
xOdo = []
yOdo = []
thetaOdo = []
qR = []
qL = []
sensor1State = []
sensor2State = []
xplot = []
yplot = []
thetaplot = []
dM_array = []
dMW_array1 = []
dMW_array2 = []
dMW_array3 = []
xs_plot = [[],[]]
ys_plot = [[],[]]
linex = [[],[]]
liney = [[],[]]
sigmaX = []
sigmaY = []
sigmaTheta = []

#Open the bag file
bag = rosbag.Bag('src/turtlebot_sim/bag_files/simulation.bag')

for topic, msg, t in bag.read_messages(topics=['dots2rad','rad2dots','rwheel','trackGauge','topRobotSpeed',
												'xSpacing','ySpacing','treal','PoseReal', 'PoseOdo','qR','qL','sensor1State','sensor2State']):
	if topic == 'dots2rad':
		dots2rad = msg.data

	if topic == 'rad2dots':
		rad2dots = msg.data

	if topic == 'rwheel':
		rwheel = msg.data

	if topic == 'trackGauge':
		trackGauge = msg.data

	if topic == 'topRobotSpeed':
		topRobotSpeed = msg.data

	if topic == 'xSpacing':
		xSpacing = msg.data

	if topic == 'ySpacing':
		ySpacing = msg.data

	if topic == 'treal':
		treal.append(msg.data)

	if topic == 'PoseReal':
		xreal.append(msg.x)
		yreal.append(msg.y)
		thetareal.append(msg.theta)

	if topic == 'PoseOdo':
		xOdo.append(msg.x)
		yOdo.append(msg.y)
		thetaOdo.append(msg.theta)

	if topic == 'qR':
		qR.append(msg.data)

	if topic == 'qL':
		qL.append(msg.data)

	if topic == 'sensor1State':
		sensor1State.append(msg.data)

	if topic == 'sensor2State':
		sensor2State.append(msg.data)
	
bag.close()


xs_array = np.zeros((len(treal), RD.nbLineDetectors))
ys_array = np.zeros((len(treal), RD.nbLineDetectors))

#Function for evolution model
def EvolutionModel(X,U):
	Xnew = np.array([X[0] + U[0]*np.cos(X[2]), X[1] + U[0]*np.sin(X[2]), X[2] + U[1]])
	return Xnew

#Merge both sensors data in one variable
sensorState = np.zeros((np.shape(sensor1State)[0],2))
for i in range(0,np.shape(sensor1State)[0]):
	sensorState[i] = [sensor1State[i],sensor2State[i]]

#Define initial position and initial errors
X = np.array([xOdo[0], yOdo[0], thetaOdo[0]])
P = DV.Pinit

#First values to plot
xplot.append(X[0])
yplot.append(X[1])
thetaplot.append(X[2])

#Store the initial values of sigmas
sigmaX.append(np.sqrt(P[0][0]))
sigmaY.append(np.sqrt(P[1][1]))
sigmaTheta.append(np.sqrt(P[2][2])*180/np.pi)

for i in range(1, np.shape(treal)[0]):

	deltaq = np.array([qR[i] - qR[i-1],qL[i] - qL[i-1]])

	U = np.matmul(RD.jointToCartesian_error,deltaq)
	X = EvolutionModel(X,U)


	#Calculate linear approximation of the system equation
	A = np.array([[1,0,-U[0]*np.sin(X[2])],[0,1,U[0]*np.cos(X[2])],[0,0,1]])
	B = np.array([[np.cos(X[2]),0],[np.sin(X[2]),0],[0,1]])

	#Error Propagation
	A = A.astype(float)
	B = B.astype(float)
	P = P.astype(float)
	P = A.dot(P).dot(np.transpose(A)) + B.dot(DV.Qbeta).dot(np.transpose(B))

	for j in range(0,RD.nbLineDetectors):
		oTm = np.array([[np.cos(X[2]), -np.sin(X[2]), X[0]],
						[np.sin(X[2]), np.cos(X[2]), X[1]],
						[0, 0, 1]])

		#Coord of the sensors on world frame (already in homogenous coords)
		oTm = oTm.astype(float)
		oSensor = oTm.dot(RD.mSensors[:,j])

		xs = oSensor[0]
		xs_array[i][j] = xs
		ys = oSensor[1]
		ys_array[i][j] = ys


		if(sensorState[i][j] == 1):
			#We get the actual line that is being measured
			Y_x = np.round(xs/RD.xSpacing)*RD.xSpacing
			Y_y = np.round(ys/RD.ySpacing)*RD.ySpacing

			#Compute the surrounding lines
			if xs/RD.xSpacing > np.round(xs/RD.xSpacing):
				wl_x = Y_x + 100
			elif xs/RD.xSpacing < np.round(xs/RD.xSpacing):
				wl_x = Y_x - 100

			if ys/RD.ySpacing > np.round(ys/RD.ySpacing):
				wl_y = Y_y + 100
			elif ys/RD.ySpacing < np.round(ys/RD.ySpacing):
				wl_y = Y_y - 100

			#Define Yhat and Y depending on the case of the measured line
			Yhat_x = xs
			Yhat_y = ys

			innov_x = Y_x - Yhat_x
			innov_y = Y_y - Yhat_y

			innov_wx = wl_x - Yhat_x
			innov_wy = wl_y - Yhat_y

			#compute distances and see which one is the closest to compute the respective mahalanobis distace
			#Done for all surrounding line for plotting purposes

			#X line case
			if np.absolute(innov_x) < np.absolute(innov_y):
				#store the closest lines coords
				linex[j].append(Y_x)
				liney[j].append(ys)
				
				#pick the respective innov value
				innov = innov_x

				#compute C
				C = np.array([1,0,-RD.mSensors[0,j]*np.sin(X[2])-RD.mSensors[1,j]*np.cos(X[2])])
				
				#compute mahalanobis distances
				dMaha = np.sqrt(innov * 1.0/((C.dot(P).dot(np.transpose(C))+DV.Qgamma)) * innov)
				dMaha_wrong1 = np.sqrt(innov_y * 1.0/((C.dot(P).dot(np.transpose(C))+DV.Qgamma)) * innov_y)
				dMaha_wrong2 = np.sqrt(innov_wx * 1.0/((C.dot(P).dot(np.transpose(C))+DV.Qgamma)) * innov_wx)
				dMaha_wrong3 = np.sqrt(innov_wy * 1.0/((C.dot(P).dot(np.transpose(C))+DV.Qgamma)) * innov_wy)
				
			#Y line case
			if np.absolute(innov_x) > np.absolute(innov_y):
				#store the closest lines coords
				liney[j].append(Y_y)
				linex[j].append(xs)

				#pick the respective innov value
				innov = innov_y

				#compute C
				C = np.array([0,1,RD.mSensors[0,j]*np.cos(X[2])-RD.mSensors[1,j]*np.sin(X[2])])

				#compute mahalanobis distances
				dMaha = np.sqrt(innov * 1.0/((C.dot(P).dot(np.transpose(C))+DV.Qgamma)) * innov)
				dMaha_wrong1 = np.sqrt(innov_x * 1.0/((C.dot(P).dot(np.transpose(C))+DV.Qgamma)) * innov_x)
				dMaha_wrong2 = np.sqrt(innov_wx * 1.0/((C.dot(P).dot(np.transpose(C))+DV.Qgamma)) * innov_wx)
				dMaha_wrong3 = np.sqrt(innov_wy * 1.0/((C.dot(P).dot(np.transpose(C))+DV.Qgamma)) * innov_wy)
			

			dM_array.append(dMaha)
			dMW_array1.append(dMaha_wrong1)
			dMW_array2.append(dMaha_wrong2)
			dMW_array3.append(dMaha_wrong3)

			xs_plot[j].append(xs)
			ys_plot[j].append(ys)

			#Check mahalanobis distances to perform estimation phase
			if dMaha <= DV.mahaThreshold:
				K = P.dot(np.transpose(C)) * 1.0/((C.dot(P).dot(np.transpose(C))+DV.Qgamma))

				X = X + K*innov
				
				P = (np.identity(len(X)) - K[:,np.newaxis].dot(C[np.newaxis,:])).dot(P)

	xplot.append(X[0])
	yplot.append(X[1])
	thetaplot.append(X[2])
	sigmaX.append(np.sqrt(P[0][0]))
	sigmaY.append(np.sqrt(P[1][1]))
	sigmaTheta.append(np.sqrt(P[2][2])*180/np.pi)

#error computation
errorX = []
errorY = []
errorT = []
for i in range(0, np.shape(treal)[0]):
	errorX.append(xreal[i]-xplot[i])
	errorY.append(yreal[i]-yplot[i])
	errorT.append((thetareal[i]-thetaplot[i])*180/np.pi)

#max error values
#print(np.max(np.absolute(errorX)))
#print(np.max(np.absolute(errorY)))
#print(np.max(np.absolute(errorT)))

sigma3X = []
sigma3Y = []
sigma3T = []
sigma3Xn = []
sigma3Yn = []
sigma3Tn = []

for i in range(0,np.shape(sigmaX)[0]):
	sigma3X.append(sigmaX[i]*3) 
	sigma3Y.append(sigmaY[i]*3) 
	sigma3T.append(sigmaTheta[i]*3) 
	sigma3Xn.append(sigmaX[i]*-3) 
	sigma3Yn.append(sigmaY[i]*-3) 
	sigma3Tn.append(sigmaTheta[i]*-3)

#plots
#Trajectory plot
plt.figure(1)
plt.title("Trajectory")
for i in range(0,10):
	plt.plot([i*100, i*100], [0,1000], 'k')
	plt.plot([0,1000],[i*100, i*100], 'k')
plt.plot(xreal, yreal, 'g-',label="Trajectory")
plt.plot(xOdo, yOdo, 'r-',label="Odometry")
plt.plot(xplot, yplot, 'b-',label="Kalman Filter")
plt.plot(linex[0],liney[0],'g.', label="detected lines")
plt.plot(linex[1],liney[1],'g.')
plt.plot(xs_plot[0],ys_plot[0], 'rx', label="sensor detections")
plt.plot(xs_plot[1],ys_plot[1], 'rx')
plt.legend(loc=1)
plt.grid(b=True,which='both')

#Errors plot
plt.figure(2)
plt.subplot(311)
plt.title("Errors")
plt.ylabel("x error (mm)")
plt.plot(treal, errorX)
plt.plot(treal, sigma3X, 'r-')
plt.plot(treal, sigma3Xn, 'r-')
plt.grid()

plt.subplot(312)
plt.ylabel("y error (mm)")
plt.plot(treal, errorY)
plt.plot(treal, sigma3Y, 'r-')
plt.plot(treal, sigma3Yn, 'r-')
plt.grid()

plt.subplot(313)
plt.ylabel("theta error (deg)")
plt.xlabel("time (seg)")
plt.plot(treal, errorT)
plt.plot(treal, sigma3T, 'r-')
plt.plot(treal, sigma3Tn, 'r-')
plt.grid()


#Mahalanobis distances plot
plt.figure(3)
plt.title("Mahalanobis Distances")
plt.xlabel("measurements over time")
plt.plot([0, np.shape(dM_array)[0]], [DV.mahaThreshold,DV.mahaThreshold], 'k')
plt.plot(dM_array, 'rx')
plt.plot(dMW_array1, 'bx')
plt.plot(dMW_array2, 'gx')
plt.plot(dMW_array3, 'gx')
plt.grid()


#Standard deviations plot
plt.figure(4)
plt.subplot(311)
plt.title("Standard deviations")
plt.ylabel("sigma x (mm)")
plt.plot(treal, sigmaX)
plt.grid()

plt.subplot(312)
plt.ylabel("sigma y (mm)")
plt.plot(treal, sigmaY)
plt.grid()

plt.subplot(313)
plt.ylabel("sigma theta")
plt.xlabel("time (seg)")

plt.plot(treal, sigmaTheta)
plt.grid()
plt.show()

