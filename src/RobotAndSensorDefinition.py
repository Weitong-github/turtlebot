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
import math
import rosbag
from std_msgs.msg import Float64, Int32
from geometry_msgs.msg import Pose2D


#Define wheel radius and error
rwheel = 21.5
rwheel_error = 1.02

#Define track gauge size and error
trackGauge = 112
trackGauge_error = 1.02

#Encoder resolution
encoderRes = 180

#Smapling frequency
samplingFrequency = 10
# had to add '1.0' to specify it's a float division, if not, will return 0
samplingPeriod = 1.0/samplingFrequency

#Top robot speed
topRobotSpeed = 100

#Width of the lines between the tiles
lineWidth = 10

#Coordinates of the sensors on the robot frame
mSensors = np.array([[0,0],
			[50,-50],
			[1,1]])
#Number of existing sensors
nbLineDetectors = np.shape(mSensors)[1]

#X and Y dimensions of the tiles
xSpacing = 100
ySpacing = 100

#Transforming the encoder resolution
dots2rad = (2.0*np.pi)/encoderRes 
rad2dots = 1.0/dots2rad

#joint to cartesian matrix without including the erros
jointToCartesian = np.array([[rwheel/2,rwheel/2],[rwheel/trackGauge,-rwheel/trackGauge]])
cartesianToJoint = np.linalg.inv(jointToCartesian)

#Matrix with the errors (to perform computations)
jointToCartesian_error = np.array([[rwheel*rwheel_error/2,rwheel/2],[rwheel*rwheel_error/trackGauge*trackGauge_error,-rwheel/trackGauge*trackGauge_error]])
