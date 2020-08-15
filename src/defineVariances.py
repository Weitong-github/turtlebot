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
from scipy.stats.distributions import chi2
from scipy.fftpack import diff
import math

import rosbag
from std_msgs.msg import Float64, Int32
from geometry_msgs.msg import Pose2D
import RobotAndSensorDefinition as RD


#Uncertainty of initial positions of the robot

sigmaX = 5
sigmaY = 5
sigmaTheta = 2*np.pi / 180

Pinit = np.diag([sigmaX**2, sigmaY**2, sigmaTheta**2])


#Measurement noise

sigmaMeasurement = np.sqrt((RD.lineWidth**2)/2)

Qgamma = sigmaMeasurement**2


#Input noise

sigmaTuning = 0.06
Qwheels = sigmaTuning**2 * np.identity(2)
Qbeta = np.matmul(np.matmul(RD.jointToCartesian,Qwheels),np.transpose(RD.jointToCartesian))

#state noise 

Qalpha = np.zeros(3)

#mahalanobis distance threshold

mahaThreshold = np.sqrt(chi2.ppf(0.95,df = 1))