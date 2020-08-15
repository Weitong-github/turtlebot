# turtlebot
Turtlebot localization using line detection and kalman filter.

July 2020

----------------------------------
Makepoints.py
----------------------------------
	Generates a trajectory based on given waypoints and shows the model velocity for given trajectory.

	To select points:
		- left click -> new point
		- right click -> erase last point
		- middle button/enter -> generate trajectory

	Creates the "trajectory" bag file inside bag_files folder

----------------------------------
simluation.py
----------------------------------

	Performs the model simulation on generated trajectory in "trajectory" bag file, giving the line detections for such trajectory

----------------------------------
kalmanFilter.py
----------------------------------

	Performs kalman filter algorithm

	-plots the trajectories (desired, odometry and estimated by kalman filter)
	-plots the mahalanobis distances plot
	-plots standard deviations
	-plots errors

----------------------------------
RobotAndSensorDefinition.py
----------------------------------

	Defines the robot physical structure and properties

	Should be modified if needed to adapt the algorithm to another robot with different dimensions

----------------------------------
defineVariances.py
----------------------------------

	Setup the input, measurement and state noise

	Defines the sigmaTuning used (for kalman filter)

	Defines the threshold for the mahalanobis distance
