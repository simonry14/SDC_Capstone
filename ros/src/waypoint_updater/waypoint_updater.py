#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.
Once you have created dbw_node, you will update this node to use the status of traffic lights too.
Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 30 # Number of waypoints we will publish. 
MAX_DECEL = 3.086419753 # The deceleration to stop the car travelling at 11.11 ms(max vel) within a distance of 20m.


class WaypointUpdater(object):
	def __init__(self):
		rospy.init_node('waypoint_updater')

		self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

		# TODO: Add other member variables you need below
		
		self.pose = None
		self.base_wps = None
		self.waypoints_2d = None
		self.waypoint_tree = None
		self.stopline_wp_idx = None
				
		# TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
		rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
		rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
		rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

		self.loop() 

	def loop(self):
		rate = rospy.Rate(25)
		while not rospy.is_shutdown():
			if self.pose and self.base_wps:
				self.publish_waypoints()
			rate.sleep()    

	
	def get_closest_waypoint_idx(self):
		x = self.pose.pose.position.x
		y = self.pose.pose.position.y
		closest_idx = self.waypoint_tree.query([x,y], 1)[1]

		#check if closest point is ahead (of car) or not
		closest_coord = self.waypoints_2d[closest_idx]
		prev_coord = self.waypoints_2d[closest_idx - 1]

		#Equation for hyperplane through closest_coords
		cl_vect = np.array(closest_coord)
		prev_vect = np.array(prev_coord)
		pos_vect = np.array([x,y])

		val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

		if val > 0:
			closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

		return closest_idx
	
	
	def publish_waypoints(self):
		final_lane = self.generate_lane()
		self.final_waypoints_pub.publish(final_lane)
		
	def generate_lane(self):
		lane = Lane()	

		# Localization of the car on the road given its pose
		closest_idx = self.get_closest_waypoint_idx()
		farthest_idx = closest_idx + LOOKAHEAD_WPS
		
		# Select a subset of total waypoints from car to lookahead index as the path the car will follow
		base_waypoints = self.base_wps.waypoints[closest_idx:farthest_idx]
		
		# If there are no RED traffic lights detected within this path these waypoints
		# are published with their velocities unaltered
		if (self.stopline_wp_idx == -1) or (self.stopline_wp_idx == None) or (self.stopline_wp_idx >= farthest_idx):
			lane.waypoints = base_waypoints
		else:
			# If a RED traffic light is observed the velocities of the waypoints are altered 
			# to allow the car to come to a smooth stop at the traffic light's stopline 
			lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

		return lane	
	
	def decelerate_waypoints(self, waypoints, closest_idx):
		temp = []
		for i, wp in enumerate(waypoints):

			# Create new waypoint message whose velocity component will be adjusted and 
			# compared to the reference waypoints
			p = Waypoint()
			p.pose = wp.pose

			stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0) # Two waypoints back from line so front of car stops at line
			
			# Ensures car comes to a full stop at stopline. All remaining waypoints within the given path
			# with indices > stop_idx will have their velocities set to 0.
			if i >= stop_idx: 
				vel = 0.
			
			else:
				dist = self.distance(waypoints, i, stop_idx)
				# Calculates velocities that are proportiional to the reducing distances as the car moves towards the stopline
				vel = math.sqrt(2 * MAX_DECEL * dist)
				if vel < 1.:
					vel = 0.

			p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
			temp.append(p)

		return temp

	def pose_cb(self, msg):
		self.pose = msg

	def waypoints_cb(self, waypoints):
		self.base_wps = waypoints
		if not self.waypoints_2d:
			self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
			self.waypoint_tree = KDTree(self.waypoints_2d)
		
	def traffic_cb(self, msg):
		# Callback for /traffic_waypoint message. 
		self.stopline_wp_idx = msg.data

	def obstacle_cb(self, msg):
		# TODO: Callback for /obstacle_waypoint message. We will implement it later
		pass

	def get_waypoint_velocity(self, waypoint):
		return waypoint.twist.twist.linear.x

	def set_waypoint_velocity(self, waypoints, waypoint, velocity):
		waypoints[waypoint].twist.twist.linear.x = velocity

	def distance(self, waypoints, wp1, wp2):
		dist = 0
		# The z component for the waypoints are superfluous and adds to the computational overhead
		# when only 2-D distances are of interest
		dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2) #  + (a.z-b.z)**2)
		for i in range(wp1, wp2+1):
			dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
			wp1 = i
		return dist


if __name__ == '__main__':
	try:
		WaypointUpdater()
	except rospy.ROSInterruptException:
		rospy.logerr('Could not start waypoint updater node.')