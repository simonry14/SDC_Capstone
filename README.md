## Udacity-CarND-Capstone
### team7: Wenhan Yuan; Michale Bailey; Qianqiao Zhang; Simon Peter Miyingo

Our project are divided into 4 parts.

| Name |account email | task |
| ------ | ------ | ------ |
|  Wenhan Yuan | wenhan_yuan@hotmail.com| Detection |
| Michale Bailey | michale.bailey@gmail.com | Waypoint updater & Detection |
| Qiaoqian Zhang | zhangqianqiao@outlook.com | Classification |
| Simon Peter Miyingo | simonpetermiyingo@gmail.com | Control |

The details are below.

#### The planning subsystem
For the project of interest are the Perception, Planning and Control subsystems. In this implementation the focus was on the traffic light detection node in the Perception subsystem, the waypoint updater node in the Planning subsystem and the Drive by wire (DBW) node of the Control subsystem.

*1] In the Perception subsystem only traffic lights were observed. The traffic light detection node consists of a traffic light detector module (tl_detector) and traffic light classifier module (tl_classifer).

subscribes to : `/base_waypoints`, `/current_pose`, `/image_color`
publishes to : `/traffic_waypoint` 

Mike implemented the 'process_traffic_light' method of the tl_detector and it uses the car's current position to determine which base_waypoints are closest to the car and subsequently determines, from a list of traffic light coordinates, which traffic light is closest and ahead of the car. Once this is ascertained the tl_classifier executes the clasification the images from the `/image_color` messages to obtain the color of the traffic light and if it is RED publishes its waypoint index. The tl_classifier was implemented using SSD and was done by Wenhan (Team Lead). 

*2] The Planning subsystem provides the trajectory for the car to follow it comprises the waypoint loader and the waypoint updater nodes.

The waypoint loader node has no subscribers and only publishes `/base_waypoints`

These base_waypoints are published once and are all the waypoints for the given track. They comprise of (x, y, z) coordinates of points on the track and the velocity and heading (yaw) the car should maintain at those points or locations.

The waypoint updater node

subscribes to: `/base_waypoints`, `/current_pose` and `/traffic_waypoints`
publishes to: `/final_waypoints`

Mike implemented this node and it updates the velocity component of the base waypoints based on traffic light conditions. If the closest traffic light ahead of the car is showing RED then the velocities of all base waypoints between the car's current position up to the LOOKAHEAD_WPS limit will be altered to facilitate the deceleration of the car to 0 m/s at the stopline. Any other traffic light condition and the car will travel at the reference velocities of these base waypoints.

There are two main functions in the node, 'generate_lane' and 'decelerate_waypoints'. Once the closest waypoint index is ascertained, by finding the index of the waypoint that is just ahead of the car's position, in the 'generate_lane' function it is used to select a subset of waypoints that will form the lane message that is then published to /final_waypoints message.

If the subscriber `/traffic_waypoints` message has a base waypoint index of [-1] or an index greater than the LOOKAHEAD limit that indicates that the traffic light is not RED or not in the current trajectory given by the subset of base waypoints then the lane message is published with unaltered velocities and the car continues at its current velocity. However, if the subscriber has an index value that falls within the subset of base waypoints the 'decelerate_waypoints' function is called and it creates a new waypoint message and adds new velocities for all the base waypoints within the range. It does this by calculating velocities that are proportiional to the reducing distances as the car moves towards the stopline which are then compared to the reference velocities and the lower selected as the car decelerates. These base waypoints with their velocities adjusted are then published to final_waypoints message.

#### Traffic light Detection and Classification

Traffic light Detection and Classification module is designed to extract the traffic light and determine its color from the camera image. 

Our team divided the problem into two parts.

*1：Separate the traffic lights from the image by the pre-trained model, which we use SSD here.

*2: Classifier the color of the image by the trained model, we train on a Lenet to achieve this function.

Step1: Loading pre-training model

First, we need a model to locate and classifier; We look up information from
the Internet and decided to use the model
ssd_inception_v2_coco_11_06_2017;
The model parameters have been placed under the folder of \CarNDCapstone\ros\src\tl_detector.

Step2: Training Lenet Model for Color Recognition

1. Datasets
The Datasets using for training comes from Udacity's ROSbag file from
Carla and Traffic lights from Udacity's simulator.
2. Build model
The images used for training is simple, so we built LeNet model for color
recognition.
3. Save Model
The model is saved as 'model.h5' and placed in the folder \CarNDCapstone\ros\src\tl_detector.
Camera Image and Model's Detections:

#### Control Subsystem

This system publishes control commands for the vehicle’s steering, throttle, and brakes based on a list of waypoints to follow. It consists of the following nodes:
1) Waypoint Follower Node
This is an Autoware package that contains code that publishes proposed linear and angular velocities to the /twist_cmd topic. No changes were made to this package

2) Drive By Wire (DBW) Node
The Udacity test vehicle (Carla) is equipped with a drive-by-wire (dbw) system, meaning the throttle, brake, and steering have electronic control. The dbw node is responsible for publishing the brake, throttle and steering commands by utilizing the vehicle’s target linear and angular velocity. Three controllers are used for each of the different commands that need to be generated.
##### Steering Controller
The steering controller utilizes the vehicle’s steering ratio and wheelbase length to translate the proposed linear and angular velocities into a steering angle. This steering angle is then passed through a low pass filter to reduce the effect of noise.
##### Throttle Controller
The throttle controller is a PID controller that compares the current vehicle velocity with the target velocity and adjusts the throttle accordingly. The throttle gains were tuned manually.
##### Braking Controller
The braking controller works by braking the vehicle based on the difference between the target velocity and the current vehicle velocity. This proportional gain was tuned manually too.


This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
