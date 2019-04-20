#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from scipy.spatial import KDTree
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from keras.models import load_model
#from PIL import ImageDraw
#from PIL import ImageColor
from scipy.stats import norm
import tensorflow
import time
import tf
import cv2
import yaml
import math
import numpy as np


STATE_COUNT_THRESHOLD = 3

class TLDetector(object): 
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        #self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        #Detection&Classification initialization
        graph_file = 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
        model_path = 'true_model.h5'

        #cmap = ImageColor.colormap
        #self.COLOR_LIST = sorted([c for c in cmap.keys()])

        #load the graph
        self.detection_graph = tensorflow.Graph()
        config = tensorflow.ConfigProto()
        config.gpu_options.allow_growth = True
        # load frozen tensorflow detection model and initialize 
        # the tensorflow graph
        with self.detection_graph.as_default():
            od_graph_def = tensorflow.GraphDef()
            with tensorflow.gfile.GFile(graph_file, 'rb') as fid:
               serialized_graph = fid.read()
               od_graph_def.ParseFromString(serialized_graph)
               tensorflow.import_graph_def(od_graph_def, name='')
               
            self.sess = tensorflow.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        #Load classification model
        self.cls_model = load_model(model_path)
        self.cls_model._make_predict_function()
        self.cls_graph = tensorflow.get_default_graph()


        
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()
        
    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp)) 
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp)) 
        self.state_count += 1
        

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        return self.waypoint_tree.query([x, y], 1)[1]

        
    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #return light.state
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        b = self.tl_detection(cv_image)
        if len(b) != 0:
            img_seg = self.tl_segmentation(cv_image, b)
            tl_color = self.tl_classification(img_seg)
            rospy.loginfo("[TL_DETECTOR] simulated_detection: Nearest TL-state is: %s", tl_color)
            if tl_color == 'Red':
                tl_num = TrafficLight.RED
            elif tl_color == 'Green':
                tl_num = TrafficLight.GREEN
            elif tl_color == 'Yellow':
                tl_num = TrafficLight.YELLOW

        else:
            tl_num = -1
            rospy.loginfo("[TL_DETECTOR] simulated_detection: Nearest TL-state is: None")


        #Get classification
        return tl_num

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None
        sld = None
        
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                #Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypoint index

                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx
                    
                    # Approximating the straight line distance (m), in the x direction, between closest stopline and car
                    sld = line[0] - self.pose.pose.position.x 
        
        # Checks the status of the traffic light only when the car is within approx. 50m of it       
        if closest_light and sld <= 50:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state
        
        return -1, TrafficLight.UNKNOWN

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        idxs_all = []
        for i in range(n):
            if scores[i] >= min_score:
                if int(classes[i]) == 10:
                    idxs_all.append(i)
        if idxs_all != []:
            idxs.append(np.argmax(scores[idxs_all]))
        
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].
        
        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        
        return box_coords
      
    def box_normal_to_pixel(self, box, dim):
        
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)  


    def tl_detection(self, image):
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        with self.detection_graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)

              # Actual detection.
            (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores,
                                                      self.detection_classes],
                                                     feed_dict={self.image_tensor: image_expanded})            
 

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            confidence_cutoff = 0.2
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.

            return boxes

    def tl_segmentation(self, detected_image, boxes, img_size=32):

        img_full_np = np.asarray(detected_image, dtype=np.uint8)
        img_full_np_copy = np.copy(img_full_np)
        box = np.squeeze(boxes)
        dim = img_full_np.shape[0:2]
        b = self.box_normal_to_pixel(box, dim)

        cv2.rectangle(img_full_np,(b[1],b[0]),(b[3],b[2]),(0,255,0),2)
        #plt.figure(figsize=(9,6))
        #plt.imshow(img_full_np)
        #plt.show()
        img_np = cv2.resize(img_full_np_copy[b[0]:b[2], b[1]:b[3]], (img_size, img_size)) 
        #plt.figure(figsize=(5,5))
        #plt.imshow(img_np)
        #plt.show()
        return img_np
    """
    def tl_classification(self, image_seg):
        signal_classes = ['Red', 'Green', 'Yellow']

        img_resize = image_seg
        # Color map conversion
        img_resize=cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB) 
        # Convert to four-dimension input as required by Keras
        img_resize = np.expand_dims(img_resize, axis=0).astype('float32')
        # Normalization
        img_resize/=255.
        # Prediction
        with self.cls_graph.as_default():
            tl_predict = self.cls_model.predict(img_resize)
            
            tl_predict = np.squeeze(tl_predict, axis =0)
            # Get color classification
            tl_color = signal_classes[np.argmax(tl_predict)]
            #print(tl_color,', Classification confidence:', predict[np.argmax(predict)])

        return tl_color
        """
    
    def tl_classification(self, image_seg):
        signal_classes = ['Red', 'Yellow', 'Green']

        test = np.array([image_seg])

        # Prediction
        with self.cls_graph.as_default():
            tl_predict = self.cls_model.predict(test)
            
            tl_predict = np.squeeze(tl_predict, axis =0)
            # Get color classification
            tl_color = signal_classes[np.argmax(tl_predict)]
            #print(tl_color,', Classification confidence:', predict[np.argmax(predict)])

        return tl_color


    
if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')