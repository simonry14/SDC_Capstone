from styx_msgs.msg import TrafficLight
from keras.models import load_model
import numpy as np
import cv2

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.cls_model = load_model('./light_classification/tl_model.h5')
        

    def get_classification(self, image_seg):
        """Determines the color of the traffic light in the image

        Args:
            image_seg (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        cls_model = load_model('./light_classification/tl_model.h5')
        signal_classes = ['Red', 'Green', 'Yellow']
        img_resize = image_seg
        # Color map conversion
        img_resize=cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB) 
        # Convert to four-dimension input as required by Keras
        img_resize = np.expand_dims(img_resize, axis=0).astype('float32')
        # Normalization
        img_resize/=255.
        # Prediction
        tl_predict = cls_model.predict(img_resize)
        tl_predict = np.squeeze(tl_predict, axis =0)
        # Get color classification
        tl_color = signal_classes[np.argmax(tl_predict)]
        #print(tl_color,', Classification confidence:', predict[np.argmax(predict)])

        return tl_color