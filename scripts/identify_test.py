#! /usr/bin/python3

# This code is a part of the LoCO AUV project.
# Copyright (C) The Regents of the University of Minnesota

# Maintainer: Junaed Sattar <junaed@umn.edu> and the Interactive Robotics and Vision Laboratory

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import cv2
from numpy import array
from scipy import spatial

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes

current_image = None
bridge_object = CvBridge()
feature_vector = list()

def hull(ymin,xmin,ymax,xmax,image):
    img=image[ymin:ymax,xmin:xmax]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale

    blur = cv2.blur(gray, (3, 3)) # blur the image
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = map(lambda x: cv2.convexHull(x, False).shape[0], contours[1])
    return float(sum(hull)/len(hull))
    
def remove_blue(ymin,xmin,ymax,xmax,image):
    img=image[ymin:ymax,xmin:xmax]
    g = img[:,:,1]
    r = img[:,:,2]
    return [sum(sum(x) for x in g), sum(sum(x) for x in r)]
    
def get_average_color(ymin,xmin,ymax,xmax,image):
    image=image[ymin:ymax,xmin:xmax]
    image=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    image=np.array(image)
    color=[]
    crop=np.array_split(image,4)
    for i in range(4):
        color.append(np.mean(crop[i]))
    return color
    
def get_moment(ymin,xmin,ymax,xmax,image):
    image=image[ymin:ymax,xmin:xmax]
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return cv2.HuMoments(cv2.moments(image)).flatten()
    
def remove_inf(ls):
    return np.where(ls == float('-inf'), 0, ls)
    
def get_amp(ymin,xmin,ymax,xmax,image):
    image=image[ymin:ymax,xmin:xmax]
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum =remove_inf(20*np.log(np.abs(fshift)))
    sums = [item for sublist in magnitude_spectrum for item in sublist]
    sums = map(sum, zip(*sums))
    return map(lambda x: x/float((ymax - ymin)*(xmax - xmin)), sums)
    
def approx_shape(ymin,xmin,ymax,xmax,image):
    pts=[]
    image=image[ymin:ymax,xmin:xmax]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    _,cnts,_=cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pts = map(lambda x: len(cv2.approxPolyDP(x, 0.02 * cv2.arcLength(x, True), True)), cnts)
    if len(pts) != 0:
        return float(sum(pts)/(len(pts)))
    else:
        return 0
    
def extract_features(top, left, bottom, right, image):
    # extract features from the detection
    temp = hull(top, left, bottom, right, image)
    average_colors = get_average_color(top, left, bottom, right, image)
    moments = get_moment(top, left, bottom, right, image)
    shape = approx_shape(top, left, bottom, right, image)
    r_amp, b_amp, g_amp = get_amp(top, left, bottom, right, image)
    features = [temp, r_amp, b_amp, g_amp, shape]
    for moment in moments:
       features.append(moment)
    features += average_colors
    return features


def image_cb(msg):
    global current_image, bridge_object
    current_image = bridge_object.imgmsg_to_cv2(msg, desired_encoding="bgr8")

def bbox_cb(msg):
    global feature_vector
    
    boxes = list()
    max_conf = -1
    max_idx = None

    for idx, b in enumerate(msg.bounding_boxes):
        boxes.append(b)
        if b.probability > max_conf:
            max_conf = b.probability
            max_idx = idx

    if (not max_idx is None) and (not current_image is None):
        selbox = boxes[max_idx]
        bbox_observation = [selbox.xmin, selbox.ymin, selbox.xmax, selbox.ymax]
        features = extract_features(bbox_observation[1], bbox_observation[0], bbox_observation[3], bbox_observation[2], current_image)
        feature_vector.insert(0,features)

        rospy.loginfo("Ayyy new feature vector just dropped:")
        rospy.loginfo(features)

        if len(feature_vector) >= 100:
            feature_vector.pop()
        

if __name__ == '__main__':
    rospy.init_node('identity_test_node', anonymous=False)
    rospy.base_image_sub = rospy.Subscriber('loco_cams/right/image_raw', Image, image_cb)
    rospy.bbox_sub = rospy.Subscriber('darknet_ros/bounding_boxes', BoundingBoxes, bbox_cb)

    r = rospy.Rate(10)

    while not rospy.is_shutdown():
        r.sleep()
            