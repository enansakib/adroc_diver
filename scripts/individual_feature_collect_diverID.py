#!/usr/bin/env python3
"""
enansakib
"""
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray

FRAMES_THR = 50
DIVER_ID = 1 ## Change this integer value sequentially before running 

class IndividualFeatureCollect:
    def __init__(self):
        self.jds_list = []
        self.diver_features = []
        self.diver_labels = []
        rospy.Subscriber('/detection/jds', Float64MultiArray, self.jdCallBack, queue_size=3)
        self.collected_frames = 0
        self.feature_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/diver'+str(DIVER_ID)+'_feature.npy'
        self.label_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/diver'+str(DIVER_ID)+'_label.npy'


    def jdCallBack(self, jds_topic):
        # print(jds_topic.data)
        self.jds_list = jds_topic.data
        features = self.featureExtractor(self.jds_list)

        if self.collected_frames < FRAMES_THR:
            self.DateUpdate(features)
        else:
            self.diver_features = np.concatenate(self.diver_features)
            self.diver_labels = np.concatenate(self.diver_labels)
            np.save(self.feature_path, self.diver_features)
            np.save(self.label_path, self.diver_labels)

            print("Feature shape:", self.diver_features.shape, ",  Label shape:", self.diver_labels)
            print()
            print(str(self.collected_frames)+" features/labels are saved for Diver "+str(DIVER_ID)+".")
            
            rospy.signal_shutdown('collected data')

    def featureExtractor(self, jds_data):
        features = np.zeros((1, 45))
        # jds_data.pop(7)        
        f_counter = 1
        for i in range(len(jds_data)):
            for j in range(i+1,len(jds_data)):                
                data = jds_data[i]/jds_data[j]
                features[0,f_counter-1] = data
                f_counter += 1
        
        return features

    def DateUpdate(self, features):
        self.diver_features.append(features)
        self.diver_labels.append([DIVER_ID])
        
        self.collected_frames += 1
        print(self.collected_frames)

if __name__ == '__main__':
    rospy.init_node('feature_collect_diverID', anonymous=False)
    IndividualFeatureCollect()
    rospy.spin()

else:
    pass

