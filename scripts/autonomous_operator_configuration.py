#!/usr/bin/python3

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

import rospy
import actionlib
import roslaunch

import subprocess, signal, os
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import math

from openpose_ros_msgs.msg import PersonDetection
from std_msgs.msg import Float64MultiArray
import adroc_diver.msg
from loco_pilot.srv import Yaw, YawRequest, YawResponse
from loco_pilot.msg import Command
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

class ADROCState:
    INIT = 0
    SEARCH = 1
    APPROACH = 2
    IDENTIFICATION = 3
    SEARCH_AND_IDENTIFY = 4
    CONCLUDE = 5
    SHUTDOWN = 6

    def id_to_string(id):
        if id == 0:
            return "INIT"
        elif id == 1:
            return "SEARCH"
        elif id == 2:
            return "APPROACH"
        elif id == 3:
            return "IDENTIFICATION"
        elif id == 4:
            return "SEARCH_AND_IDENTIFY"
        elif id == 5:
            return "CONCLUDE"
        elif id == 6:
            return "SHUTDOWN"

class ADROC_Action:
    _feedback = adroc_diver.msg.ApproachDiverFeedback()
    _result = adroc_diver.msg.ApproachDiverResult()

    def __init__(self, name):
        self.rate = 10
        self.state = ADROCState.INIT

        self.x_error_tolerance = rospy.get_param('adroc_diver/x_error_tolerance', 0.01)
        self.y_error_tolerance = rospy.get_param('adroc_diver/y_error_tolerance', 0.01)
        self.pd_error_tolerance = rospy.get_param('adroc_diver/pd_error_tolerance', 0.1)
        self.drp_active_timeout = rospy.get_param('adroc_diver/drp_active_timeout', 1)

        rospy.Subscriber('/drp/drp_target', adroc_diver.msg.DiverRelativePosition, self.drp_cb)
        self.drp_msgs = list()
        self.last_drp_msg = 0

        self.prevID = False
        self.pose_detection = False
        self.jds_list = []
        rospy.Subscriber('/detected_poses_keypoints', PersonDetection, self.pose_msg_cb, queue_size=5)
        X = np.load('/home/irvlab/Downloads/x_features_small.npy')
        y = np.load('/home/irvlab/Downloads/y_labels_small.npy')
        self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        self.clf.fit(X, y)
        self.total_cnt = 0
        self.correct_cnt = 0
        self.num_pred = 0
        self.pred_list = []


        self.activate_drp_controller = rospy.ServiceProxy('drp_reactive_controller/start', Trigger)
        self.deactivate_drp_controller = rospy.ServiceProxy('drp_reactive_controller/stop', Trigger)
        
        self.cmd_pub = rospy.Publisher('/loco/command', Command, queue_size=10)
        self.search_yaw_speed = 0.2
        self.search_it_count = 0

        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, adroc_diver.msg.ApproachDiverAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

        self.bag_launch = None
        self.lf_cmd = "roslaunch /home/irvlab/catkin_ws/src/adroc_diver/launch/data_record.launch"
        self.dg_name = "/data/adroc_diver_digest.txt"
        self.initated_time = None

    def begin_data_recoding(self):
        pass

    def end_data_recording(self):
        # Write to digest file.
        current_time = rospy.get_time()
        duration = current_time - self.initated_time
        final_drp = self.drp_msgs[-1]

        # Write trial ID, DRP (x,y,pd), duration, and final state.
        with open(self.dg_name, 'a+') as f:
            f.write("Trial at %f,%d,%d,%f,%f,%s\n"%(self.initated_time, final_drp.target_x, final_drp.target_y, final_drp.pseudo_distance, duration, ADROCState.id_to_string(self.state)))

    def drp_cb(self, msg):
        if len(self.drp_msgs) == 5:
            self.drp_msgs.pop(0)
        
        self.drp_msgs.append(msg)
        self.last_drp_msg = rospy.Time.now().to_sec()

        return

    # Return true if there's an active DRP value.
    def drp_active(self):
        return ((rospy.Time.now().to_sec() - self.last_drp_msg) < self.drp_active_timeout)
    
    # Return true if there's a stable DRP estimation (we're in the appropriate relative postion)
    def drp_stable(self):
        if len(self.drp_msgs) >0:
            x_errs = list()
            y_errs = list()
            pd_errs = list()

            image_setpoint_x = self.drp_msgs[0].image_w/2.0
            # image_setpoint_y = self.drp_msgs[0].image_h/2.0 #########
            image_setpoint_y = self.drp_msgs[0].image_h/4.0 #########


            for drp_msg in self.drp_msgs:
                x_errs.append((drp_msg.target_x - image_setpoint_x)/ float(drp_msg.image_w)) #Pixel difference between target point and DRP point, normalized by image size.
                y_errs.append((drp_msg.target_y - image_setpoint_y)/ float(drp_msg.image_h))
                pd_errs.append(1.0 - drp_msg.pseudo_distance)

        
            x_err_mean = abs(sum(x_errs)/len(x_errs))
            y_err_mean = abs(sum(y_errs)/len(y_errs))
            pd_err_mean = abs(sum(pd_errs)/len(pd_errs))
            print(x_err_mean, y_err_mean)

            return (x_err_mean < self.x_error_tolerance) and (y_err_mean < self.y_error_tolerance) and (pd_err_mean < self.pd_error_tolerance)
        else:
            return False

    # Search for a diver
    def perform_search(self):
        msg = Command() 
        msg.roll = 0
        msg.pitch = 0
        msg.yaw = self.search_yaw_speed
        msg.throttle = 0 
        msg.heave = 0

        if(self.search_it_count<10):
            self.cmd_pub.publish(msg)
            self.search_it_count+=1
            rospy.loginfo("ADROC Searching...yawing at %f", msg.yaw)
        elif(self.search_it_count<20):
            self.search_it_count+=1
            rospy.loginfo("ADROC Searching...waiting")
        else:
            self.search_it_count = 0

        return

    def perform_search_lessYaw(self):
        msg = Command() 
        msg.roll = 0
        msg.pitch = 0
        msg.yaw = self.search_yaw_speed
        msg.throttle = 0 
        msg.heave = 0

        if(self.search_it_count<5):
            self.cmd_pub.publish(msg)
            self.search_it_count+=1
            rospy.loginfo("ADROC Searching...yawing at %f", msg.yaw)
        elif(self.search_it_count<10):
            self.search_it_count+=1
            rospy.loginfo("ADROC Searching...waiting")
        else:
            self.search_it_count = 0

        return
    
    def dist(self, p,q):
        return math.sqrt( (p[0]-q[0])**2 + (p[1]-q[1])**2 )
    
    def pose_msg_cb(self, msg):

        left_shoulder = (msg.left_shoulder.x, msg.left_shoulder.y)
        right_shoulder = (msg.right_shoulder.x, msg.right_shoulder.y)
        left_hip = (msg.left_hip.x, msg.left_hip.y)
        right_hip = (msg.right_hip.x, msg.right_hip.y)
        left_elbow = (msg.left_elbow.x, msg.left_elbow.y)
        right_elbow = (msg.right_elbow.x, msg.right_elbow.y)
        left_wrist = (msg.left_wrist.x, msg.left_wrist.y)
        right_wrist = (msg.right_wrist.x, msg.right_wrist.y)
        left_knee = (msg.left_knee.x, msg.left_knee.y)
        right_knee = (msg.right_knee.x, msg.right_knee.y)

        if all(left_shoulder) and all(right_shoulder) and all(left_hip) and all(right_hip) and \
            all(left_elbow) and all(right_elbow) and all(left_wrist) and all(right_wrist) and \
                all(left_knee) and all(right_knee):
            self.poses = [left_shoulder, right_shoulder, left_hip, right_hip, left_elbow, right_elbow, left_wrist, right_wrist, left_knee, right_knee]
            self.pose_detection = True
        else:
            self.pose_detection = False

        if self.pose_detection:
            correct = self.pose_correctness()
            if correct:
                self.compute_JDs()
                # print(self.jds_list)
                # features = self.featureExtractor(jds_list)
                # self.perform_identification(features)

            else:
                print("INCORRECT POSE FOUND. SKIPPING IDENTIFICATION.")

        else:
            print("NO POSE FOUND. SKIPPING IDENTIFICATION.")

    def pose_correctness(self):
        [left_shoulder, right_shoulder, left_hip, right_hip, left_elbow, right_elbow, left_wrist, right_wrist, left_knee, right_knee] = self.poses
        
        lh_lk_dist = self.dist(left_hip, left_knee)
        rh_rk_dist = self.dist(right_hip, right_knee)
        h_k_condition = (lh_lk_dist/rh_rk_dist < 1.2) and (lh_lk_dist/rh_rk_dist > 0.8)
        lw_le_dist = self.dist(left_wrist, left_elbow)
        rw_re_dist = self.dist(right_wrist, right_elbow)
        le_ls_dist = self.dist(left_elbow, left_shoulder)
        re_rs_dist = self.dist(right_elbow, right_shoulder)
        k_x_condition = abs(left_knee[0]-right_knee[0]) > 5
        w_e_condition = (lw_le_dist/rw_re_dist < 1.3) and (lw_le_dist/rw_re_dist > 0.7)
        arm_condition = (lw_le_dist/le_ls_dist < 1.3) and (lw_le_dist/le_ls_dist > 0.7) and (rw_re_dist/re_rs_dist < 1.3) and (rw_re_dist/re_rs_dist > 0.7)


        if ((left_shoulder[1] < left_hip[1]) and (left_hip[1] < left_knee[1]) and
                        abs(left_hip[0]-right_hip[0]) > 8 and h_k_condition and 
                        w_e_condition and arm_condition and k_x_condition                        
                        ):
            return True
            # print(left_shoulder, left_hip, left_knee)
        else:
            return False


    def compute_JDs(self):
        [left_shoulder, right_shoulder, left_hip, right_hip, left_elbow, right_elbow, left_wrist, right_wrist, left_knee, right_knee] = self.poses
        temp_list = []        
        temp_list.append( self.dist(left_shoulder, right_shoulder) )
        temp_list.append( self.dist(left_shoulder, left_hip) )
        temp_list.append( self.dist(left_shoulder, left_elbow) )
        temp_list.append( self.dist(left_elbow, left_wrist) )
        temp_list.append( self.dist(right_hip, right_shoulder) )
        temp_list.append( self.dist(right_elbow, right_shoulder) )
        temp_list.append( self.dist(right_wrist, right_elbow) )
        temp_list.append( self.dist(left_hip, right_hip) )
        temp_list.append( self.dist(left_hip, left_knee) )
        temp_list.append( self.dist(right_hip, right_knee) )

        self.jds_list = temp_list



    def featureExtractor(self, jds_data):
            jds = {}
            jd_names = ['jd'+str(i) for i in range(1,11)]
            print("JDS length, ", len(jds_data))
            for i in range(len(jds_data)):
                jds[jd_names[i]] = jds_data[i]
            
            # jd_names.remove('jd8')
            
            # features = np.zeros((1, 38))
            # f_counter = 1

            # for i in range(len(jd_names)-1):
            #     for j in range(i+1,len(jd_names)):
            #         # if i != j:
            #         data = jds[jd_names[i]]/jds[jd_names[j]]
            #         features[:,f_counter-1] = data
            #         f_counter += 1
                    
            # data = jds['jd1']/(jds['jd3']+jds['jd4']+jds['jd6']+jds['jd7'])         
            # features[:,36] = data

            # data = (jds['jd2']+jds['jd5'])/(jds['jd9']+jds['jd10'])         
            # features[:,37] = data

            upper=['jd1','jd3','jd4','jd6','jd7']
            lower=['jd2','jd5','jd9','jd10']
            features = np.zeros((1, 11))
            f_counter = 1

            for i in range(len(upper)-1):
                for j in range(i+1,len(upper)):
                    data = jds[upper[i]]/jds[upper[j]]
                    features[:,f_counter-1] = data
                    f_counter += 1

            data = jds[upper[0]]/(jds[upper[0]]+jds[upper[1]]+jds[upper[2]]+jds[upper[3]]+jds[upper[4]])
            features[:,f_counter-1] = data

            return features



    def perform_identification(self, features):      
        print("INSIDE IDENTIFICATION")  
        sample = features[[0],:]
        # print(sample.shape)

        pred = self.clf.predict(sample)
        self.total_cnt += 1
        
        self.pred_list.append(str(pred[0]))
        print(self.pred_list)

        acc_thr = 0.6
        ### need to flush out previous values from queue..
        if self.total_cnt >= 6:
            print(self.pred_list)
            self.pred_list.pop(0)
            self.num_pred +=1
            pred_5items = max(set(self.pred_list), key = self.pred_list.count)

            print(f"num_pred:{self.num_pred}, and prediction: {pred_5items}")
            if pred_5items== '1':
                self.correct_cnt += 1
            avg_acc = self.correct_cnt/self.num_pred
            print(f"prediction accuracy is {avg_acc}")   

            if avg_acc > acc_thr:
                return True
            else:
                return False


    # State change function
    def change_state(self, state):
        #TODO check for illegal state transitions.
        self.state = state 

    # Based on current state, check for state change. Change state if required, process state if not.
    def run(self):
        if self.state == ADROCState.INIT:
            if not self.drp_active():
                rospy.loginfo("ADROC State -> SEARCH")
                self.change_state(ADROCState.SEARCH) # If we don't see anyone, go to search
            else:
                rospy.loginfo("ADROC State -> APPROACH")
                self.change_state(ADROCState.APPROACH) #If we see someone, go to them.
                req = TriggerRequest()
                self.activate_drp_controller(req)


        elif self.state == ADROCState.SEARCH:
            if not self.drp_active():
                    rospy.loginfo("ADROC searching...")
                    self.perform_search() #If we're still searching and haven't found yet, continue search operations.
            else:
                self.activate_drp_controller()
                rospy.loginfo("ADROC State -> APPROACH")
                req = TriggerRequest()
                self.activate_drp_controller(req)
                self.change_state(ADROCState.APPROACH) # If we find someone, switch to approach.

        elif self.state == ADROCState.APPROACH:
            if not self.drp_stable():
                rospy.loginfo("ADROC waiting for stable DRP...")
                if not self.drp_active():
                    rospy.loginfo("ADROC approach failed, returning to search")
                    req = TriggerRequest()
                    self.deactivate_drp_controller(req)
                    rospy.loginfo("ADROC State -> SEARCH")
                    self.change_state(ADROCState.SEARCH) # If we don't see anyone, go to search
                else:
                    return #If DRP isn't stable yet, keep waiting for DRP to handle it. We don't need to do anything extra.
            else:
                req = TriggerRequest()
                self.deactivate_drp_controller(req)
                rospy.loginfo("ADROC State -> IDENTIFICATION")
                self.change_state(ADROCState.IDENTIFICATION) # TODO instead of going to ADROCState.IDENTIFICATION, go to GREET.

        elif self.state == ADROCState.IDENTIFICATION:
            rospy.loginfo("ADROC State -> IDENTIFICATION")
            if self.jds_list:
                features = self.featureExtractor(self.jds_list)            
                dec = self.perform_identification(features)
                if dec:
                    rospy.loginfo("ADROC State -> CONCLUDE")
                    self.change_state(ADROCState.CONCLUDE) # If we found the correct person, go to conclude
                else:
                    self.prevID = True
                    rospy.loginfo("ADROC State -> SEARCH_AND_IDENTIFY")
                    self.change_state(ADROCState.SEARCH_AND_IDENTIFY) # If we do not find the correct person, go to search and identify
            else:
                req = TriggerRequest()
                self.deactivate_drp_controller(req)
                rospy.loginfo("ADROC State -> SEARCH")
                self.change_state(ADROCState.SEARCH) # If we don't see anyone, go to search
            # return

        elif self.state == ADROCState.SEARCH_AND_IDENTIFY:
            rospy.loginfo("ADROC State -> SEARCH_AND_IDENTIFY")
            if self.prevID:
                self.perform_search_lessYaw()
                
                req = TriggerRequest()
                self.deactivate_drp_controller(req)
                rospy.loginfo("ADROC State -> IDENTIFICATION")
                self.change_state(ADROCState.IDENTIFICATION) # TODO instead of going to ADROCState.CONCLUDE, go to GREET.


            # if not self.drp_active():
            #         rospy.loginfo("ADROC searching...")
            #         self.perform_search() #If we're still searching and haven't found yet, continue search operations.
            # else:
            #     self.activate_drp_controller()
            #     rospy.loginfo("ADROC State -> APPROACH")
            #     req = TriggerRequest()
            #     self.activate_drp_controller(req)
            #     self.change_state(ADROCState.APPROACH) # If we find someone, switch to approach.            

        elif self.state == ADROCState.CONCLUDE:
            # Print out last stuff or whatever, then switch to shutdown.
            rospy.loginfo("ADROC State -> SHUTDOWN")
            self.change_state(ADROCState.SHUTDOWN)

        else:
            rospy.logerr("ADROC state not recognized.")
            return

    def execute_cb(self, goal):
        # helper variables
        r = rospy.Rate(self.rate)
        success = True
        
        self.initated_time = rospy.get_time()
        self.begin_data_recoding()

        # start executing the action
        while not rospy.is_shutdown() and self.state != ADROCState.SHUTDOWN:
            # check that preempt has not been requested by the client
            if self._as.is_preempt_requested():
                rospy.loginfo('%s: Preempted' % self._action_name)
                self._as.set_preempted()
                success = False
                break

            # Actually do the processing of ADROC states and such
            adroc_diver.run()

            # publish the feedback
            self._feedback.adroc_state_id = self.state
            self._feedback.adroc_state_name = ADROCState.id_to_string(self.state)
            self._as.publish_feedback(self._feedback)

            # this step is not necessary, the sequence is computed at 1 Hz for demonstration purposes
            r.sleep()

        if success:
            rospy.loginfo('%s: Succeeded' % self._action_name)  
            self._result.success = success
            self._result.final_relative_position = self.drp_msgs[-1]
            self._as.set_succeeded(self._result)

        self.end_data_recording()

        #Reset for new ADROC, regardless of what the previous ADROC finished as.
        self.state = ADROCState.INIT
        req = TriggerRequest()
        self.deactivate_drp_controller(req)
    

if __name__ == '__main__':
    rospy.init_node('adroc_diver', anonymous=False)
    adroc_diver = ADROC_Action(rospy.get_name())
    rospy.spin()

else:
    pass
