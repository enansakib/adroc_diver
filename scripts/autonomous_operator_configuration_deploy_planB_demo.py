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
import collections
import pickle
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from BodyDataset import BodyData
# from identification_modules import TripletNetwork, ClassificationNetwork
# from identification_modules2 import Net, ClassifierNet
from identification_modules_hrnet import Net, ClassifierNet
from identification_modules_hrnet_diver import DiverNet, DiverClassifierNet
from std_msgs.msg import String

import random
import math
from std_msgs.msg import Float64MultiArray
import adroc_diver.msg
# from loco_pilot.srv import Yaw, YawRequest, YawResponse
from loco_pilot.msg import Command
from std_srvs.srv import Trigger, TriggerRequest#, TriggerResponse
import time

## GLOBAL VARIABLES
DIVERS_NUM = 5
FRAMES_THR = 25
# TRAINING_DONE = False
DETECTION_THR = 0.5
# TARGET_DIVER = 4 #3

class ADROCState:
    INIT = 0
    SEARCH = 1
    APPROACH = 2
    DATA_COLLECT = 3
    IDENTIFICATION = 4
    MODEL_TRAINING = 5
    TIMED_YAW = 6
    SUCCESS = 7
    FAIL = 8
    SHUTDOWN = 9

    def id_to_string(id):
        if id == 0:
            return "INIT"
        elif id == 1:
            return "SEARCH"
        elif id == 2:
            return "APPROACH"
        elif id == 3:
            return "DATA_COLLECT"
        elif id == 4:
            return "IDENTIFICATION"
        elif id == 5:
            return "MODEL_TRAINING"
        elif id == 6:
            return "TIMED_YAW"
        elif id == 7:
            return "SUCCESS"
        elif id == 8:
            return "FAIL"
        elif id == 9:
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

        self.prediction = []
        
        self.state_pub = rospy.Publisher('/adroc_diver/state', String, queue_size=1)
        self.label_pub = rospy.Publisher('/adroc_diver/label', String, queue_size=1)
        self.target_pub = rospy.Publisher('/adroc_diver/target', String, queue_size=1)
        self.predict_pub = rospy.Publisher('/adroc_diver/predict', String, queue_size=1)

        self.newMSG = False
        self.jds_list = []
        rospy.Subscriber('/detection/jds', Float64MultiArray, self.jdCallBack, queue_size=1)
        self.collected_frames = 0
        self.processed_diver = 0
        # self.diver_found = False
        self.correct_diver = False
        self.temp_features = []
        self.temp_labels = []
        self.all_features = []
        self.all_labels = []
        self.data_type = 'all' ## ['all', 'diver']
        self.model = 'nn-knn' ## ['nn-knn', nn-svm', 'knn', 'svm']
        self.n_neighbors = 5
        if self.data_type == 'all':
            self.knn_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_knn' # or clf_knndiver
            self.svm_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_svm'
            self.nn_knn_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_nn_knn'
            self.nn_svm_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_nn_svm'
            self.nn1_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/weights/emb_network_all_weights/metric_model_newdata_all_16300_0.9764.pth' ### diver only: weights/emb_network_diver/metric_model_newdata_diver_16200_0.9760.pth
            self.nn2_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/weights/cls_all_weights/metric_model_newdata_cls_700_95.8914.pth' ### diver only: weights/cls_diver_weights/metric_model_newdata_diveronly_cls_200_97.3945.pth
            self.feature_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/all_features.npy'
            self.label_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/all_labels.npy'
        elif self.data_type == 'diver':
            self.knn_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_knn'
            self.svm_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_svm'
            self.nn_knn_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_nn_knndiver'
            self.nn_svm_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_nn_svmdiver'
            self.nn1_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/weights/emb_network_diver_weights/metric_model_newdata_diver_16200_0.9760.pth'
            self.nn2_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/weights/cls_diver_weights/metric_model_newdata_diveronly_cls_200_97.3945.pth'
            self.feature_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/diver_features.npy'
            self.label_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/diver_labels.npy'

        self.batch_size = 256
        self.TRAINING_DONE = False ## Plan A and False for plan B
        self.TARGET_DIVER = 1 #this will only be used when the script is initially run with self.TRAINING_DONE = True

        if self.TRAINING_DONE:
            self.loading_trained_models()

        self.activate_drp_controller = rospy.ServiceProxy('drp_reactive_controller/start', Trigger)
        self.deactivate_drp_controller = rospy.ServiceProxy('drp_reactive_controller/stop', Trigger)
        
        self.cmd_pub = rospy.Publisher('/loco/command', Command, queue_size=10)
        self.search_yaw_speed = 0.1
        self.timed_yaw_speed = 0.1
        self.search_it_count = 0

        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, adroc_diver.msg.ApproachDiverAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

        self.bag_launch = None
        self.lf_cmd = "roslaunch /home/irvlab/adroc_diver_id_ws/src/adroc_diver/launch/data_record.launch"
        self.dg_name = "/data/adroc_diver_digest.txt"
        self.initated_time = None

    def loading_trained_models(self):
        if self.model == 'knn':
            self.clf = pickle.load(open(self.knn_path, 'rb'))

        elif self.model == 'svm':
            self.clf = pickle.load(open(self.svm_path, 'rb'))

        elif self.model == 'nn':
            if self.data_type == 'all':
                net1 = Net()
                net2 = ClassifierNet()      
            elif self.data_type == 'diver':
                net1 = DiverNet()
                net2 = DiverClassifierNet()  

            device = torch.device("cpu")        
            self.clf1, self.clf2 = self.TwoNN_load(device, net1, net2, self.nn1_path, self.nn2_path)

        elif self.model == 'nn-knn':
            self.clf = pickle.load(open(self.nn_knn_path, 'rb'))

        elif self.model == 'nn-svm':
            self.clf = pickle.load(open(self.nn_svm_path, 'rb'))

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
            image_setpoint_y = self.drp_msgs[0].image_h/3.0 #########


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
    

    def jdCallBack(self, jds_topic):
        # print(jds_topic.data)
        self.jds_list = jds_topic.data
        self.newMSG = True        


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
    
    ## Training & Classification
    def TwoNN_load(self, device, net1, net2, PATH1, PATH2):
        clf1 = net1.to(device)
        print(clf1.parameters)
        clf2 = net2.to(device)
        print(clf2.parameters)

        ## NN16NN4 version
        # clf2.fc1= torch.nn.Linear(16, 4)
        
        clf1.load_state_dict(torch.load(PATH1))
        clf2.load_state_dict(torch.load(PATH2))

        return clf1, clf2

    def NN16_load(self, device, net1, PATH1):
        clf1 = net1.to(device)
        # print(clf1.parameters)    
        clf1.load_state_dict(torch.load(PATH1))

        return clf1

    def TwoNN_predict(self, device, clf1, clf2, testloader):
        clf1.eval()
        clf2.eval()
        correct, total = 0, 0
        preds = []
        # no need to calculate gradients during inference
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                # calculate output by running through the network
                output = clf1(inputs.to(device))
                final = clf2(output)
                # get the predictions
                __, predicted = torch.max(final.data, 1)
                preds.append(np.asarray(predicted, dtype=int))
                # update results
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
        
        preds = np.concatenate(preds)
        avg_acc = correct / total

        return avg_acc, preds

    def NN16_out(self, device, clf1, dataloader):
        xout = [] 
        yout = []   
        clf1.eval()
        # no need to calculate gradients during inference
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                # calculate output by running through the network
                output = clf1(inputs.to(device))
                xout.append(output)
                yout.append(labels)
        
        xout = np.concatenate(xout)
        yout = np.concatenate(yout)   

        return xout, yout

    def KNN_fit(self, trainloader, backbone='none'):
        X_train, Y_train = self.torch_dataloader_to_numpy(trainloader)
        clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)        
        
        if backbone == 'none':    
            clf.fit(X_train, Y_train)
            print(f"KNN is fit on {len(trainloader.dataset)} train data.")
            print("saving knn model.")
            
            self.knn_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_knn_runtme'            
            pickle.dump(clf, open(self.knn_path, 'wb'))

        else:
            if self.data_type == 'all':
                net1 = Net()
            elif self.data_type == 'diver':
                net1 = DiverNet()

            device = torch.device("cpu")        
            clf_nn16 = self.NN16_load(device, net1, self.nn1_path)
            xout, yout = self.NN16_out(device, clf_nn16, trainloader)
            # print(xout.shape, yout.shape)

            traindata = BodyData(xout, yout)
            trainloader = DataLoader(traindata, batch_size=self.batch_size, 
                            shuffle=True, num_workers=2)
            
            X_train, Y_train = self.torch_dataloader_to_numpy(trainloader)                
            clf.fit(X_train, Y_train)
            print(f"KNN is fit on {len(trainloader.dataset)} train data.")
            print("saving nn-knn model.")
            
            if self.data_type == 'all':
                self.nn_knn_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_nn_knn_runtme'
            elif self.data_type == 'diver':
                self.nn_knn_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_nn_knndiver_runtme'            
            pickle.dump(clf, open(self.nn_knn_path, 'wb'))
                
        
    def KNN_predict(self, testloader):
        X_test, Y_test = self.torch_dataloader_to_numpy(testloader)
        avg_acc = self.clf.score(X_test, Y_test)
        preds = self.clf.predict(X_test)
        
        return avg_acc, preds
        

    def SVM_fit(self, trainloader, backbone='none'):
        X_train, Y_train = self.torch_dataloader_to_numpy(trainloader)
        clf = svm.SVC()
        
        if backbone == 'none':       
            clf.fit(X_train, Y_train)     
            print(f"SVM is fit on {len(trainloader.dataset)} train data.")
            print("saving svm model.")

            self.svm_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_svm_runtime'                
            pickle.dump(clf, open(self.svm_path, 'wb'))

        else:
            if self.data_type == 'all':
                net1 = Net()
            elif self.data_type == 'diver':
                net1 = DiverNet()

            device = torch.device("cpu")        
            clf_nn16 = self.NN16_load(device, net1, self.nn1_path)
            xout, yout = self.NN16_out(device, clf_nn16, trainloader)
            # print(xout.shape, yout.shape)

            traindata = BodyData(xout, yout)
            trainloader = DataLoader(traindata, batch_size=self.batch_size, 
                            shuffle=True, num_workers=2)
            
            X_train, Y_train = self.torch_dataloader_to_numpy(trainloader)                
            clf.fit(X_train, Y_train)
            print(f"SVM is fit on {len(trainloader.dataset)} train data.")
            print("saving nn-svm model.")

            if self.data_type == 'all':
                self.nn_svm_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_nn_svm_runtime'
            elif self.data_type == 'diver':
                self.nn_svm_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_nn_svmdiver_runtime'

            pickle.dump(clf, open(self.nn_svm_path, 'wb'))


    def SVM_predict(self, testloader):
        X_test, Y_test = self.torch_dataloader_to_numpy(testloader)
        avg_acc = self.clf.score(X_test, Y_test)
        preds = self.clf.predict(X_test)
        
        return avg_acc, preds

    def torch_dataloader_to_numpy(self, dataloader):
        x, y = [], []
        for data in dataloader:
            inputs, labels = data
            x.append(inputs)
            y.append(labels)
        
        x = np.concatenate(x)
        y = np.concatenate(y)

        return x, y

    def perform_identification(self, testdata):
        testloader = DataLoader(testdata, batch_size=self.batch_size, 
                         shuffle=True, num_workers=2)
        
        if self.model == 'knn':            
            avg_acc, preds = self.KNN_predict(testloader) ## avg_acc may be needed for plan B            

        elif self.model == 'svm':            
            avg_acc, preds = self.SVM_predict(testloader) ## avg_acc may be needed for plan B

        elif self.model == 'nn':
            device = torch.device("cpu")
            avg_acc, preds = self.TwoNN_predict(device, self.clf1, self.clf2, testloader)

        elif self.model == 'nn-knn':
            if self.data_type == 'all':
                net1 = Net()
            elif self.data_type == 'diver':
                net1 = DiverNet()

            device = torch.device("cpu")        
            clf_nn16 = self.NN16_load(device, net1, self.nn1_path)
            xout, yout = self.NN16_out(device, clf_nn16, testloader)
            # print(xout.shape, yout.shape)

            testdata = BodyData(xout, yout)
            testloader = DataLoader(testdata, batch_size=self.batch_size, 
                            shuffle=True, num_workers=2)

            avg_acc, preds = self.KNN_predict(testloader)

        elif self.model == 'nn-svm':
            if self.data_type == 'all':
                net1 = Net()
            elif self.data_type == 'diver':
                net1 = DiverNet()

            device = torch.device("cpu")        
            clf_nn16 = self.NN16_load(device, net1, self.nn1_path)
            xout, yout = self.NN16_out(device, clf_nn16, testloader)
            # print(xout.shape, yout.shape)

            testdata = BodyData(xout, yout)
            testloader = DataLoader(testdata, batch_size=self.batch_size, 
                            shuffle=True, num_workers=2)

            avg_acc, preds = self.SVM_predict(testloader)
        
        ## Calculating mode of the predictions
        labels = collections.Counter(preds)
        labels_dict = dict(labels)

        max_value = max(list(labels.values()))
        mode_val = [num for num, freq in labels_dict.items() if freq == max_value]
        print("Predicted : ", mode_val[0], " Target: ", self.TARGET_DIVER)        
        if mode_val[0] == self.TARGET_DIVER: ## mode_val == TARGET_DIVER and processed frame (maybe -1 ) == mode_val --> assumes correct sequence of divers
            self.correct_diver = True

        self.prediction = [str(mode_val[0])]
        # if avg_acc >= DETECTION_THR: ## Revisit for plan B
        #     self.diver_found = True 
        dc = 0
        while dc < 100: # for demo video purpose
            self.predict_pub.publish(str(mode_val[0]))
            time.sleep(0.02)
            dc += 1

    def perform_model_training(self, trainloader):
        if self.model == 'knn':            
            self.KNN_fit(trainloader)        

        elif self.model == 'svm':            
            self.SVM_fit(trainloader)    
        
        elif self.model == 'nn-knn':
            self.KNN_fit(trainloader, backbone='nn16')        

        elif self.model == 'nn-svm':
            self.SVM_fit(trainloader, backbone='nn16')             


    def perform_TimedYaw(self, counter):
        msg = Command() 
        msg.roll = 0
        msg.pitch = 0
        msg.yaw = self.timed_yaw_speed
        msg.throttle = 0 
        msg.heave = 0
        # r = rospy.Rate(30)

        while self.search_it_count < counter:
            self.cmd_pub.publish(msg)
            self.search_it_count+=1
            rospy.loginfo("ADROC is performing a TIMED yaw at %f", msg.yaw)

        self.search_it_count = 0
        time.sleep(8) ## we add this for testing on bench, to have time between switching people
        return


    # State change function
    def change_state(self, state):
        #TODO check for illegal state transitions.
        self.state = state 

    # Based on current state, check for state change. Change state if required, process state if not.
    def run(self):
        if self.TRAINING_DONE:
            self.target_pub.publish(str(self.TARGET_DIVER))

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
                rospy.loginfo("ADROC State -> DATA_COLLECT")                             
                self.change_state(ADROCState.DATA_COLLECT) 
                
        elif self.state == ADROCState.DATA_COLLECT:
            if not self.drp_stable():
                rospy.loginfo("ADROC waiting for stable DRP...")
                if not self.drp_active(): ## NEED TO REVISIT about staying in the same state, see SEARCH
                    rospy.loginfo("ADROC data_collect failed, returning to search")
                    req = TriggerRequest()
                    self.deactivate_drp_controller(req)
                    rospy.loginfo("ADROC State -> SEARCH")
                    self.change_state(ADROCState.SEARCH) # If we don't see anyone, go to search
                else:
                    return #If DRP isn't stable yet, keep waiting for DRP to handle it. We don't need to do anything extra.
            else:

                if self.collected_frames < FRAMES_THR:                
                    ## process frames and add to temporary storage
                    if self.jds_list and self.newMSG:
                        features = self.featureExtractor(self.jds_list)            
                        self.temp_features.append(features)
                        self.temp_labels.append([self.processed_diver])
                        self.collected_frames += 1
                        print("frame # ", self.collected_frames)
                        if not self.TRAINING_DONE:
                            self.label_pub.publish(str(self.processed_diver))
                        else:
                            self.label_pub.publish("")

                    # print("num frames ", self.collected_frames)
                else:
                    ## save all the features
                    self.diver_features = np.concatenate(self.temp_features)
                    self.diver_labels = np.concatenate(self.temp_labels)
                    # print("diver shape: ", self.diver_features.shape, self.diver_labels.shape)
                    self.temp_features = []
                    self.temp_labels = []
                    self.collected_frames = 0

                    if not self.TRAINING_DONE:
                        self.label_pub.publish(str(self.processed_diver))
                    else:
                        self.label_pub.publish("")       

                    self.processed_diver += 1

                    if self.TRAINING_DONE:
                        req = TriggerRequest()
                        self.deactivate_drp_controller(req)
                        rospy.loginfo("ADROC State -> IDENTIFICATION")
                        self.change_state(ADROCState.IDENTIFICATION)
                    
                    else: # PLAN B
                        self.all_features.append(self.diver_features)
                        self.all_labels.append(self.diver_labels)
                        
                        if self.processed_diver == DIVERS_NUM:
                            # creating the final feature to train our model
                            self.all_features = np.concatenate(self.all_features)
                            self.all_labels = np.concatenate(self.all_labels)     
                            # save the combined features
                            np.save(self.feature_path, self.all_features)   
                            np.save(self.label_path, self.all_labels)              
                        
                        req = TriggerRequest()
                        self.deactivate_drp_controller(req)    
                        rospy.loginfo("ADROC State -> TIMED_YAW")           
                        self.change_state(ADROCState.TIMED_YAW)

        elif self.state == ADROCState.IDENTIFICATION:  
            print("TARGET ID: ", self.TARGET_DIVER)
            testdata = BodyData(self.diver_features, self.diver_labels)
            
            self.perform_identification(testdata)          

            if self.processed_diver < DIVERS_NUM:                
                if self.correct_diver:
                    rospy.loginfo("ADROC State -> SUCCESS")
                    self.change_state(ADROCState.SUCCESS)
                else:   
                    self.predict_pub.publish("")
                    rospy.loginfo("ADROC State -> TIMED_YAW")           
                    self.change_state(ADROCState.TIMED_YAW)

            else:                
                if self.correct_diver:
                    rospy.loginfo("ADROC State -> SUCCESS")
                    self.change_state(ADROCState.SUCCESS)
                else: 
                    rospy.loginfo("ADROC State -> FAIL")
                    self.change_state(ADROCState.FAIL)

        elif self.state == ADROCState.MODEL_TRAINING:            
            traindata = BodyData(self.all_features, self.all_labels)
            trainloader = DataLoader(traindata, batch_size=self.batch_size, 
                            shuffle=True, num_workers=2)

            self.perform_model_training(trainloader)
            self.TRAINING_DONE = True
            print("MODEL TRAINED ON COLLECTED DATA")
            self.loading_trained_models()
            print("SWITCHING TO DIVER IDENTIFICATION")
            # self.TARGET_DIVER = DIVERS_NUM - random.randrange(DIVERS_NUM+1)
            self.TARGET_DIVER = 2 ## This is the variable to be used for comparison for online deployment
            

            ## Reset the variables
            self.processed_diver = 0
            self.correct_diver = False
            self.all_features = []
            self.all_labels = []
            
            rospy.loginfo("ADROC State -> INIT")
            self.change_state(ADROCState.INIT)           
                    

        elif self.state == ADROCState.TIMED_YAW:            
            self.perform_TimedYaw(100) ## TUNE this argument, increasing will make it more yaw

            if self.processed_diver == DIVERS_NUM: 
                rospy.loginfo("ADROC State -> MODEL_TRAINING")           
                self.change_state(ADROCState.MODEL_TRAINING)
            
            else:
                rospy.loginfo("ADROC State -> SEARCH")
                self.change_state(ADROCState.SEARCH) # ADROC loses the previous diver using the timed yaw, go to search

        elif self.state == ADROCState.FAIL:
            self.predict_pub.publish(self.prediction[0])

            print("Incorrect diver was recognized.")
            rospy.loginfo("ADROC State -> SHUTDOWN")
            self.change_state(ADROCState.SHUTDOWN)

        elif self.state == ADROCState.SUCCESS:
            self.predict_pub.publish(self.prediction[0])

            print("Correct Diver Found.")
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

            self.state_pub.publish(ADROCState.id_to_string(self.state))

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