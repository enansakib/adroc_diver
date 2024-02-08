import numpy as np
import pickle
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from BodyDataset import BodyData
from identification_modules_hrnet import Net
from identification_modules_hrnet_diver import DiverNet

## Global variables
DIVERS_NUM = 2

class OfflineModelTraining:
    def __init__(self):        
        self.train_features = []
        self.train_labels = []

        self.create_training_data()
        self.n_neighbors = 5
        self.model = 'nn-knn' ## ['nn-knn', nn-svm', 'knn', 'svm']
        self.data_type = 'diver' ## ['all', 'diver']
        
        if self.data_type == 'all':
            self.nn1_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/weights/emb_network_all_weights/metric_model_newdata_all_16300_0.9764.pth'
        elif self.data_type == 'diver':
            self.nn1_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/weights/emb_network_diver_weights/metric_model_newdata_diver_16200_0.9760.pth'
        
        self.batch_size = 256
        traindata = BodyData(self.train_features, self.train_labels)
        trainloader = DataLoader(traindata, batch_size=self.batch_size, 
                            shuffle=True, num_workers=2)
        
        self.perform_model_training(trainloader)
        

    def create_training_data(self):

        for i in range(DIVERS_NUM):
            temp_features = np.load('/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/diver'+str(i)+'_feature.npy')
            temp_labels = np.load('/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/diver'+str(i)+'_label.npy')

            self.train_features.append(temp_features)
            self.train_labels.append(temp_labels)
        
        self.train_features = np.concatenate(self.train_features)
        self.train_labels = np.concatenate(self.train_labels)


    def NN16_load(self, device, net1, PATH1):
        clf1 = net1.to(device)
        # print(clf1.parameters)    
        clf1.load_state_dict(torch.load(PATH1))

        return clf1

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

            self.knn_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_knn_offline'                        
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
                self.nn_knn_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_nn_knn_offline'
            elif self.data_type == 'diver':
                self.nn_knn_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_nn_knndiver_offline'

            pickle.dump(clf, open(self.nn_knn_path, 'wb'))


    def SVM_fit(self, trainloader, backbone='none'):
        X_train, Y_train = self.torch_dataloader_to_numpy(trainloader)
        clf = svm.SVC()   
        
        if backbone == 'none':
            clf.fit(X_train, Y_train)
            print(f"SVM is fit on {len(trainloader.dataset)} train data.")
            print("saving svm model.")

            self.svm_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_svm_offline'                
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
                self.nn_svm_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_nn_svm_offline'
            elif self.data_type == 'diver':
                self.nn_svm_path = '/home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/clf_nn_svmdiver_offline'
            pickle.dump(clf, open(self.nn_svm_path, 'wb'))


    def torch_dataloader_to_numpy(self, dataloader):
        x, y = [], []
        for data in dataloader:
            inputs, labels = data
            x.append(inputs)
            y.append(labels)
        
        x = np.concatenate(x)
        y = np.concatenate(y)

        return x, y

    def perform_model_training(self, trainloader):
        if self.model == 'knn':            
            self.KNN_fit(trainloader)        

        elif self.model == 'svm':            
            self.SVM_fit(trainloader)    
        
        elif self.model == 'nn-knn':
            self.KNN_fit(trainloader, backbone='nn16')        

        elif self.model == 'nn-svm':
            self.SVM_fit(trainloader, backbone='nn16')


OfflineModelTraining()