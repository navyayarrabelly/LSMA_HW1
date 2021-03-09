#!/bin/python

import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pickle
import argparse
import sys
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
# Train MLP classifier with labels

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir_sn")
parser.add_argument("feat_dir_mfcc")
parser.add_argument("feat_dim", type=int)
parser.add_argument("train_list_videos")
parser.add_argument("val_list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")


def readFeatfile(videos_file): 
  fread = open(videos_file, "r")
  feat_list = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(videos_file).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category

  nf=0
  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath1 = os.path.join(args.feat_dir_sn, video_id + args.feat_appendix)
    feat_filepath2 = os.path.join(args.feat_dir_mfcc, video_id + args.feat_appendix)
    # for videos with no audio, ignored in training
    features=[]
    if os.path.exists(feat_filepath1) and os.path.exists(feat_filepath2):
      features1 = np.expand_dims(np.genfromtxt(feat_filepath1, delimiter=";", dtype="float"),axis=0)
      features2 = np.expand_dims(np.genfromtxt(feat_filepath2, delimiter=";", dtype="float"),axis=0)
      #features1 = features1[:,250:]
    
      #features = features[:,-200:]
      
      #features = np.concatenate((features1,features2),axis=1)
      features=np.squeeze(features1)
      print(features.shape)
      feat_list.append(features)
      #print(feat_list.shape)

      label_list.append(int(df_videos_label[video_id]))
      #features = 
    else:
        nf +=1
      
    
  print("not found",nf)
  return np.array(feat_list), np.array(label_list)

if __name__ == '__main__':

  args = parser.parse_args()
  

  # 1. read all features in one array.
  


  
  X,y =readFeatfile(args.train_list_videos)
  X= X[:,-1*args.feat_dim:]
  X_val, y_val = readFeatfile(args.val_list_videos)
  X_val= X_val[:,-1*args.feat_dim:]
  print("number of samples: %s" % X.shape[0])
 
  print(X.shape)
  print(y.shape)
  #print(y_val)
  #Hl_list =[(200,100),(200,100,50),(100,50)]
  #lr_list =[]
  mlp = MLPClassifier(hidden_layer_sizes=(100,100,50),max_iter=500, activation="relu", solver="adam",verbose=True,alpha=0.1)
  #mlp = SVC(cache_size=2000, decision_function_shape='ovr', kernel="poly")
  pca = PCA(n_components=200)
  clf = Pipeline(steps=[('pca', pca), ('mlp', mlp)])
  #clf = Pipeline(steps=[('mlp', mlp)])
  clf.fit(X,y)

  bestacc=0
  bestmodel =None
  
  for epoch in range(1,2): 
     #clf =clf.fit(X, y)
     acc= clf.score(X_val, y_val)
     print("epoch",epoch," val acc:", acc)
     if(acc>bestacc): 
            bestmodel = deepcopy(clf)
  
    

  # save trained MLP in output_file
  pickle.dump(clf, open(args.output_file, 'wb'))
  print('MLP classifier trained successfully')
