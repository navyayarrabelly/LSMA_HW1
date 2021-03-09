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

# Train MLP classifier with labels

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir_sn")
parser.add_argument("feat_dir_mfcc")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
#parser.add_argument("val_list_videos")
parser.add_argument("model_file")
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
      features = np.genfromtxt(feat_filepath1, delimiter=";", dtype="float")
      
      np.concatenate((np.expand_dims(features, axis=0),np.expand_dims(np.genfromtxt(feat_filepath2, delimiter=";", dtype="float"),axis=0)),axis=1)
      np.expand_dims
     
      feat_list.append(features)

      label_list.append(int(df_videos_label[video_id]))
      #features = 
    else:
        nf +=1
      
    
  print("not found",nf)
  return np.array(feat_list), np.array(label_list)


  

  # 1. read all features in one array.
  
if __name__ == '__main__':

  args = parser.parse_args()

  # 1. load mlp model
  mlp = pickle.load(open(args.model_file, "rb"))

  # 2. Create array containing features of each sample
  fread = open(args.list_videos, "r")
  feat_list = []
  video_ids = []
  feat_dim_full =0
  for line in fread.readlines():
    # HW00006228
    video_id = os.path.splitext(line.strip())[0]
    video_ids.append(video_id)
    feat_filepath1 = os.path.join(args.feat_dir_sn, video_id + args.feat_appendix)
    feat_filepath2 = os.path.join(args.feat_dir_mfcc, video_id + args.feat_appendix)
    # for videos with no audio, ignored in training
    features=[]
    if os.path.exists(feat_filepath1) and os.path.exists(feat_filepath2):
      features = np.genfromtxt(feat_filepath1, delimiter=";", dtype="float")
      
      np.concatenate((np.expand_dims(features, axis=0),np.expand_dims(np.genfromtxt(feat_filepath2, delimiter=";", dtype="float"),axis=0)),axis=1)
      np.expand_dims
     
      feat_list.append(features)
      feat_dim_full = (features.shape[0])
    else: 
      feat_list.append(np.zeros(feat_dim_full))
        
  X = np.array(feat_list)
  X= X[:,-1*args.feat_dim:]
  print("number of samples: %s" % X.shape[0])
  pred_classes = mlp.predict(X)

  # 4. save for submission
  with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(pred_classes):
      f.writelines("%s,%d\n" % (video_ids[i], pred_class))
