import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



########################## DECISION TREE #################################


### your code goes here--now create 2 decision tree classifiers,
### one with min_samples_split=2 and one with min_samples_split=50
### compute the accuracies on the testing data and store
### the accuracy numbers to acc_min_samples_split_2 and
### acc_min_samples_split_50, respectively
clf = DecisionTreeClassifier(min_samples_split = 2)
clf.fit(features_train, labels_train)

acc_min_samples_split_2 = clf.score(features_test, labels_test)

clf = DecisionTreeClassifier(min_samples_split = 50)
clf.fit(features_train, labels_train)

acc_min_samples_split_50 = clf.score(features_test, labels_test)

def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}