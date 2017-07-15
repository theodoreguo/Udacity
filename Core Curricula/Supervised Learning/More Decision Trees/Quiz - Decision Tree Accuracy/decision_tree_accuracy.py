import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = DecisionTreeClassifier(min_samples_split = 50)
clf.fit(features_train, labels_train)

acc = clf.score(features_test, labels_test)
### be sure to compute the accuracy on the test set

def submitAccuracies():
  return {"acc":round(acc,3)}