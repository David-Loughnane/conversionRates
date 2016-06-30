import sys
from time import time
sys.path.append("../tools/")
from memory_profiler import memory_usage

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


data_string = "conversion_data.csv"
conv_data = pd.read_csv(data_string)
con_array = np.array(conv_data)

target = conv_data["converted"].values
features = conv_data[["country", "age", "new_user", "total_pages_visited"]].values

for i in range(len(features)):
	if features[i][0] == "UK":
		features[i][0] = 0
	elif features[i][0] == "US":
		features[i][0] = 1
	elif features[i][0] == "China":
		features[i][0] = 2
	else:
		features[i][0] = 3


split = int(0.75*len(target))
features_train = features[0:split]
features_test  = features[split:]
target_train = target[0:split]
target_test  = target[split:]

#memory_tracker = tracker.SummaryTracker()
print("Memory usage before: {}MB".format(memory_usage()))

### FEATURE EXTRACTION ####
#print(features_train)
print("features: ", features_train.shape)

#### INSTATNTIATE REGRESSION ####
reg = LogisticRegression()

#### TRAIN #####
t0 = time()
reg.fit(features_train, target_train)
t1 = time()
print("Regression Coefficients: {}".format(reg.coef_))
print("Training time: {} seconds".format(round(t1-t0,3)))

#### PREDICT ####
t0 = time()
LRprediciton = reg.predict(features_test)
t1 = time()
print("Prediction time: {} seconds".format(round(t1-t0,3)))
print("Prediciton accuracy: {:.2%}".format(metrics.accuracy_score(target_test, LRprediciton)))

#memory_tracker.print_diff()
print("Memory usage after: {}MB".format(memory_usage()))

