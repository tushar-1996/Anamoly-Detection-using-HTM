import csv
import re
#import pandas as pd
import numpy as np
#import tensorflow as tf
#from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

START_CUTOFF = 30.0

train_features = []
train_labels = []

val_features = []
val_labels = []

test_features = []
test_labels = []

for trial in [i for i in range(1,8)]:
   if(trial in range(1,5)):
      features = train_features
      labels = train_labels
   elif(trial in range(5,7)):
      features = val_features
      labels = val_labels
   else:
      features = test_features
      labels = test_labels
   csv_reader = csv.reader(open('../Datasets/DataSet/' + str(trial) + '/data.log'), delimiter=',')
   meta = open('../Datasets/DataSet/' + str(trial) + '/META')
   params = meta.read()
   meta.close()
   ATTACK_ONSET = int(re.search('ATTACK_ONSET = (\\d\\d)', params).group(1))
   ATTACK_DURATION = int(re.search('ATTACK_DURATION = (\\d\\d)', params).group(1))
   car1=car2=car3=[]
   for row in csv_reader:
        try:
            if(float(row[6]) > START_CUTOFF):
                if re.match(".*INFO - 1", row[1]):
                    car1=[float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                if re.match(".*INFO - 2", row[1]):
                    car2=[float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                if re.match(".*INFO - 3", row[1]):
                    car3=[float(row[2]), float(row[3]), float(row[4]), float(row[5])]

            if [] not in (car1,car2,car3):
                features.append(car1+car2+car3)
                if (float(row[6]) > ATTACK_ONSET and float(row[6]) <= (ATTACK_ONSET + ATTACK_DURATION)):
                    labels.append(True)
                else:
                    labels.append(False)
                car1=car2=car3=[]
        except:
             pass
"""
print("Train Test Val Features Sizes")
train=np.append(np.array(train_features),np.expand_dims(np.array(train_labels),axis=1),axis=1)
val=np.append(np.array(val_features),np.expand_dims(np.array(val_labels),axis=1),axis=1)
test=np.append(np.array(test_features),np.expand_dims(np.array(test_labels),axis=1),axis=1)
print(np.shape(train))
print(np.shape(val))
print(np.shape(test))

total=np.vstack((train,val,test))
np.savetxt("cardataset2.tsv", total, delimiter="\t", fmt='%.3f')
print(np.shape(total))

newf=np.vstack((np.array(train_features),np.array(test_features),np.array(val_features)))
newl=np.vstack((np.array(train_labels),np.array(test_labels),np.array(val_labels)))
"""

def tsne_scatter(features, labels, dimensions=2, save_as='graph.png'):
    if dimensions not in (2, 3):
        raise ValueError('tsne_scatter can only plot in 2d or 3d (What are you? An alien that can visualise >3d?). Make sure the "dimensions" argument is in (2, 3)')

    # t-SNE dimensionality reduction
    features_embedded = TSNE(n_components=dimensions, random_state=42).fit_transform(features)
    
    # initialising the plot
    fig, ax = plt.subplots(figsize=(8,8))
    
    # counting dimensions
    if dimensions == 3: ax = fig.add_subplot(111, projection='3d')

    # plotting data
    ax.scatter(
        *zip(*features_embedded[np.where(labels==1)]),
        marker='o',
        color='r',
        s=2,
        alpha=0.7,
        label='Fraud'
    )
    ax.scatter(
        *zip(*features_embedded[np.where(labels==0)]),
        marker='o',
        color='g',
        s=2,
        alpha=0.3,
        label='Clean'
    )

    # storing it to be displayed later
    plt.legend(loc='best')
    plt.savefig(save_as);
    plt.show;

tsne_scatter(train_features,train_labels)
"""
train_x = tf.reshape(tf.convert_to_tensor(train_features, dtype=tf.float32), [-1,7,1])
train_y = tf.convert_to_tensor(train_labels)

train_x = tf.data.Dataset.from_tensor_slices(train_x)
train_y = tf.data.Dataset.from_tensor_slices(train_y)
train = tf.data.Dataset.zip((train_x, train_y))

val_x = tf.reshape(tf.convert_to_tensor(val_features, dtype=tf.float32), [-1,7,1])
val_y = tf.convert_to_tensor(val_labels)

val_x = tf.data.Dataset.from_tensor_slices(val_x)
val_y = tf.data.Dataset.from_tensor_slices(val_y)
val = tf.data.Dataset.zip((val_x, val_y))

test_x = tf.reshape(tf.convert_to_tensor(test_features, dtype=tf.float32), [-1,7,1])
test_y = tf.convert_to_tensor(test_labels)

test_x = tf.data.Dataset.from_tensor_slices(test_x)
test_y = tf.data.Dataset.from_tensor_slices(test_y)
test = tf.data.Dataset.zip((test_x, test_y))
"""
