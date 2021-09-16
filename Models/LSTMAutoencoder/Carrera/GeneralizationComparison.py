import csv
import re
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


model = tf.keras.Sequential([tf.keras.layers.InputLayer(batch_input_shape=(128,7,1)),
                             tf.keras.layers.LSTM(32, activation='relu', return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.00)),
                             tf.keras.layers.LSTM(8, activation='relu', return_sequences=False),
                             tf.keras.layers.RepeatVector(7),
                             tf.keras.layers.LSTM(8, activation='relu', return_sequences=True),
                             tf.keras.layers.LSTM(32, activation='relu', return_sequences=True),
                             tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))])
model.compile(optimizer='RMSProp', loss='mae')

model.load_weights('../../Models/LSTM__Autoencoder_model')


threshold_level = 0.07


features_1 = []
labels_1 = []
features_2 = []
labels_2 = []
features_3 = []
labels_3 = []

a = False

START_CUTOFF = 0

for trial in [i for i in range(7,8)]:
   csv_reader = csv.reader(open('../../Datasets/DataSet/' + str(trial) + '/data.log'), delimiter=',')
   meta = open('../../Datasets/DataSet/' + str(trial) + '/META')
   params = meta.read()
   ATTACK_ONSET = int(re.search('\tATTACK_ONSET = (\\d\\d)', params).group(1))
   ATTACK_DURATION = int(re.search('\tATTACK_DURATION = (\\d\\d)', params).group(1))
   for row in csv_reader:
      try:
         if(float(row[6]) > START_CUTOFF):
            if(re.match('.*INFO - 1', row[1])):
               features_3.append(
                  [1, float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])])
               if(float(row[6]) > ATTACK_ONSET and float(row[6]) <= (ATTACK_ONSET + ATTACK_DURATION)):
                  labels_3.append(1)
               else:
                  labels_1.append(0)
           if(re.match('.*INFO - 3', row[1])):
               features_1.append(
                  [2, float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])])
              if(float(row[6]) > ATTACK_ONSET and float(row[6]) <= (ATTACK_ONSET + ATTACK_DURATION)):
                 labels_1.append(1)
              else:
                 labels_1.append(0)
           if(re.match('.*INFO - 2', row[1])):
               features_2.append(
                  [2, float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])])
              if(float(row[6]) > ATTACK_ONSET and float(row[6]) <= (ATTACK_ONSET + ATTACK_DURATION)):
                 labels_2.append(1)
              else:
                 labels_2.append(0)
      except:
         pass

counter = 0
for features, labels in [zip(features_1, labels_1), zip(features_2, labels_2), zip(features_3, labels_3)]:
	counter += 1
	features_np = np.array(features[0:len(features) - (len(features)%128)])
	features = tf.reshape(tf.convert_to_tensor(features, dtype=tf.float32),[-1,7,1])
	features = tf.keras.utils.normalize(features, axis=1, order=2)
	features = tf.data.Dataset.from_tensor_slices(features)

	labels =np.array(labels[0:len(labels)- (len(labels)%128)])

	predictions = model.predict(features.padded_batch(128, padded_shapes=(7,1), drop_remainder=True))
	MSE = np.array([sklearn.metrics.mean_squared_error(target, predict) for predict, target in zip(sklearn.preprocessing.normalize(features_np,axis=1), 		predictions.reshape((-1,7)))])

	anomaly_labels = np.where(MSE > threshold_level, 1, 0).astype(int)

	precision = sklearn.metrics.precision_score(labels, anomaly_labels, zero_division=0)
	recall = sklearn.metrics.recall_score(labels, anomaly_labels, zero_division=0)
	f1_score = sklearn.metrics.f1_score(labels, anomaly_labels, zero_division=0)
	print(precision)
	print(recall)
	print(f1_score)


	df = pd.DataFrame(features_np[int(MSE.shape[0]*.25):],
                     columns=['car#', 'position', 'velocity', 'acceleration', 'predecessor_distance', 'timestamp', 'delta_time'])
	label_df = pd.DataFrame(labels, columns=['label'])
	df['predictions'] = anomaly_labels
	df['labels'] = label_df

	df[['position','velocity', 'acceleration', 'predecessor_distance', 'predictions', 'labels']][df['car#']==counter].plot(subplots=True, layout=(6,1))
	plt.savefig('../../Reports/AutoEncoderCompareResult' + str(counter) + '.png')