import csv
import re
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.metrics


train_features = []

csv_reader = csv.reader(open('Datasets/Carrera/8/data.log'), delimiter=',')
for row in csv_reader:
	try:
		if(re.match(".*INFO - 1", row[1])):
			train_features.append([1, float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])])
#		if(re.match(".*INFO - 3"), row[2]):
#			train_features.append([2, float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])])
	except:
		pass

train_x = tf.reshape(tf.convert_to_tensor(train_features, dtype=tf.float32),[-1,7,1])
train_y = tf.reshape(tf.convert_to_tensor(train_features, dtype=tf.float32),[-1,7,1])

train_x = tf.keras.utils.normalize(train_x, axis=1, order=2)
train_y = tf.keras.utils.normalize(train_y, axis=1, order=2)

train_x = tf.data.Dataset.from_tensor_slices(train_x)
train_y = tf.data.Dataset.from_tensor_slices(train_y)
train = tf.data.Dataset.zip((train_x, train_y))

class F1_Score(tf.keras.metrics.Recall):
   def __init__(self, name='F1', **kwargs):
      super().__init__(name=name, **kwargs)
      self.precision = tf.keras.metrics.Precision()
      self.recall = tf.keras.metrics.Recall()

   def reset_state(self):
      super.reset_state()
      self.precision.reset_state()
      self.recall.reset_state()

   def result(self):
      p = self.precision.result()
      r = self.precision.result()
      return (2 * (p * r)/(p + r + 1e-6),)

   def update_state(self, y_true, y_pred, sample_weight=None):
      self.precision.update_state(y_true, y_pred, sample_weight=sample_weight)
      self.recall.update_state(y_true, y_pred, sample_weight=sample_weight)

model = tf.keras.Sequential([tf.keras.layers.InputLayer(batch_input_shape=(128,7,1)),
                             tf.keras.layers.LSTM(32, activation='relu', return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.00)),
                             tf.keras.layers.LSTM(8, activation='relu', return_sequences=False),
                             tf.keras.layers.RepeatVector(7),
                             tf.keras.layers.LSTM(8, activation='relu', return_sequences=True),
                             tf.keras.layers.LSTM(32, activation='relu', return_sequences=True),
                             tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))])
model.compile(optimizer='RMSProp', loss='mae')
model.summary()
try:
   model.load_weights('Models/Carrera/LSTM_Auto/model')
except:
   model.fit(
      tf.data.Dataset.zip((train_x, train_y)).padded_batch(128,padded_shapes=((7,1),(7,1)), drop_remainder=True),
	   epochs=200,
       shuffle=False,
		callbacks=[tf.keras.callbacks.CSVLogger('Notebooks/Carrera/LSTM_Auto/autoencoder_log.csv'),
	               tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50),
	               tf.keras.callbacks.ModelCheckpoint('Models/Carrera/LSTM__Auto/model', monitor='loss', mode='min',save_best_only=True, save_weights_only=True)]
   )

train_predictions = model.predict(train_x.padded_batch(128, padded_shapes=(7,1), drop_remainder=True))


threshold_level = 0.07


features = []
labels = []

a = False

START_CUTOFF = 0

for trial in [i for i in range(7,8)]:
   csv_reader = csv.reader(open('Datasets/Carrera/' + str(trial) + '/data.log'), delimiter=',')
   meta = open('Datasets/Carrera/' + str(trial) + '/META')
   params = meta.read()
   ATTACK_ONSET = int(re.search('\tATTACK_ONSET = (\\d\\d)', params).group(1))
   ATTACK_DURATION = int(re.search('\tATTACK_DURATION = (\\d\\d)', params).group(1))
   for row in csv_reader:
      try:
         if(float(row[6]) > START_CUTOFF):
            if(re.match('.*INFO - 1', row[1])):
               features.append(
                  [1, float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])])
               if(float(row[6]) > ATTACK_ONSET and float(row[6]) <= (ATTACK_ONSET + ATTACK_DURATION)):
                  labels.append(1)
               else:
                  labels.append(0)
#           if(re.match('.*INFO - 3', row[1])):
#               features.append(
#                  [2, float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])])
#              if(float(row[6]) > ATTACK_ONSET and float(row[6]) <= (ATTACK_ONSET + ATTACK_DURATION)):
#                 labels.append(1)
#              else:
#                 labels.append(0)
      except:
         pass

features_np = np.array(features[0:len(features) - (len(features)%128)])
features = tf.reshape(tf.convert_to_tensor(features, dtype=tf.float32),[-1,7,1])
features = tf.keras.utils.normalize(features, axis=1, order=2)
features = tf.data.Dataset.from_tensor_slices(features)

labels =np.array(labels[0:len(labels)- (len(labels)%128)])
labels = labels[int(labels.shape[0]*.25):]

predictions = model.predict(features.padded_batch(128, padded_shapes=(7,1), drop_remainder=True))
MSE = np.array([sklearn.metrics.mean_squared_error(target, predict) for predict, target in zip(sklearn.preprocessing.normalize(features_np,axis=1), predictions.reshape((-1,7)))])

anomaly_labels = np.where(MSE > threshold_level, 1, 0).astype(int)[int(MSE.shape[0]*.25):]
window = 100
for i in range(0,anomaly_labels.size,window):
   if(np.mean(anomaly_labels[i:i+window])>= 0.25):
      anomaly_labels[i:i+window] = 1
   else:
      anomaly_labels[i:i+window] = 0

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

df[['position','velocity', 'acceleration', 'predecessor_distance', 'predictions', 'labels']][df['car#']==1].plot(subplots=True, layout=(6,1))
plt.savefig('Reports/Carrera/LSTM-Auto/AutoEncoderResult1.png')
