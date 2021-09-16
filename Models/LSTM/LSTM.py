import csv
import re
import pandas as pd
import numpy as np
import tensorflow as tf
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
   elif(trial in range(5,6)):
      features = val_features
      labels = val_labels
   else:
      features = test_features
      labels = test_labels
   csv_reader = csv.reader(open('Datasets/Carrera/' + str(trial) + '/data.log'), delimiter=',')
   meta = open('Datasets/Carrera/' + str(trial) + '/META')
   params = meta.read()
   meta.close()
   ATTACK_ONSET = int(re.search('\tATTACK_ONSET = (\\d\\d)', params).group(1))
   ATTACK_DURATION = int(re.search('\tATTACK_DURATION = (\\d\\d)', params).group(1))
   for row in csv_reader:
	   try:
	      if(float(row[6]) > START_CUTOFF):
	         if re.match(".*INFO - 1", row[1]):
	            features.append(
	            [1, float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])])
	         if re.match(".*INFO - 2", row[1]):
	            features.append(
	            [2, float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])])
	         if re.match(".*INFO - 3", row[1]):
	            features.append(
	            [3, float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])])
	         if (float(row[6]) > ATTACK_ONSET and float(row[6]) <= (ATTACK_ONSET + ATTACK_DURATION)):
	            labels.append(1)
	         else:
	            labels.append(0)
	   except:
	         pass

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
      r = self.recall.result()
      return (2 * (p * r)/(p + r + 1e-6),)

   def update_state(self, y_true, y_pred, sample_weight=None):
      self.precision.update_state(y_true, y_pred, sample_weight=sample_weight)
      self.recall.update_state(y_true, y_pred, sample_weight=sample_weight)

model = tf.keras.Sequential([tf.keras.layers.InputLayer(batch_input_shape=(128,7,1)),
                             tf.keras.layers.LSTM(256, return_sequences=True, stateful=True),
                             tf.keras.layers.LSTM(128, return_sequences=True, stateful=True),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(256, activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal()),
                             tf.keras.layers.Dense(1,activation='sigmoid')])
model.compile(optimizer='RMSProp', loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), F1_Score()])
model.summary()
try:
   model.load_weights('ModelsL/Carrera/LSTM/model')
except:
   model.fit(
       train.padded_batch(128,padded_shapes=((7,1),()), drop_remainder=True),
	    validation_data=val.padded_batch(128, padded_shapes=((7,1),()), drop_remainder=True),
       epochs=200,
       shuffle=False,
	    class_weight={0: 1/6, 1: 5/6},
	    callbacks=[tf.keras.callbacks.CSVLogger('Notebooks/Carrera/LSTM/log.csv'),
	               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_F1', factor=0.5, patience=50),
	               tf.keras.callbacks.ModelCheckpoint('Models/Carrera/LSTM/model', monitor='val_F1', mode='max', save_best_only=True, save_weights_only=True)]
   )

model.load_weights('Models/Carrera/LSTM/model')
model.evaluate(test.padded_batch(128, padded_shapes=((7,1),()), drop_remainder=True))
predictions = np.ravel(model.predict(test_x.padded_batch(128, padded_shapes=(7,1), drop_remainder=True)))
df = pd.DataFrame(predictions)
df.to_csv('Notebooks/Carrera/LSTM/test_results.csv', index=False)

df = pd.DataFrame(test_features,
                   columns=['car#', 'position','velocity','acceleration','predecessor_distance','timestamp','delta_time'])
label_df = pd.DataFrame(test_labels, columns=['label'])
df = df.truncate(after=df.index.size - (df.index.size % 128 + 1), copy=False)
label_df = label_df.truncate(after=df.index.size - (df.index.size % 128 + 1), copy=False)
df.reset_index(drop=True)
label_df.reset_index(drop=True)
df.set_index(df['timestamp'])
df['predictions'] = predictions
df['labels'] = label_df
df[['position','velocity','acceleration','predecessor_distance','predictions','labels']][df['car#']==1].plot(subplots=True, layout=(6,1))
plt.savefig('Reports/LineCharts/LSTM/result1.png')
df[['position','velocity','acceleration','predecessor_distance','predictions','labels']][df['car#']==2].plot(subplots=True, layout=(6,1))
plt.savefig('Reports/LineCharts/LSTM/result2.png')
df[['position','velocity','acceleration','predecessor_distance','predictions','labels']][df['car#']==3].plot(subplots=True, layout=(6,1))
plt.savefig('Reports/LineCharts/LSTM/result3.png')
