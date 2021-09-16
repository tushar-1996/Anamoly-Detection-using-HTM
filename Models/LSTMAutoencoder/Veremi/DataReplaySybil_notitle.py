#standard library
import os

#third party
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.metrics

#Local
from Scripts.resolve_shortcut import resolve_shortcut

TRAIN = False
EPOCHS = 1
BATCH_SIZE = 128
NUM_FEATS = 24

TRAIN_DATA_LOCATION = resolve_shortcut('Datasets/Veremi/DataReplaySybil_0709.lnk', '/VeReMi_25200_28800_2019-11-27_16_31_14')
TEST_DATA_LOCATION = resolve_shortcut('Datasets/Veremi/DataReplaySybil_0709.lnk', '/VeReMi_28800_32400_2019-11-27_16_31_14')
MODEL_LOCATION = 'Models/LSTM__Autoencoder_model_Veremi_NoTIle/Veremi_NoTile'
LOG_LOCATION = 'Notebooks/Veremi/DataReplaySybil_0709_NoTile/'
RESULTS_LOCATION = 'Notebooks/Veremi/DataReplaySybil_0709_NoTile/'

data_location = TRAIN_DATA_LOCATION

train_features = pd.read_csv(os.path.join(data_location, 'combined.csv'))
train_features.sort_values('2', ascending=True, inplace=True)
train_features = train_features[train_features['30']==0][[str(i) for i in range(6,31)]].drop(labels='30', axis=1).to_numpy()

features_np = train_features[0:len(train_features) - (len(train_features)%BATCH_SIZE)]
train_x = tf.reshape(tf.convert_to_tensor(train_features, dtype=tf.float32),[-1,NUM_FEATS,1])
train_y = tf.reshape(tf.convert_to_tensor(train_features, dtype=tf.float32),[-1,NUM_FEATS,1])

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

model = tf.keras.Sequential([tf.keras.layers.InputLayer(batch_input_shape=(BATCH_SIZE,NUM_FEATS,1)),
                             tf.keras.layers.LSTM(32, activation='relu', return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.00)),
                             tf.keras.layers.LSTM(8, activation='relu', return_sequences=False),
                             tf.keras.layers.RepeatVector(NUM_FEATS),
                             tf.keras.layers.LSTM(8, activation='relu', return_sequences=True),
                             tf.keras.layers.LSTM(32, activation='relu', return_sequences=True),
                             tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))])
model.compile(optimizer='RMSProp', loss='mae')
model.summary()
try:
   model.load_weights(MODEL_LOCATION)
   if TRAIN:
       model.fit(
           tf.data.Dataset.zip((train_x, train_y)).padded_batch(BATCH_SIZE, padded_shapes=((NUM_FEATS, 1), (NUM_FEATS, 1)),
                                                                drop_remainder=True),
           epochs=EPOCHS,
           shuffle=False,
           callbacks=[tf.keras.callbacks.CSVLogger(LOG_LOCATION +'log.csv'),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50),
                      tf.keras.callbacks.ModelCheckpoint(MODEL_LOCATION, monitor='loss',
                                                         mode='min', save_best_only=True, save_weights_only=True)]
       )
except:
   model.fit(
      tf.data.Dataset.zip((train_x, train_y)).padded_batch(BATCH_SIZE,padded_shapes=((NUM_FEATS,1),(NUM_FEATS,1)), drop_remainder=True),
	   epochs=EPOCHS,
       shuffle=False,
		callbacks=[tf.keras.callbacks.CSVLogger(LOG_LOCATION + 'log.csv'),
	               tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50),
	               tf.keras.callbacks.ModelCheckpoint(MODEL_LOCATION, monitor='loss', mode='min',save_best_only=True, save_weights_only=True)]
   )

train_predictions = model.predict(train_x.padded_batch(BATCH_SIZE, padded_shapes=(NUM_FEATS,1), drop_remainder=True))
MSE = np.array([sklearn.metrics.mean_squared_error(target, predict) for predict, target in zip(sklearn.preprocessing.normalize(features_np,axis=1), train_predictions.reshape((-1,NUM_FEATS)))])

threshold_level = np.mean(MSE)


features = []
labels = []

a = False

START_CUTOFF = 0

data_location = TEST_DATA_LOCATION

test_features = pd.read_csv(os.path.join(data_location, 'combined.csv'))
test_features.sort_values('2', ascending=True, inplace=True)
labels = test_features['30'].to_numpy()
test_features = test_features[[str(i) for i in range(6,31)]].drop(labels='30', axis=1).to_numpy()

test_features = test_features[0:len(test_features) - (len(test_features)%BATCH_SIZE)]
test_features_np = test_features
test_features = tf.reshape(tf.convert_to_tensor(test_features, dtype=tf.float32),[-1,NUM_FEATS,1])
test_features = tf.keras.utils.normalize(test_features, axis=1, order=2)
test_features = tf.data.Dataset.from_tensor_slices(test_features)

labels =np.array(labels[0:len(labels)- (len(labels)%BATCH_SIZE)])

predictions = model.predict(test_features.padded_batch(BATCH_SIZE, padded_shapes=(NUM_FEATS,1), drop_remainder=True))
MSE = np.array([sklearn.metrics.mean_squared_error(target, predict) for predict, target in zip(sklearn.preprocessing.normalize(test_features_np,axis=1), predictions.reshape((-1,NUM_FEATS)))])

np.savetxt(RESULTS_LOCATION + 'actuals.csv', sklearn.preprocessing.normalize(test_features_np, axis=1), delimiter=',')
np.savetxt(RESULTS_LOCATION + 'predictions.csv', predictions.reshape((-1,NUM_FEATS)), delimiter=',')

anomaly_labels = np.where(MSE > threshold_level, 1, 0).astype(int)

precision = sklearn.metrics.precision_score(labels, anomaly_labels, zero_division=0)
recall = sklearn.metrics.recall_score(labels, anomaly_labels, zero_division=0)
f1_score = sklearn.metrics.f1_score(labels, anomaly_labels, zero_division=0)
print(precision)
print(recall)
print(f1_score)