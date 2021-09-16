import csv
import datetime
import os
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters

from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood #FIXME use TM.anomaly instead, but it gives worse results than the py.AnomalyLikelihood now
from htm.bindings.algorithms import Predictor


base_dir = 'C:/Users/Ethan Swistak/OneDrive/Ulm/Summer 2021/Autonomous Vehicles Project/Project'
os.chdir(base_dir)

_DATA_DIR = 'Datasets/Veremi/ConstPos_0709'
_INPUT_FILE_PATH = os.path.join(_DATA_DIR, "combined1.csv")

"""
default_parameters = {
 'enc': {
      "position" :
         {'resolution': 0.88, 'size': 1000, 'sparsity': 0.02},
      "velocity" :
         {'resolution': 0.88, 'size': 800, 'sparsity': 0.02},
      "acceleration" :
         {'resolution': 0.88, 'size': 1500, 'sparsity': 0.02},
      "distance" :
         {'resolution': 0.88, 'size': 600, 'sparsity': 0.02},
      "time": 
         {'timeOfDay': (30, 1), 'weekend': 21}
 },
 'sp': {'boostStrength': 3.0,
        'columnCount': 6000,
        'localAreaDensity': 0.04395604395604396,
        'potentialPct': 0.85,
        'synPermActiveInc': 0.04,
        'synPermConnected': 0.13999999999999999,
        'synPermInactiveDec': 0.006},
 'tm': {'activationThreshold': 17,
        'cellsPerColumn': 13,
        'initialPerm': 0.21,
        'maxSegmentsPerCell': 128,
        'maxSynapsesPerSegment': 64,
        'minThreshold': 10,
        'newSynapseCount': 32,
        'permanenceDec': 0.1,
        'permanenceInc': 0.1},
}
"""
default_parameters = {'enc': {
          'sendtime': {'resolution': 0.9828312017527027,
                          'size': 500,
                          'sparsity': 0.02295940926954918},

         'sender': {'resolution': 0.8715000068165106,
                      'size': 1226,
                      'sparsity': 0.019115440524140735},

         'position': {'resolution': 0.8499176653753064,
                      'size': 500,
                      'sparsity': 0.018592851908209966},

         'velocity': {'resolution': 0.8499176653753064,
                      'size': 900,
                      'sparsity': 0.018592851908209966},

          'acceleration': {'resolution': 0.8499176653753064,
                      'size': 900,
                      'sparsity': 0.018592851908209966},

          'distance': {'resolution': 0.8499176653753064,
                      'size': 900,
                      'sparsity': 0.018592851908209966}},


'sp': {'boostStrength': 3.1156287207658115,

        'columnCount': 5113,

        'localAreaDensity': 0.04414495631003853,

        'potentialPct': 0.8432597172127386,

        'synPermActiveInc': 0.03430438570209145,

        'synPermConnected': 0.15813545310507207,

        'synPermInactiveDec': 0.006409654753680373},

'tm': {'activationThreshold': 21,

        'cellsPerColumn': 12,

        'initialPerm': 0.20779610481030433,

        'maxSegmentsPerCell': 129,

        'maxSynapsesPerSegment': 65,

        'minThreshold': 9,

        'newSynapseCount': 38,

        'permanenceDec': 0.09400339238429853,

        'permanenceInc': 0.07722921060280212}}


def main(parameters=default_parameters, argv=None, verbose=True):
  if verbose:
    import pprint
    print("Parameters:")
    pprint.pprint(parameters, indent=4)
    print("")

  # Read the input file.
  records = []
  with open(_INPUT_FILE_PATH, "r") as fin:
    reader = csv.reader(fin)
    next(reader)
    for record in reader:
      records.append(record)

  TimescalarEncoderParams            = RDSE_Parameters()
  TimescalarEncoderParams.size       = parameters["enc"]["sendtime"]["size"]
  TimescalarEncoderParams.sparsity   = parameters["enc"]["sendtime"]["sparsity"]
  TimescalarEncoderParams.resolution = parameters["enc"]["sendtime"]["resolution"]
  TimescalarEncoder = RDSE( TimescalarEncoderParams )

  SenderscalarEncoderParams            = RDSE_Parameters()
  SenderscalarEncoderParams.size       = parameters["enc"]["sender"]["size"]
  SenderscalarEncoderParams.sparsity   = parameters["enc"]["sender"]["sparsity"]
  SenderscalarEncoderParams.resolution = parameters["enc"]["sender"]["resolution"]
  SenderscalarEncoder = RDSE( SenderscalarEncoderParams )
  #Encoder for Position
  PosscalarEncoderParams            = RDSE_Parameters()
  PosscalarEncoderParams.size       = parameters["enc"]["position"]["size"]
  PosscalarEncoderParams.sparsity   = parameters["enc"]["position"]["sparsity"]
  PosscalarEncoderParams.resolution = parameters["enc"]["position"]["resolution"]
  PosscalarEncoder = RDSE( PosscalarEncoderParams )
  #Encoder for Velocity
  VelscalarEncoderParams            = RDSE_Parameters()
  VelscalarEncoderParams.size       = parameters["enc"]["velocity"]["size"]
  VelscalarEncoderParams.sparsity   = parameters["enc"]["velocity"]["sparsity"]
  VelscalarEncoderParams.resolution = parameters["enc"]["velocity"]["resolution"]
  VelscalarEncoder = RDSE( VelscalarEncoderParams )
  #Encoder for Acceleration
  AccscalarEncoderParams            = RDSE_Parameters()
  AccscalarEncoderParams.size       = parameters["enc"]["acceleration"]["size"]
  AccscalarEncoderParams.sparsity   = parameters["enc"]["acceleration"]["sparsity"]
  AccscalarEncoderParams.resolution = parameters["enc"]["acceleration"]["resolution"]
  AccscalarEncoder = RDSE( AccscalarEncoderParams )
  #Encoder for Distance
  DisscalarEncoderParams            = RDSE_Parameters()
  DisscalarEncoderParams.size       = parameters["enc"]["distance"]["size"]
  DisscalarEncoderParams.sparsity   = parameters["enc"]["distance"]["sparsity"]
  DisscalarEncoderParams.resolution = parameters["enc"]["distance"]["resolution"]
  DisscalarEncoder = RDSE( DisscalarEncoderParams )

  #Change
  encodingWidth = ( SenderscalarEncoder.size+TimescalarEncoder.size+PosscalarEncoder.size*2+VelscalarEncoder.size*2+AccscalarEncoder.size*2+DisscalarEncoderParams.size*2)
  enc_info = Metrics( [encodingWidth], 999999999 )

  # Make the HTM.  SpatialPooler & TemporalMemory & associated tools.
  spParams = parameters["sp"]
  sp = SpatialPooler(
    inputDimensions            = (encodingWidth,),
    columnDimensions           = (spParams["columnCount"],),
    potentialPct               = spParams["potentialPct"],
    potentialRadius            = encodingWidth,
    globalInhibition           = True,
    localAreaDensity           = spParams["localAreaDensity"],
    synPermInactiveDec         = spParams["synPermInactiveDec"],
    synPermActiveInc           = spParams["synPermActiveInc"],
    synPermConnected           = spParams["synPermConnected"],
    boostStrength              = spParams["boostStrength"],
    wrapAround                 = True
  )
  sp_info = Metrics( sp.getColumnDimensions(), 999999999 )

  tmParams = parameters["tm"]
  tm = TemporalMemory(
    columnDimensions          = (spParams["columnCount"],),
    cellsPerColumn            = tmParams["cellsPerColumn"],
    activationThreshold       = tmParams["activationThreshold"],
    initialPermanence         = tmParams["initialPerm"],
    connectedPermanence       = spParams["synPermConnected"],
    minThreshold              = tmParams["minThreshold"],
    maxNewSynapseCount        = tmParams["newSynapseCount"],
    permanenceIncrement       = tmParams["permanenceInc"],
    permanenceDecrement       = tmParams["permanenceDec"],
    predictedSegmentDecrement = 0.0,
    maxSegmentsPerCell        = tmParams["maxSegmentsPerCell"],
    maxSynapsesPerSegment     = tmParams["maxSynapsesPerSegment"]
  )
  tm_info = Metrics( [tm.numberOfCells()], 999999999 )

  # setup likelihood, these settings are used in NAB
  """
  anParams = parameters["anomaly"]["likelihood"]
  probationaryPeriod = int(math.floor(float(anParams["probationaryPct"])*len(records)))
  learningPeriod     = int(math.floor(probationaryPeriod / 2.0))
  anomaly_history = AnomalyLikelihood(learningPeriod= learningPeriod,
                                      estimationSamples= probationaryPeriod - learningPeriod,
                                      reestimationPeriod= anParams["reestimationPeriod"])

  predictor = Predictor( steps=[1, 5], alpha=parameters["predictor"]['sdrc_alpha'] )
  predictor_resolution = 1
  """

  # Iterate through every datum in the dataset, record the inputs & outputs.
  itime = []
  isender = []
  iposition = []
  ivelocity = []
  iacceleration = []
  idistance = []
  ianoclass = []
  anomaly   = []
  anoscore  = []
  #anomalyProb = []
  #predictions = {1: [], 5: []}
  window=[]
  for count, record in enumerate(records):

    # Convert date string into Python date object.
    #dateString = datetime.datetime.strptime(record[0], "%m/%d/%y %H:%M")
    # Convert data value string into float.
    time = float(record[0])
    sender = float(record[1])
    postion1 = float(record[2])
    postion2 = float(record[3])
    velocity1 = float(record[4])
    velocity2 = float(record[5])
    acceleration1 = float(record[6])
    acceleration2 = float(record[7])
    distance1 = float(record[8])
    distance2 = float(record[9])
    #inputs.append( acceleration )

    #iposition.append(postion)
    #iacceleration.append(acceleration)
    #ivelocity.append(velocity)
    #idistance.append(distance)

    # Call the encoders to create bit representations for each value.  These are SDR objects.
    #dateBits        = dateEncoder.encode(dateString)
    TimeBits = TimescalarEncoder.encode(time)
    SenderBits = SenderscalarEncoder.encode(sender)

    Postion1Bits = PosscalarEncoder.encode(postion1)
    Velocity1Bits = VelscalarEncoder.encode(velocity1)
    Accleration1Bits = AccscalarEncoder.encode(acceleration1)
    Distance1Bits = DisscalarEncoder.encode(distance1)

    Postion2Bits = PosscalarEncoder.encode(postion2)
    Velocity2Bits = VelscalarEncoder.encode(velocity2)
    Accleration2Bits = AccscalarEncoder.encode(acceleration2)
    Distance2Bits = DisscalarEncoder.encode(distance2)

    # Concatenate all these encodings into one large encoding for Spatial Pooling.
    encoding = SDR( encodingWidth ).concatenate([TimeBits,SenderBits,Postion1Bits,Postion2Bits,Velocity1Bits,Velocity2Bits,Accleration1Bits,Accleration2Bits,Distance1Bits,Distance2Bits])
    enc_info.addData( encoding )

    # Create an SDR to represent active columns, This will be populated by the
    # compute method below. It must have the same dimensions as the Spatial Pooler.
    activeColumns = SDR( sp.getColumnDimensions() )

    # Execute Spatial Pooling algorithm over input space.
    sp.compute(encoding, True, activeColumns)
    sp_info.addData( activeColumns )

    # Execute Temporal Memory algorithm over active mini-columns.
    tm.compute(activeColumns, learn=True)
    tm_info.addData( tm.getActiveCells().flatten() )

    # Predict what will happen, and then train the predictor based on what just happened.
    '''
    pdf = predictor.infer( tm.getActiveCells() )
    for n in (1, 5):
      if pdf[n]:
        predictions[n].append( np.argmax( pdf[n] ) * predictor_resolution )
      else:
        predictions[n].append(float('nan'))
    '''

    #anomalyLikelihood = anomaly_history.anomalyProbability( acceleration, tm.anomaly )
    anoscore.append(tm.anomaly)

    """
    if tm.anomaly>=0.05 :
      window.append(1)
    else:
      window.append(0)

    winlen=20
    split=0.5

    if (count%winlen == 1):
      total=sum(window)
      if total>=(winlen*split):
        anomaly.extend([1]*winlen)
      else:
        anomaly.extend([0]*winlen)
      window=[]      
    """

    #anomalyProb.append( anomalyLikelihood )

    #predictor.learn(count, tm.getActiveCells(), int(acceleration / predictor_resolution))

  """
  # Print information & statistics about the state of the HTM.
  print("Encoded Input", enc_info)
  print("")
  print("Spatial Pooler Mini-Columns", sp_info)
  print(str(sp))
  print("")
  print("Temporal Memory Cells", tm_info)
  print(str(tm))
  print("")
  """

  # Shift the predictions so that they are aligned with the input they predict.
  """
  for n_steps, pred_list in predictions.items():
    for x in range(n_steps):
        pred_list.insert(0, float('nan'))
        pred_list.pop()

  # Calculate the predictive accuracy, Root-Mean-Squared
  accuracy         = {1: 0, 5: 0}
  accuracy_samples = {1: 0, 5: 0}

  for idx, inp in enumerate(inputs):
    for n in predictions: # For each [N]umber of time steps ahead which was predicted.
      val = predictions[n][ idx ]
      if not math.isnan(val):
        accuracy[n] += (inp - val) ** 2
        accuracy_samples[n] += 1
  for n in sorted(predictions):
    accuracy[n] = (accuracy[n] / accuracy_samples[n]) ** .5
    print("Predictive Error (RMS)", n, "steps ahead:", accuracy[n])
  """
  
  """
  # Show info about the anomaly (mean & std)
  print("Anomaly Mean", np.mean(anomaly))
  print("Anomaly Std ", np.std(anomaly))

  # Plot the Predictions and Anomalies.

  iposition = np.array(iposition) / max(iposition)
  ivelocity = np.array(ivelocity) / max(ivelocity)
  iacceleration = np.array(iacceleration) / max(iacceleration)
  idistance = np.array(idistance) / max(idistance)
  """
  #Printing metrics
  '''
  a = np.array(anomaly)
  a[a>=0.1] = 1
  a[a<0.1] = 0
  '''

  """
  while(len(ianoclass)!=len(anomaly)):
    ianoclass.append(0)


  # accuracy: (tp + tn) / (p + n)
  accuracy = accuracy_score(ianoclass[1000:], anomaly[1000:])
  print('Accuracy: %f' % accuracy)
  # precision tp / (tp + fp)
  precision = precision_score(ianoclass[1000:], anomaly[1000:])
  print('Precision: %f' % precision)
  # recall: tp / (tp + fn)
  recall = recall_score(ianoclass[1000:], anomaly[1000:])
  print('Recall: %f' % recall)
  # f1: 2 tp / (2 tp + fp + fn)
  f1 = f1_score(ianoclass[1000:], anomaly[1000:])
  print('F1 score: %f' % f1)






  plt.style.use('dark_background')
  plt.subplot(6,1,1)
  plt.xlabel("Time")
  plt.ylabel("Position")
  plt.plot(np.arange(len(iposition)), iposition, 'blue',)

  plt.subplot(6,1,2)
  plt.xlabel("Time")
  plt.ylabel("Velocity")
  plt.plot(np.arange(len(iposition)), ivelocity, 'yellow',)

  plt.subplot(6,1,3)
  plt.xlabel("Time")
  plt.ylabel("Acceleration")
  plt.plot(np.arange(len(iposition)), iacceleration, 'green',)

  plt.subplot(6,1,4)
  plt.xlabel("Time")
  plt.ylabel("Predecessor Distance")
  plt.plot(np.arange(len(iposition)), idistance, 'orange',)

  plt.subplot(6,1,5)
  plt.xlabel("Time")
  plt.ylabel("Anomaly Score")
  plt.bar(np.arange(len(anoscore)), anoscore, color='red',width=1)
  """
  plt.subplot(1,1,1)
  plt.xlabel("Time")
  plt.ylabel("Anomaly Predicted")
  plt.bar(np.arange(len(anoscore)), anoscore, color='red',width=1)

  np.savetxt("anomalyscore.txt", np.array(anoscore), delimiter=",",fmt='%.4f')

  plt.savefig('line_plot_hq.png', dpi=300)  


  return -1


if __name__ == "__main__":
  main()
