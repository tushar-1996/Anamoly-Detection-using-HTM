import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def multi_graph(features, predictions, labels, charts):
   df = pd.DataFrame(features, columns=labels, index, index_col=labels[6])
   df.truncate(after=df.size - (df.size % 128))
   df['predictions'] = predictions
   df[charts.append('predictions')].plot(subplots=True, figsize=(6,6))
   
   
