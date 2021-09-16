import json
import os
from collections.abc import Iterable
import pandas
import csv

features = []
import csv

TILE_SIZE = 1

with open('../groundtruth.csv', mode='r') as infile:
    reader = csv.reader(infile)
    gt = {rows[0]:rows[1] for rows in reader}
with open('../minimums.txt', mode='r', encoding='utf16') as infile:
	mins = []
	for row in infile.readlines():
		mins.append(float(row))
for file in os.listdir(os.getcwd())[4:]:
	with open(file, mode='r', encoding='utf-8') as jsonfile:
		print(jsonfile.name)
		all = "["
		for i in jsonfile.readlines()[:-1]:
			all += i + ","
		all = all[:-1] + "]"
		try:
			data = json.loads(str(all))
		except(json.decoder.JSONDecodeError):
			continue
	for row in data:
		if(row['type']==3):
			val = []
			try:
				row['GT'] = gt[str(row['sender'])]
			except:
				print("Rejected Row: " + str(row['sender']))
				continue
			row['tile'] = [(row['pos'][0]-mins[0])//TILE_SIZE, (row['pos'][1]-mins[1])//TILE_SIZE]
			for i in row.keys():
				item = row[i]
				if(isinstance(item, Iterable)):
					val.extend(item)
				else:
					val.append(item)
			features.append(val)

pandas.DataFrame(data=features).to_csv('combined.csv')