import csv
import numpy as np
import re

csv_reader = csv.reader(open('../Datasets/DataSet/1/data.log'), delimiter=',')

list1=[]
list2=[]
list3=[]
for row in csv_reader:
	try:
		float(row[2])
		float(row[3])
		float(row[4])
		float(row[5])
		if re.match(".*INFO - 1", row[1]):
			list1.append([float(row[2]),float(row[3]),float(row[4]),float(row[5])])
		if re.match(".*INFO - 2", row[1]):
			list2.append([float(row[2]),float(row[3]),float(row[4]),float(row[5])])
		if re.match(".*INFO - 3", row[1]):
			list3.append([float(row[2]),float(row[3]),float(row[4]),float(row[5])])
	except:
		pass

nlist1=np.array(list1)
nlist2=np.array(list2)
nlist3=np.array(list3)

save1=(nlist1-np.min(nlist1,axis=0))
save2=(nlist2-np.min(nlist2,axis=0))
save3=(nlist3-np.min(nlist3,axis=0))


np.savetxt('car1data.csv', save1, delimiter=',', fmt='%.3f')
np.savetxt('car2data.csv', save2, delimiter=',', fmt='%.3f')
np.savetxt('car3data.csv', save3, delimiter=',', fmt='%.3f')

		
