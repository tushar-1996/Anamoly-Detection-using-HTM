import csv
data_file = open('output.csv', 'w', newline='')
csv_writer = csv.writer(data_file)

#header=["type","sendTime","sender","senderPseudo","messageID","pos1","pos2","pos3","pos_noise1","pos_noise2","pos_noise3","spd1","spd2","spd3","spd_noise1","spd_noise2","spd_noise3","acl1","acl2","acl3","acl_noise1","acl_noise2","acl_noise3","hed1","hed2","hed3","hed_noise1","hed_noise2","hed_noise3"]
#header=["type","sendTime","sender","senderPseudo","messageID","pos1","pos2","pos_noise1","pos_noise2","spd1","spd2","spd_noise1","spd_noise2","acl1","acl2","acl_noise1","acl_noise2","hed1","hed2","hed_noise1","hed_noise2"]
#header=["type","sendTime","sender","senderPseudo","messageID","pos1","pos2","spd1","spd2","acl1","acl2","hed1","hed2"]
#header=["sendTime","sender","pos1","pos2","spd1","spd2","acl1","acl2","hed1","hed2"]

#csv_writer.writerow(header)

row=[]
f_train = open("traceJSON-11073-11071-A4-28800-8.json", 'r')
for line in f_train:
	a=line.strip('\n{}\"').split(',\"')
	for l in a:
		col=l[:l.index("\":")]
		val=l[l.index("\":")+2:]
		if (("type" in col) and int(val)==2):
			break;
		if ("noise" not in col) and ("senderPseudo" not in col) and ("type" not in col) and ("messageID" not in col) and ("rcvTime" not in col):
			try:
				row.append(float("{:.2f}".format(float(val))))
			except:
				ext=val.strip('[]').split(',')
				for i in ext[:2]:
					row.append(float("{:.2f}".format(float(i))))
	csv_writer.writerow(row)
	row=[]

f_train.close()
data_file.close()
