path = '/home/mtech0/18CS60R31/AD/Semantic-Segmentation-Suite/sal_75_relu/'

import os

for files in os.listdir(path):
	try:
		x = (int)(files)
		if x % 15 != 0 and not x > 150:
			print(files)
			for f in os.listdir(path+files):
				if 'png' in f:
					# print(path+files+f)
					os.remove(path+files+f)
					# os.remove(path+files+'/checkpoint')
					# os.remove(path+files+'/model.ckpt.data-00000-of-00001')
					# os.remove(path+files+'/model.ckpt.index')
					# os.remove(path+files+'/model.ckpt.meta')
	except: 		
		continue	

