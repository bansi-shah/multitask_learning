import os, numpy as np, pickle
zeros =0
zerotoone = 0
grone = 0
folder = 'x/'
i = 0

for files in os.listdir('x/'):
   x = pickle.load(open(folder+files, 'rb'))
   zeros = np.where(x < 0, 1, 0).sum()
   zerotoone = np.where(((x >= 0) &  (x <= 1)), 1,0).sum()
   grone = np.where(x > 1, 1, 0).sum() 
   i += 1
   if( i > 40):
      break
   #print(0, 262144, 0)   
   print(zeros, zerotoone, grone)
