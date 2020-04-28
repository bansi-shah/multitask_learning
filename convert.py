
f = open('complexity_109.txt', 'r').read().split()
dict = {}

for i in range(0, len(f), 2):
  dict[f[i]] = int(f[i+1])

import pickle as pkl
with open('complexity_selectivesearch.pickle', 'wb') as x:
  pkl.dump(dict, x, protocol = pkl.HIGHEST_PROTOCOL)



