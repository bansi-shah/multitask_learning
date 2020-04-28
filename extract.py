iou_single = []
iou_multi = []
with open('pspnet_single_105.txt') as f:
	for row in f:
		if 'Binary loss' in row:
			row = row.replace('Binary loss : ', '')
			iou_single.append(float(row))



with open('psp_101_multi_2.txt') as f:
	for row in f:
		if 'Validation Binary Loss score' in row:
			row = row.replace('Validation Binary Loss score = ', '')
			iou_multi.append(float(row))


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots(figsize=(11, 8))
plt.plot(range(len(iou_single)), iou_single)
plt.plot(range(len(iou_multi)), iou_multi)
ax1.set_title("Validation Thresholded Binary Accuracy vs Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Thresholded Binary Accuracy")
ax1.legend(['Single Task', 'Multi Task'])
plt.savefig('bta_all.png')
