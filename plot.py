import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
import warnings
warnings.filterwarnings("ignore")

def plotsegnet(pick):
	img = pickle.load(open(pick, 'rb'))

	img1 = img[0]
	fig, axs = plt.subplots(8, 8, figsize = (8, 8))

	for i in range(8):
		for j in range(8):
			current = np.array(img1[:, :, i*8+j])*255.0
			current = current.reshape((256, 256, 1))
			current = cv2.cvtColor(current, cv2.COLOR_GRAY2RGB)
			axs[i, j].imshow(current)
			axs[i, j].get_yaxis().set_visible(False)
			axs[i, j].get_xaxis().set_visible(False)

	plt.subplots_adjust(wspace=0.001, hspace=0.001)
	plt.savefig('layer1(256, 256, 64).png', bbox_inches='tight')
	plt.clf()

	img1 = img[2]
	fig, axs = plt.subplots(16, 16, figsize = (16, 16))

	for i in range(16):
		for j in range(16):
			current = np.array(img1[:, :, i*16+j])*255.0
			current = current.reshape((64, 64, 1))
			current = cv2.cvtColor(current, cv2.COLOR_GRAY2RGB)
			axs[i, j].imshow(current)
			axs[i, j].get_yaxis().set_visible(False)
			axs[i, j].get_xaxis().set_visible(False)

	plt.subplots_adjust(wspace=0.001, hspace=0.001)
	plt.savefig('layer3(64, 64, 256).png',bbox_inches='tight')
	plt.clf()


	img1 = img[1]
	fig, axs = plt.subplots(8, 8, figsize = (8, 8))

	for i in range(8):
		for j in range(8):
			current = np.array(img1[:, :, i*8+j])*255.0
			current = current.reshape((128, 128, 1))
			current = cv2.cvtColor(current, cv2.COLOR_GRAY2RGB)
			axs[i, j].imshow(current)
			axs[i, j].get_yaxis().set_visible(False)
			axs[i, j].get_xaxis().set_visible(False)

	plt.subplots_adjust(wspace=0.001, hspace=0.001)
	plt.savefig('layer2(128, 128, 0..64).png', bbox_inches='tight')
	plt.clf()

	fig, axs = plt.subplots(8, 8, figsize = (8, 8))
	for i in range(8):
		for j in range(8, 16):
			current = np.array(img1[:, :, i*8+j])*255.0
			current = current.reshape((128, 128, 1))
			current = cv2.cvtColor(current, cv2.COLOR_GRAY2RGB)
			axs[i, j-8].imshow(current)
			axs[i, j-8].get_yaxis().set_visible(False)
			axs[i, j-8].get_xaxis().set_visible(False)

	plt.subplots_adjust(wspace=0.001, hspace=0.001)
	plt.savefig('layer2(128, 128, 64..128).png', bbox_inches='tight')
	plt.clf()

	img1 = cv2.imread('layer2(128, 128, 0..64).png')
	img2 = cv2.imread('layer2(128, 128, 64..128).png')
	vis = np.concatenate((img1, img2), axis=1)
	cv2.imwrite('layer2(128, 128, 128).png', vis)
	os.remove("layer2(128, 128, 0..64).png")
	os.remove("layer2(128, 128, 64..128).png")

	img1 = img[3]
	fig, axs = plt.subplots(16, 16, figsize = (16, 16))

	for i in range(16):
		for j in range(16):
			current = np.array(img1[:, :, i*16+j])*255.0
			current = current.reshape((32, 32, 1))
			current = cv2.cvtColor(current, cv2.COLOR_GRAY2RGB)
			axs[i, j].imshow(current)
			axs[i, j].get_yaxis().set_visible(False)
			axs[i, j].get_xaxis().set_visible(False)

	plt.subplots_adjust(wspace=0.001, hspace=0.001)
	plt.savefig('layer4(32, 32, 0..256).png', bbox_inches='tight')
	plt.clf()

	fig, axs = plt.subplots(16, 16, figsize = (16, 16))

	for i in range(16):
		for j in range(16, 32):
			current = np.array(img1[:, :, i*16+j])*255.0
			current = current.reshape((32, 32, 1))
			current = cv2.cvtColor(current, cv2.COLOR_GRAY2RGB)
			axs[i, j-16].imshow(current)
			axs[i, j-16].get_yaxis().set_visible(False)
			axs[i, j-16].get_xaxis().set_visible(False)

	plt.subplots_adjust(wspace=0.001, hspace=0.001)
	plt.savefig('layer4(32, 32, 256..512).png', bbox_inches='tight')
	plt.clf()

	img1 = cv2.imread('layer4(32, 32, 0..256).png')
	img2 = cv2.imread('layer4(32, 32, 256..512).png')
	vis = np.concatenate((img1, img2), axis=1)
	cv2.imwrite('layer4(32, 32, 512).png', vis)
	os.remove("layer4(32, 32, 0..256).png")
	os.remove("layer4(32, 32, 256..512).png")	


def plotpspnet(pick):
	img = pickle.load(open(pick, 'rb'))
	# img = img.reshape(-1, 64, 64, 3)
	fig, axs = plt.subplots(32, 36)
	k, l = 0, 0

	for i in range(1152):
		current = np.array(img[:, :, i])*255.0
		current = current.reshape((64, 64, 1))
		current = cv2.cvtColor(current, cv2.COLOR_GRAY2RGB)
		axs[k, l].imshow(current)
		axs[k, l].get_yaxis().set_visible(False)
		axs[k, l].get_xaxis().set_visible(False)
		k += 1
		if k == 32:
			k = 0
			l += 1

	plt.subplots_adjust(wspace=0.001, hspace=0.001)
	plt.savefig('psp1.png', bbox_inches='tight')
	plt.clf()
	
	fig, axs = plt.subplots(32, 36)
	k, l = 0, 0

	for i in range(1152, 2304):
		current = np.array(img[:, :, i])*255.0
		current = current.reshape((64, 64, 1))
		current = cv2.cvtColor(current, cv2.COLOR_GRAY2RGB)
		axs[k, l].imshow(current)
		axs[k, l].get_yaxis().set_visible(False)
		axs[k, l].get_xaxis().set_visible(False)
		k += 1
		if k == 32:
			k = 0
			l += 1

	plt.subplots_adjust(wspace=0.001, hspace=0.001)
	plt.savefig('psp2.png', bbox_inches='tight')
	plt.clf()

	img1 = cv2.imread('psp1.png')
	img2 = cv2.imread('psp2.png')
	vis = np.concatenate((img1, img2), axis=1)
	cv2.imwrite('Test/'+pick.split('/')[-1].split('.')[0] +'.png', vis)
	os.remove("psp1.png")
	os.remove("psp2.png")


def plotpspnet_(pick):
	img = pickle.load(open(pick, 'rb'))
	current = np.array(img[:, :, :])*255.0

	pick  = pick.replace('seg', 'sal')
	img = pickle.load(open(pick, 'rb'))
	current1 = np.array(img[:, :, :])*255.0

	pick  = pick.replace( 'sal', 'outs')
	img = pickle.load(open(pick, 'rb'))
	current2 = np.array(img[:, :, :])*255.0

	# print(pick.replace('_outs.pkl', ''))
	print('seg', np.count_nonzero(current1), 'sal', np.count_nonzero(current), 'all', np.count_nonzero(current2))	
	

for files in os.listdir('Test/'):
	if 'seg.pkl' in files:
		plotpspnet_('Test/'+ files)