import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("npy_files")

for r_type in ("L2","L4"):
	for loss_type in ("logistic_loss","square_hinge_loss","perceptron_loss"):
		plt.figure()
		for c in ("0.5","1.0","5.0","10.0"):
			a = np.load(loss_type+r_type+c+".npy")
			plt.plot(a)
		plt.savefig(loss_type+r_type+".png")
r_type = "None"
for loss_type in ("logistic_loss","square_hinge_loss","perceptron_loss"):
	plt.figure()
	a = np.load(loss_type+"1.0.npy")
	plt.plot(a)
	plt.savefig(loss_type+"no_reg_c1.png")			