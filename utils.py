from __future__ import division
import numpy as np 

def square_hinge_loss(targets, outputs):
	# Write thee square hinge loss here
	sh_loss = 0.00
	if(len(outputs)!=len(targets)): print "something's wrong"
	for i in range(len(outputs)):
		a = targets[i]*outputs[i]
		if(a<1):
			sh_loss += (1-a)*(1-a)
	return sh_loss

def logistic_loss(targets, outputs):	
	# Write thee logistic loss loss here
	if(len(outputs)!=len(targets)): print "something's wrong"
	l_loss = 0.00
	for i in range(len(outputs)):
		a = targets[i]*outputs[i]
		l_loss += np.log(1+np.exp(-a))
	return l_loss

def perceptron_loss(targets, outputs):
	# Write thee perceptron loss here
	if(len(outputs)!=len(targets)): print "something's wrong"	
	p_loss = 0.0
	for i in range(len(outputs)):
		a = targets[i]*outputs[i]
		if(a<0):
			p_loss += -(a)	
	return p_loss

def L2_regulariser(weights):
		# Write the L2 loss here
		l2 = 0.0
		for i in range(len(weights)):
			l2 += weights[i]*weights[i]
		return l2

def L4_regulariser(weights):
		# Write the L4 loss here
		l4 = 0.0
		for i in range(len(weights)):
			a = weights[i]
			l4 += a*a*a*a
		return l4

def y_dot_xVec(y,x):				#return -y*x with last element zero as that corresponds to the bias term
	a = np.zeros(len(x))
	for i in range (len(x)):
		a[i] = -y*x[i]
	return a[i]	

def square_hinge_grad(weights,inputs, targets, outputs):
	# Write thee square hinge loss gradient here
	sh_grad = np.zeros(len(weights))
	for inst in range(len(inputs)):
		if(targets[inst]*outputs[inst]<1):
			c = -2*(1-targets[inst]*outputs[inst])
			sh_grad = np.add(sh_grad,c*inputs[inst]*targets[inst])
	#print sh_grad	
	return sh_grad

def logistic_grad(weights,inputs, targets, outputs):
	# Write thee logistic loss loss gradient here
		l_grad = np.zeros(len(weights))
		for inst in range(len(inputs)):
			expn = np.exp(-outputs[inst]*targets[inst])
			c = -expn/(1+expn)
			l_grad = np.add(l_grad , c*targets[inst]*inputs[inst])
		return l_grad

def perceptron_grad(weights,inputs, targets, outputs):
	# Write thee perceptron loss gradient here
	p_grad = np.zeros(len(weights))
	for inst in range(len(inputs)):
		if(targets[inst]*outputs[inst]<0):	
			p_grad = np.add(p_grad,-targets[inst]*inputs[inst])	
	return p_grad

def L2_grad(weights):
		# Write the L2 loss gradient here
		l2_g = np.zeros(len(weights))
		for i in range(len(weights)):
			l2_g[i] += 2*weights[i]
		return l2_g

def L4_grad(weights):
		# Write the L4 loss gradient here
		l4_g = np.zeros(len(weights))
		for i in range(len(weights)):
			l4_g[i] += 4*weights[i]*weights[i]*weights[i]
		return l4_g

loss_functions = {"square_hinge_loss" : square_hinge_loss, 
									"logistic_loss" : logistic_loss,
									"perceptron_loss" : perceptron_loss}

loss_grad_functions = {"square_hinge_loss" : square_hinge_grad, 
											 "logistic_loss" : logistic_grad,
											 "perceptron_loss" : perceptron_grad}

regularizer_functions = {"L2": L2_regulariser,
												 "L4": L4_regulariser}

regularizer_grad_functions = {"L2" : L2_grad,
															"L4" : L4_grad}
