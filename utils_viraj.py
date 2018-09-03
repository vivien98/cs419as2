import numpy as np
 

def square_hinge_loss(targets, outputs):
  # Write thee square hinge loss here
  hsl = 0
  for i in range(np.size(outputs)):
    #print(targets[i]*outputs[i])
    if targets[i]*outputs[i]<1:
      hsl += (1 - targets[i]*outputs[i])*(1 - targets[i]*outputs[i])

  return hsl

def logistic_loss(targets, outputs):
  # Write thee logistic loss loss here
  return 1.0

def perceptron_loss(targets, outputs):
  # Write thee perceptron loss here
  hsl = 0
  for i in range(np.size(outputs)):
    if targets[i]*outputs[i]<0:
      hsl += (-targets[i]*outputs[i])
  return hsl

def L2_regulariser(weights):
    # Write the L2 loss here
    l2 = 0;
    for i in range(np.size(weights)):
      l2 += weights[i]*weights[i]
    return l2

def L4_regulariser(weights):
    # Write the L4 loss here
    l4 = 0;
    for i in range(np.size(weights)):
      l4 += weights[i]*weights[i]*weights[i]*weights[i]
    return l4

def square_hinge_grad(weights,inputs, targets, outputs):
  # Write thee square hinge loss gradient here
  grad = np.zeros(np.size(weights))
  for i in range(np.size(targets)):
      if targets[i]*outputs[i]<1:
        grad = np.add(grad,(-2)*(1 - targets[i]*outputs[i])*targets[i]*inputs[i])
  
  return grad

def logistic_grad(weights,inputs, targets, outputs):
  # Write thee logistic loss loss gradient here
    return 1.00 

def perceptron_grad(weights,inputs, targets, outputs):
  # Write thee perceptron loss gradient here
  grad = np.zeros(np.size(weights))
  for i in range(np.size(targets)):
      if targets[i]*outputs[i]<0:
        grad = np.add(grad,(-1)*targets[i]*inputs[i])
  
  return grad

def L2_grad(weights):
    # Write the L2 loss gradient here
    l2g = 2*weights
    return l2g

def L4_grad(weights):
    # Write the L4 loss gradient here
    l4g = weights
    for i in range(np.size(weights)):
      l4g[i] = 4*l4g[i]*weights[i]*weights[i]

    return l4g

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
