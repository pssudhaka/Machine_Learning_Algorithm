import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

 def linear_forward(A,W,b):
    # Z=wx+b is the formula
    Z=np.dot(W,A)+b
    stack=(A,W,b)

    return Z,stack

 def activate_linear_forward(a_prev,W,b,mode)
     Z,stack=linear_forward(a_prev,W,b)

     if mode=="sigmoid":
         A,stack=sigmoid(Z)

     elif mode=="relu"
         A,stack=relu(Z)

     elif mode=="tanh"
        A,stack=tanh(Z)


     stack=(linear_stack , activation_stack)

     return A,stack


def forward_propagation(X,param):
    stacks=[]
    A=X
    L=len(param)//2

    for a in range(1,L):
        a_prev=A
        A,stack=activate_linear_forward(param["W"+str(a)], param["b"+str(a)], 'relu')
        stacks.append(stack)


    AL, stack = linear_forward(A, param["W"+str(L)], param["b"+str(L)], 'sigmoid')
    stacks.append(stack)

    return AL,stacks

def linear_backward(dZ,stack):

    prev_a,W,b=stack
    weight=prev_a.shape[1]

    dW=np.dot(dZ,prev_a.T)
    db=np.sum(dZ,axis=1,keepdims=True)/weight
    da_prev=np.dot(W.T,dZ)

    return da_prev,dW,db

def activate_linear_backward(dA,stack,mode):
    linear_stack,activation_stack=stack

    if mode=="relu"
        dZ=relu_backward(dA,activation_stack)

    elif mode=="sigmoid"
        dZ=sigmoid_backward(dA,activation_stack)

    elif mode=="tanh"
        dZ=tanh_backward(dA,activation_stack)
    da_prev,dW,db=linear_backward(dZ,linear_stack)

    return da_prev,dW,db

def back_propagation(AL,Y,stacks):
    gradients={}
    L=len(stacks)
    m=AL.shae[1]
    Y=Y.reshape(AL.shape)

    dAl=-(np.divide(Y,AL)-np.divide(1-Y,1-AL))

    current_stack=stacks

    gradients["dA"+str(L)], gradients["dW"+str(L)], gradients["db"+str(L)]

    for i in reversed(range(L-1)):
        current_stack=stacks
        dA_prev_temp, dW_temp, db_temp = activate_linear_backward(gradientss["dA" + str(l+2)], current_stack[i], activation = "relu")
        gradients["dA" + str(i + 1)] = dA_prev_temp
        gradients["dW" + str(i + 1)] = dW_temp
        gradients["db" + str(i + 1)] = db_temp

    return gradients

def L_model(X,Y,layer_dims,learning_rate=0.0075,iterations=3000,cost=True):

    cost=[]
    accuracies=[]
    params=initialize_params(layer_dims)
    m=Y.shape[1]

    for i in range(0,iterations):

        AL,stacks=forward_propagation(X,params)
        cost=compute_cost(AL,Y)
        gradients=back_propagation(AL,Y,stacks)
        params=update_params(params, gradients,learning_rate)

        if print_cost and i%100==0:
            print("Cost after iteration",cost)

            accuracy = float(np.sum(np.argmax(Y, axis = 0) == np.argmax(AL, axis = 0)))*100.0/float(42000)
            accuracies.append(accuracy)
            x = randint(0,m)
            print(np.argmax(Y[:, x]), np.argmax(AL[:, x]))
            x = randint(0,m)
            print(np.argmax(Y[:, x]), np.argmax(AL[:, x]))
            x = randint(0,m)
            print(np.argmax(Y[:, x]), np.argmax(AL[:, x]))
            x = randint(0,m)
            print(np.argmax(Y[:, x]), np.argmax(AL[:, x]))
            print("Accuracy: "+str(accuracy))

    return params

# add helper functions

def initialize_params(layer_dims):
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.1
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def update_params(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - (learning_rate * grads["dW" + str(l+1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - (learning_rate * grads["db" + str(l+1)])

    return parameters

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    stack = Z

    return A, stack

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    stack = Z
    return A, stack

def sigmoid_backward(dA, stack):
    Z = stack

    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)

    return dZ

def relu_backward(dA, stack):
    Z = stack
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ

def tanh_backward(dA, cache):
    Z = cache
    s = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    dZ = 1 - (np.exp(s) - np.exp(-s)) / (np.exp(s) + np.exp(-s))
    return dZ

def predict(parameters, data, layers_dims):
    layers = len(layers_dims)
    A = []
    A.append(data)
    for i in range(1, layers):
        A.append(np.dot(parameters["W" + str(i)], A[i-1]) + parameters["b" + str(i)])
    predictions = np.argmax(A[layers-1], axis=0)
    return predictions

