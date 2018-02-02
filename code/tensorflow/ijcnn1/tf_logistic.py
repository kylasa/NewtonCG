from __future__ import print_function

import sys
import math
import tensorflow as tf
import numpy as np
import csv

import cPickle as pickle
import time
import StringIO

trainmat = 'train_mat.txt'
trainvec = 'train_vec.txt'
testmat = 'test_mat.txt'
testvec = 'test_vec.txt'

curpath = sys.argv[2]

print ()
print (curpath)
print ()
print ()


# Test
if sys.argv[2].find("raw-data") != -1:
    X_train = np.genfromtxt( curpath + trainmat ,delimiter=',',dtype=None )
    X_train = X_train.astype(np.float64)
    Y_train = np.genfromtxt( curpath + trainvec ,delimiter=',',dtype=None)
    Y_train = Y_train.astype(np.float64)
    Y_train = Y_train.reshape( len(Y_train), 1); 
    X_test = np.genfromtxt( curpath + testmat, delimiter=',')
    X_test = X_test.astype(np.float64)
    Y_test = np.genfromtxt( curpath + testvec, delimiter=',')
    Y_test = Y_test.astype(np.float64)
    Y_test = Y_test.reshape( len( Y_test ), 1 );
else:
    X_train = np.genfromtxt( curpath + trainmat ,delimiter=',')
    X_train = X_train.astype(np.float64)
    Y_train = np.genfromtxt( curpath + trainvec ,delimiter=',')
    Y_train = Y_train.astype(np.float64)
    Y_train = Y_train.reshape( len(Y_train), 1); 
    X_test = np.loadtxt( curpath + testmat, delimiter=',')
    Y_test = np.loadtxt( curpath + testvec, delimiter=',')
    Y_test = Y_test.reshape( len( Y_test ), 1 );


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

print ()
print ()
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print ()



#for label in y:

#    if (label==0):
#        Y_one_hot.append(np.array([1,0]))
#    elif (label==1):
#        Y_one_hot.append(np.array([0,1]))
#Y_one_hot = np.array(Y_one_hot)



## get BATCH_SIZE data points ##
def get_batch(X,Y):
    idx = np.random.randint(len(X), size=batch_size)
    batch_X= X[idx,:]
    batch_Y = Y[idx]
    return (batch_X,batch_Y)

# Network Parameters
n_input = X_train.shape[1]
n_classes = 2

# Create the network, tf variables and cost function here. 
x = tf.placeholder("float64", [None, n_input])
y = tf.placeholder("float64", [None, n_classes - 1])

W= tf.Variable(tf.zeros([n_input, n_classes-1]))

Matrix_Mul=  tf.matmul(x,W)
pred = tf.sigmoid(tf.matmul(x, W)) # predictions
scores = tf.matmul( x, W );

## to prevent overfitting ###
Zeros = tf.zeros([ tf.shape(x)[0], 1 ],tf.float64)
Matrix_concat= tf.concat([Matrix_Mul, Zeros], 1)
Mx =tf.expand_dims( tf.reduce_max( Matrix_concat, reduction_indices=[1]) ,1)
Ax = tf.add( tf.exp(-Mx), tf.expand_dims( tf.reduce_sum( tf.exp(Matrix_Mul-Mx) ,1),1) ) 

# Minimize error using cross entropy


##  changed to this to prevent overflow
cost = tf.reduce_sum(tf.subtract( Mx+tf.log(Ax), tf.multiply(y, scores)))

## Regularization Term Here. 
regularization = tf.nn.l2_loss(W)


#  Tensorflow built in cost function
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#config = tf.ConfigProto( device_count={'GPU': 0})
if sys.argv[4] == 'GPU': 
   config = tf.ConfigProto( intra_op_parallelism_threads=1 );  
   #config = tf.ConfigProto( ); 
else:
   config = tf.ConfigProto( device_count={'GPU': 0}) 


# Parameters
training_epochs = 100
if sys.argv[3] == 'fixed' : 
    batch_size = 128
else:
    batch_size = int (math.floor( len(X_train) * 0.2 ))

display_step = 1
index = 1



if sys.argv[2].find("raw-data") != -1: 
    #lipschitz constant is 1e-4
    ll = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2] # based on lipschitz constant
    #rterm = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    #rterm = [1e-2]
    rterm = [1e-6]
else:
    #lipschitz constant is 10
    ll = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    #rterm = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    rterm = [1e-7]


for lmethod in [ sys.argv[1] ]:

    for r in rterm: 

            final_cost = cost + r * regularization
        
            for learning_rate in ll: 

                if sys.argv[2].find("raw-data") != -1: 
                	outfile = open("raw_" + lmethod  + "_" + str(index) + "_readings.txt", "w", 0)
                else: 
                	outfile = open("norm_" + lmethod  + "_" + str(index) + "_readings.txt", "w", 0)
                index += 1
                outfile.write("------------------------------------------\n")
                outfile.write("Method: " +  lmethod + "\n")
                outfile.write("Step Length: " +  str(learning_rate) + "\n")
                outfile.write("Regularization: " +  str(r) + "\n")
                outfile.write("Path: " + sys.argv[2]  + "\n")
                outfile.write("BatchSize: " + str(batch_size) + "\n"); 
            
                outfile.write("Begin simulation ...... \n");

                if(lmethod =="GD"):
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(final_cost)
                elif(lmethod =="Adadelta"):
                    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta').minimize(final_cost)
                elif(lmethod =="Adagrad"):
                    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=0.1, use_locking=False, name='Adagrad').minimize(final_cost)
                elif(lmethod =="Adam"):
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(final_cost)
                elif(lmethod =="RMSProp"):
                    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, centered=False, name='RMSProp').minimize(final_cost)

                # Initializing the variables
                init = tf.global_variables_initializer()

                # Launch the graph
                with tf.Session(config=config) as sess:
                #with tf.Session() as sess:
                    sess.run(init)
                    ### thresholding , if >0.5 , TRUE, else FALSE
                    predicted_class = tf.greater(pred,0.5)
                    correct = tf.equal(predicted_class, tf.equal(y,1.0)) 
                    accuracy = tf.reduce_mean( tf.cast(correct, 'float64')) 

                    outfile.write ("%3d:%4.3f:%3.2f:%3.2f:%e:%e\n" % \
                                            ((0), \
                                            (0), \
                                            accuracy.eval({x:X_train, y:Y_train})*100., \
                                            accuracy.eval({x: X_test, y: Y_test})*100., \
                                            sess.run(final_cost, feed_dict={x: X_train, y: Y_train}), \
                                            sess.run(final_cost, feed_dict={x: X_test, y: Y_test})))
                    #import pdb;pdb.set_trace();
                    # Training cycle
                    for epoch in range(training_epochs):
                        avg_cost = 0.
                        total_batch = int(len(X_train)/batch_size)
                        # Loop over all batches

                        start_time = time.time()
                        for i in range(total_batch):
                            batch_x, batch_y = get_batch(X_train,Y_train)
                            # Run optimization op (backprop) and cost op (to get loss value)
                            _, c = sess.run([optimizer,final_cost], feed_dict={x: batch_x,y: batch_y})
                            # Compute average loss
                            avg_cost += c 

                        end_time = time.time ()

                        # Display logs per epoch step
                        if epoch % display_step == 0:
                            # Test model
                            # Calculate accuracy

                    
                            ### thresholding , if >0.5 , TRUE, else FALSE
                            predicted_class = tf.greater(pred,0.5)
                            correct = tf.equal(predicted_class, tf.equal(y,1.0)) 
                            accuracy = tf.reduce_mean( tf.cast(correct, 'float64'))  
                            outfile.write("%3d:%4.3f:%3.2f:%3.2f:%e:%e\n" % \
                                            ((epoch+1), \
                                            (end_time-start_time), \
                                            accuracy.eval({x:X_train, y:Y_train})*100., \
                                            accuracy.eval({x: X_test, y: Y_test})*100., \
                                            sess.run(final_cost, feed_dict={x: X_train, y: Y_train}), \
                                            sess.run(final_cost, feed_dict={x: X_test, y: Y_test})))
                            print ("%3d:%4.3f:%3.2f:%3.2f:%e:%e\n" % \
                                            ((epoch+1), \
                                            (end_time-start_time), \
                                            accuracy.eval({x:X_train, y:Y_train})*100., \
                                            accuracy.eval({x: X_test, y: Y_test})*100., \
                                            sess.run(final_cost, feed_dict={x: X_train, y: Y_train}), \
                                            sess.run(final_cost, feed_dict={x: X_test, y: Y_test})))

                outfile.write("End of Simulation Here..... \n")
                outfile.write("\n");
                outfile.write("\n");
                outfile.write("\n");

                outfile.close ()
