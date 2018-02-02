from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np
import math

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

#load the data here. 
X_train = np.loadtxt(curpath + trainmat ,delimiter=',')
X_train = X_train.astype(np.float64)
Y_train = np.loadtxt(curpath + trainvec ,delimiter=',')
Y_train = Y_train.astype(np.float64)

# Test
X_test = np.loadtxt(curpath + testmat, delimiter=',')
X_test = X_test.astype(np.float64)
Y_test = np.loadtxt(curpath + testvec, delimiter=',')
Y_teset = Y_test.astype(np.float64)

print ("Done loading data..... ")


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

Y_one_hot=[]
for l in Y_train: 
    if (l == 1): 
        Y_one_hot.append(np.array([1,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==2):
        Y_one_hot.append(np.array([0,1,0,0,0,0,0,0,0,0,0]))
    elif (l ==3):
        Y_one_hot.append(np.array([0,0,1,0,0,0,0,0,0,0,0]))
    elif (l ==4):
        Y_one_hot.append(np.array([0,0,0,1,0,0,0,0,0,0,0]))
    elif (l ==5):
        Y_one_hot.append(np.array([0,0,0,0,1,0,0,0,0,0,0]))
    elif (l ==6):
        Y_one_hot.append(np.array([0,0,0,0,0,1,0,0,0,0,0]))
    elif (l ==7):
        Y_one_hot.append(np.array([0,0,0,0,0,0,1,0,0,0,0]))
    elif (l ==8):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,1,0,0,0]))
    elif (l ==9):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,1,0,0]))
    elif (l ==10):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,1,0]))
    elif (l ==11):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,1]))
Y_train = np.array(Y_one_hot)

Y_one_hot=[]
for l in Y_test: 
    if (l == 1): 
        Y_one_hot.append(np.array([1,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==2):
        Y_one_hot.append(np.array([0,1,0,0,0,0,0,0,0,0,0]))
    elif (l ==3):
        Y_one_hot.append(np.array([0,0,1,0,0,0,0,0,0,0,0]))
    elif (l ==4):
        Y_one_hot.append(np.array([0,0,0,1,0,0,0,0,0,0,0]))
    elif (l ==5):
        Y_one_hot.append(np.array([0,0,0,0,1,0,0,0,0,0,0]))
    elif (l ==6):
        Y_one_hot.append(np.array([0,0,0,0,0,1,0,0,0,0,0]))
    elif (l ==7):
        Y_one_hot.append(np.array([0,0,0,0,0,0,1,0,0,0,0]))
    elif (l ==8):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,1,0,0,0]))
    elif (l ==9):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,1,0,0]))
    elif (l ==10):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,1,0]))
    elif (l ==11):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,1]))

Y_test = np.array(Y_one_hot)



## get BATCH_SIZE data points ##
def get_batch(X,Y):
    idx = np.random.randint(len(X), size=batch_size)
    batch_X= X[idx,:]
    batch_Y = Y[idx]
    return (batch_X,batch_Y)

# Network Parameters
n_input = X_train.shape[1]
n_classes = len( Y_train[0] )

## select specific element by index for each row
def sel_ele_2d(a,b):
    b= tf.cast(b, tf.int32)
    b_2 = tf.expand_dims(b, 1)
    the_range = tf.expand_dims(tf.range(tf.shape(b)[0]), 1)
    ind = tf.concat([the_range, b_2],1)
    res = tf.gather_nd(a, ind)
    return res


# Create the network, tf variables and cost function here. 
x = tf.placeholder("float64", [None, n_input])
y = tf.placeholder("float64", [None, n_classes])

#W= tf.Variable(tf.random_normal([n_input, n_classes-1]))
W= tf.Variable(tf.zeros([n_input, n_classes-1], dtype=tf.float64), dtype=tf.float64)

Matrix_Mul=  tf.matmul(x,W)
Zeros = tf.zeros([ tf.shape(x)[0], 1 ],tf.float64)
Matrix_concat= tf.concat([Matrix_Mul, Zeros], 1)
Mx =tf.expand_dims( tf.reduce_max( Matrix_concat, reduction_indices=[1]) ,1)
Ax = tf.add( tf.exp(-Mx), tf.expand_dims( tf.reduce_sum( tf.exp(Matrix_Mul-Mx) ,1),1) )

T= tf.one_hot((n_classes-1)*tf.ones([tf.shape(x)[0]],tf.int64) ,depth=n_classes,on_value=np.float64(0.0),off_value=np.float64(1.0),dtype=tf.float64)

## (y==c)*e^<x,vc>
pre_temp= tf.multiply(T,tf.exp(Matrix_concat))

## 1+sigma(e^<x,cb>)
pre_temp1 = tf.add( np.float64(1.0),tf.expand_dims( tf.reduce_sum( tf.exp(Matrix_Mul),1 ),1) )

## last-th class prob##
pre_temp2 = np.float64(1.0) - tf.reduce_sum(tf.div(pre_temp,pre_temp1),1)

## get the first 6 classes prob
pre_temp3 = tf.slice( tf.div(pre_temp,pre_temp1) ,[0,0],[tf.shape(x)[0],n_classes-1])

##  concat 6 classes prob with last class prob
pred = tf.concat([pre_temp3,tf.expand_dims(pre_temp2,1)],1)
pred_labels_tf = tf.argmax(pred,1);


## Our cost function
cost = tf.reduce_sum ( (Mx+tf.log(Ax)) - tf.expand_dims( sel_ele_2d( Matrix_concat , tf.argmax(y,1) ),1) )

## Regularization Term Here. 
#regularization = (float(sys.argv[2]) / 2.0) * tf.pow(tf.norm( W, ord='euclidean' ), 2.)
#regularization = tf.pow(tf.norm( W, ord='euclidean' ), 2.)
regularization = tf.nn.l2_loss(W)


#  Tensorflow built in cost function
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

prefix = ''

if sys.argv[4] == 'GPU': 
   config = tf.ConfigProto( intra_op_parallelism_threads=1 );  
   prefix = 'GPU'
   #config = tf.ConfigProto( ); 
else:
   config = tf.ConfigProto( device_count={'GPU': 0}) 
   prefix = 'CPU'


# Parameters
training_epochs = 100
display_step = 1
index = 0

if sys.argv[3] == 'fixed' :
   batch_size = 128 
else:
   batch_size = int (math.floor( len(X_train) * 0.2 ))


if sys.argv[2].find("raw-data") != -1: 
   #lipschitz constant is 1e-7
	ll = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
	#rlist = [1e-4]
	rlist = [1]
	prefix += '_raw_'
else:
   #lipschitz constant is 1e-1
	ll = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
	#rlist = [1e-7]
	#rlist = [1e-6]
	rlist = [1e-3]
	prefix += '_norm_'


for lmethod in [ sys.argv[1] ]:
    for r in rlist:

            final_cost = cost + r * regularization
        
            for learning_rate in ll: 

                outfile = open(prefix + lmethod + "_" + str(index) +  "_readings.txt", "w", 0)
                index += 1

                outfile.write("------------------------------------------\n")
                outfile.write("Method: " +  lmethod + "\n")
                outfile.write("Step Length: " +  str(learning_rate) + "\n")
                outfile.write("Regularization: " +  str(r) + "\n")
                outfile.write("Normalization: " + sys.argv[2] + "\n")
                outfile.write("Batch Size: "+ str( batch_size ) + "\n") 
            
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
                elif(lmethod =="Momentum"):
                    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_locking=False, name='Momentum', use_nesterov=False).minimize(final_cost)

                # Initializing the variables
                init = tf.global_variables_initializer()

                # Launch the graph
                with tf.Session(config=config) as sess:
                #with tf.Session() as sess:
                    sess.run(init)

                    if True:
                            correct_prediction = tf.equal( tf.argmax(pred,1), tf.argmax(y,1) )
                            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float64"))
                            outfile.write("%3d:%4.2f:%3.2f:%3.2f:%e:%e\n" % \
                                            ((0), \
                                            (0), \
                                            accuracy.eval({x:X_train, y:Y_train})*100., \
                                            accuracy.eval({x: X_test, y: Y_test})*100., \
                                            sess.run(final_cost, feed_dict={x: X_train, y: Y_train}), \
                                            sess.run(final_cost, feed_dict={x: X_test, y: Y_test})))

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
                            correct_prediction = tf.equal( tf.argmax(pred,1), tf.argmax(y,1) )
                            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float64"))
                            #", cost=","{:.9f}".format(avg_cost), \
                            outfile.write("%3d:%4.2f:%3.2f:%3.2f:%e:%e\n" % \
                                            ((epoch+1), \
                                            (end_time-start_time), \
                                            accuracy.eval({x:X_train, y:Y_train})*100., \
                                            accuracy.eval({x: X_test, y: Y_test})*100., \
                                            sess.run(final_cost, feed_dict={x: X_train, y: Y_train}), \
                                            sess.run(final_cost, feed_dict={x: X_test, y: Y_test})))
                            print ("%3d:%4.2f:%3.2f:%3.2f:%e:%e\n" % \
                                            ((epoch+1), \
                                            (end_time-start_time), \
                                            accuracy.eval({x:X_train, y:Y_train})*100., \
                                            accuracy.eval({x: X_test, y: Y_test})*100., \
                                            sess.run(final_cost, feed_dict={x: X_train, y: Y_train}), \
                                            sess.run(final_cost, feed_dict={x: X_test, y: Y_test})))

                            #pred_labels = sess.run( pred_labels_tf, feed_dict={x: X_test, y: Y_test} )

                            #c = [0, 0, 0, 0, 0, 0, 0]
                            #for i in range(0, len(pred_labels)): 
                            #    c[ pred_labels[i] ] += 1

                            #for i in range(0, len(c)): 
                            #    print ("Class: %d --- > %d" % (i, c[ i ]) )

                outfile.write("End of Simulation Here..... \n")
                outfile.write("\n");
                outfile.write("\n");
                outfile.write("\n");

                outfile.close ()
