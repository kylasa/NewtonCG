from __future__ import print_function

import sys
import math
import tensorflow as tf
import numpy as np

import scipy.sparse as sparse

import cPickle as pickle
import time
import StringIO

def getShape( entries ):
    mrow = 0
    mcol = 0
    for item in entries:
        if mrow < item[0]:
            mrow = item[0]
        if mcol < item[1]:
            mcol = item[1]
    return mrow, mcol

def getDenseMatrix(entries, rows, cols):
    rowIdx = np.empty([len(entries)], dtype=int)
    colIdx = np.empty([len(entries)], dtype=int)
    val = np.empty([len(entries)], dtype=np.float64)

    print( rows ); 
    print( cols ); 

    for idx,item in enumerate(entries):
        rowIdx[ idx ] = item[0] - 1
        colIdx[ idx ] = item[1] - 1
        val[ idx ] = item[2]
    return sparse.csr_matrix( (val, (rowIdx, colIdx)), shape=(rows, cols) ).toarray ()

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
X_train = np.genfromtxt( curpath + trainmat ,delimiter=',', dtype=None)
Y_train = np.genfromtxt( curpath + trainvec ,delimiter=',', dtype=None)
Y_train = Y_train.astype(np.float64)
Y_train = Y_train.reshape( len(Y_train), 1); 

# Test
X_test = np.genfromtxt( curpath + testmat, delimiter=',', dtype=None)
Y_test = np.genfromtxt( curpath + testvec, delimiter=',', dtype=None)
Y_test = Y_test.astype(np.float64)
Y_test = Y_test.reshape( len( Y_test ), 1 ); 

# Convert to the usable format here. 

x_row, x_col = getShape( X_train )
y_row, y_col = getShape( X_test )

shapey = 0

if x_col < y_col:
    shapey = y_col
else:
    shapey = x_col


X_train = getDenseMatrix( X_train, x_row, shapey )
X_test = getDenseMatrix( X_test, y_row, shapey )

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

## shuffle data points ##
def shuffle(X,Y):
    idx = np.random.randint(len(X), size=len(X))
    X = X[idx,:]
    Y = Y[idx]
    return (X,Y)

X_train,Y_train=shuffle(X_train,Y_train)



# Network Parameters
n_input = X_train.shape[1]
n_classes = 2

# Create the network, tf variables and cost function here. 
#x = tf.placeholder("float", [None, n_input])
#y = tf.placeholder("float", [None, n_classes - 1])

x=tf.sparse_placeholder(tf.float64)
y=tf.sparse_placeholder(tf.float64)




W= tf.Variable(tf.zeros([n_input, n_classes-1], dtype=tf.float64), dtype=tf.float64)


Matrix_Mul= tf.sparse_tensor_dense_matmul(x,W)
pred = tf.sigmoid(tf.sparse_tensor_dense_matmul(x, W)) # predictions
scores = tf.sparse_tensor_dense_matmul( x, W );

#Matrix_Mul=  tf.matmul(x,W)
#pred = tf.sigmoid(tf.matmul(x, W)) # predictions
#scores = tf.matmul( x, W );

## to prevent overfitting ###
Zeros = tf.zeros([ tf.shape(x)[0], 1 ],tf.float64)
Matrix_concat= tf.concat([Matrix_Mul, Zeros], 1)
Mx =tf.expand_dims( tf.reduce_max( Matrix_concat, reduction_indices=[1]) ,1)
Ax = tf.add( tf.exp(-Mx), tf.expand_dims( tf.reduce_sum( tf.exp(Matrix_Mul-Mx) ,1),1) ) 

# Minimize error using cross entropy


##  changed to this to prevent overflow
cost = tf.reduce_sum(tf.subtract( Mx+tf.log(Ax), tf.multiply(tf.sparse_tensor_to_dense(y), scores)))

#cost = tf.reduce_sum ( (Mx+tf.log(Ax)) - tf.expand_dims( sel_ele_2d( Matrix_concat , tf.argmax(tf.sparse_tensor_to_dense(y),1) ),1) )
## Regularization Term Here. 
regularization = tf.nn.l2_loss(W)


#  Tensorflow built in cost function
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
prefix = ''
if sys.argv[4] == 'GPU': 
   config = tf.ConfigProto( intra_op_parallelism_threads=1)
   config.gpu_options.allow_growth = True
   prefix += 'GPU'
else:
   config = tf.ConfigProto(device_count={'GPU': 0} )
   prefix += 'CPU'


# Parameters
training_epochs = 100
if sys.argv[3] == 'fixed' : 
    batch_size = 128
else:
    batch_size = int (math.floor( len(X_train) * 0.2 ))

display_step = 1
index = 1


with tf.Session(config=config) as sess:

    x_t= tf.placeholder("float64", [None, n_input])
    #x_t = tf.constant(X_test)
    idx_x = tf.where(tf.not_equal(x_t, 0))
    sparse_Test_x = tf.SparseTensor(idx_x, tf.gather_nd(x_t, idx_x), tf.cast(tf.shape(x_t),tf.int64))

    #sparse_X_test=sess.run([sparse_Test_x],feed_dict={x_t:X_test})
    #import pdb;pdb.set_trace();
    y_t= tf.placeholder("float64", [None, n_classes-1])

    #y_t = tf.constant(Y_test)
    idx_y = tf.where(tf.not_equal(y_t, 0))
    sparse_Test_y = tf.SparseTensor(idx_y, tf.gather_nd(y_t, idx_y),tf.cast(tf.shape(y_t),tf.int64))
    ### creating batch list ###
    x_t2 =   tf.placeholder("float64", [None, n_input])  #tf.constant(X_train)
    idx_x = tf.where(tf.not_equal(x_t2, 0))
    sparse_Train_x = tf.SparseTensor(idx_x, tf.gather_nd(x_t2, idx_x), tf.cast(tf.shape(x_t2),tf.int64) )
    sparse_Train_x_list=tf.sparse_split(sp_input=sparse_Train_x,axis=0,num_split=int (np.floor( len(X_train)/batch_size )))
    y_t2 =  tf.placeholder("float64", [None, n_classes-1])   #tf.constant(Y_train)
    idx_y = tf.where(tf.not_equal(y_t2, 0))
    sparse_Train_y = tf.SparseTensor(idx_y, tf.gather_nd(y_t2, idx_y), tf.cast(tf.shape(y_t2),tf.int64)) 
    sparse_Train_y_list=tf.sparse_split(sp_input=sparse_Train_y,axis=0,num_split=int (np.floor( len(X_train)/batch_size )))

    X_test,Y_test,X_train,Y_train,batch_x_list,batch_y_list = sess.run([sparse_Test_x,sparse_Test_y,sparse_Train_x,sparse_Train_y,sparse_Train_x_list,sparse_Train_y_list ],feed_dict={x_t: X_test, y_t: Y_test,x_t2: X_train,y_t2: Y_train})    

#import pdb; pdb.set_trace();




if sys.argv[2].find("raw-data") != -1:
    #lipschitz constant is 1e-12
    # this dataset is normalized from the source
    #no need to run the raw-data set of runs. 
    #ll = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    #rterm = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
#	 print 'This is NOT DEFINED FOR THIS DATASET... '
	 exit ()
else:
    #lipschitz constant is 1e-3
    ll = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    #rterm = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    #rterm = [1e-2]
    rterm = [1e-3]
    prefix += '_norm_'


for lmethod in [ sys.argv[1] ]:

    for r in rterm: 

            final_cost = cost + r * regularization
        
            for learning_rate in ll: 

                outfile = open(prefix + lmethod  + "_" + str(index) + "_readings.txt", "w", 0)
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
                elif(lmethod =="Momentum"):
                    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_locking=False, name='Momentum', use_nesterov=False).minimize(final_cost)

                # Initializing the variables
                init = tf.global_variables_initializer()

                # Launch the graph
                with tf.Session(config=config) as sess:
                #with tf.Session() as sess:
                    sess.run(init)
                    ### thresholding , if >0.5 , TRUE, else FALSE
                    predicted_class = tf.greater(pred,0.5)
                    correct = tf.equal(predicted_class, tf.equal(tf.sparse_tensor_to_dense(y),1.0)) 
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
                        #total_batch = int(len(X_train)/batch_size)
                        # Loop over all batches

                        start_time = time.time()
                        for i in range(len(batch_x_list )):
                            #batch_x, batch_y = get_batch(X_train,Y_train)
                            # Run optimization op (backprop) and cost op (to get loss value)
                            _, c = sess.run([optimizer,final_cost], feed_dict={x: batch_x_list[i],y: batch_y_list[i]})
                            # Compute average loss
                            avg_cost += c 

                        end_time = time.time ()

                        # Display logs per epoch step
                        if epoch % display_step == 0:
                            # Test model
                            # Calculate accuracy

                    
                            ### thresholding , if >0.5 , TRUE, else FALSE
                            predicted_class = tf.greater(pred,0.5)
                            correct = tf.equal(predicted_class, tf.equal(tf.sparse_tensor_to_dense(y),1.0)) 
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
