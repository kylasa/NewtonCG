from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np

import cPickle as pickle
import time
import StringIO

import scipy.sparse as sparse
import math

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

    for idx,item in enumerate(entries): 
        rowIdx[ idx ] = item[0] - 1
        colIdx[ idx ] = item[1] - 1
        val[ idx ] = item[2]

    return sparse.csr_matrix( (val, (rowIdx, colIdx)), shape=(rows, cols) ).toarray ()
        

trainmat = 'train_mat.txt'
trainvec = 'train_vec.txt'
testmat = 'test_mat.txt'
testvec = 'test_vec.txt'


#load the data here. 
curpath = sys.argv[2]

print ()
print (curpath)
print ()
print ()


X_train = np.loadtxt(curpath + trainmat ,delimiter=',')
X_train = X_train.astype(np.float64)
Y_train = np.loadtxt(curpath + trainvec ,delimiter=',')
Y_train = Y_train.astype(np.float64)

# Test
X_test = np.loadtxt(curpath + testmat, delimiter=',')
X_test = X_test.astype(np.float64)
Y_test = np.loadtxt(curpath + testvec, delimiter=',')
Y_test = Y_test.astype(np.float64)

# Convert to the usable format here. 

x_row, x_col = getShape( X_train )
y_row, y_col = getShape( X_test )

shapey = 0

if x_col < y_col: 
    shapey = y_col
else: 
    shapey = x_col

print ("Done loading data..... ")
print (X_train.shape)
print (Y_train.shape)
print (X_test.shape)
print (Y_test.shape)
print (Y_test)
print (Y_train)


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

Y_one_hot=[]
for l in Y_train: 
    if (l == 1): 
        Y_one_hot.append(np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==2):
        Y_one_hot.append(np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==3):
        Y_one_hot.append(np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==4):
        Y_one_hot.append(np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==5):
        Y_one_hot.append(np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==6):
        Y_one_hot.append(np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==7):
        Y_one_hot.append(np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==8):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==9):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==10):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]))
    if (l == 11): 
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]))
    elif (l ==12):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]))
    elif (l ==13):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]))
    elif (l ==14):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]))
    elif (l ==15):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]))
    elif (l ==16):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]))
    elif (l ==17):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]))
    elif (l ==18):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]))
    elif (l ==19):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]))
    elif (l ==20):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]))
Y_train = np.array(Y_one_hot)

Y_one_hot=[]
for l in Y_test: 
    if (l == 1): 
        Y_one_hot.append(np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==2):
        Y_one_hot.append(np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==3):
        Y_one_hot.append(np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==4):
        Y_one_hot.append(np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==5):
        Y_one_hot.append(np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==6):
        Y_one_hot.append(np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==7):
        Y_one_hot.append(np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==8):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==9):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]))
    elif (l ==10):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]))
    if (l == 11): 
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]))
    elif (l ==12):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]))
    elif (l ==13):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]))
    elif (l ==14):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]))
    elif (l ==15):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]))
    elif (l ==16):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]))
    elif (l ==17):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]))
    elif (l ==18):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]))
    elif (l ==19):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]))
    elif (l ==20):
        Y_one_hot.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]))
Y_test = np.array(Y_one_hot)



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

X_train = getDenseMatrix( X_train, x_row, shapey )
X_test = getDenseMatrix( X_test, y_row, shapey )

X_train,Y_train=shuffle(X_train,Y_train)

#import pdb; pdb.set_trace();
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
#x = tf.placeholder("float", [None, n_input])
#y = tf.placeholder("float", [None, n_classes])

x=tf.sparse_placeholder(tf.float64)
y=tf.sparse_placeholder(tf.float64)
W= tf.Variable(tf.zeros([n_input, n_classes-1], dtype=tf.float64), dtype=tf.float64)

#Matrix_Mul=  tf.matmul(x,W)
Matrix_Mul= tf.sparse_tensor_dense_matmul(x,W)
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
cost = tf.reduce_sum ( (Mx+tf.log(Ax)) - tf.expand_dims( sel_ele_2d( Matrix_concat , tf.argmax(tf.sparse_tensor_to_dense(y),1) ),1) )

## Regularization Term Here. 
#regularization = (float(sys.argv[2]) / 2.0) * tf.pow(tf.norm( W, ord='euclidean' ), 2.)
#regularization = tf.pow(tf.norm( W, ord='euclidean' ), 2.)
regularization = tf.nn.l2_loss(W)


#  Tensorflow built in cost function
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

prefix = ''
if sys.argv[4] == 'GPU': 
   config = tf.ConfigProto( intra_op_parallelism_threads=1)
   prefix += 'GPU'
else:
   config = tf.ConfigProto(device_count={'GPU': 0} )
   prefix += 'CPU'


# Parameters
training_epochs = 100
display_step = 1
index = 0

# Parameters
if sys.argv[3] == 'fixed' :
   batch_size = 128 
else:
   batch_size = int (math.floor( len(X_train) * 0.2 ))


with tf.Session() as sess:
    ### creating sparse_tensor for test x, test y ###

    x_t= tf.placeholder("float64", [None, n_input])
    #x_t = tf.constant(X_test)
    idx_x = tf.where(tf.not_equal(x_t, 0))
    sparse_Test_x = tf.SparseTensor(idx_x, tf.gather_nd(x_t, idx_x), tf.cast(tf.shape(x_t),tf.int64))

    #sparse_X_test=sess.run([sparse_Test_x],feed_dict={x_t:X_test})
    #import pdb;pdb.set_trace();
    y_t= tf.placeholder("float64", [None, n_classes])

    #y_t = tf.constant(Y_test)
    idx_y = tf.where(tf.not_equal(y_t, 0))
    sparse_Test_y = tf.SparseTensor(idx_y, tf.gather_nd(y_t, idx_y),tf.cast(tf.shape(y_t),tf.int64))
    ### creating batch list ###
    x_t2 =   tf.placeholder("float64", [None, n_input])  #tf.constant(X_train)
    idx_x = tf.where(tf.not_equal(x_t2, 0))
    sparse_Train_x = tf.SparseTensor(idx_x, tf.gather_nd(x_t2, idx_x), tf.cast(tf.shape(x_t2),tf.int64) )
    sparse_Train_x_list=tf.sparse_split(sp_input=sparse_Train_x,axis=0,num_split=int (np.floor( len(X_train)/batch_size )))
    y_t2 =  tf.placeholder("float64", [None, n_classes])   #tf.constant(Y_train)
    idx_y = tf.where(tf.not_equal(y_t2, 0))
    sparse_Train_y = tf.SparseTensor(idx_y, tf.gather_nd(y_t2, idx_y), tf.cast(tf.shape(y_t2),tf.int64)) 
    sparse_Train_y_list=tf.sparse_split(sp_input=sparse_Train_y,axis=0,num_split=int (np.floor( len(X_train)/batch_size )))

    X_test,Y_test,X_train,Y_train,batch_x_list,batch_y_list = sess.run([sparse_Test_x,sparse_Test_y,sparse_Train_x,sparse_Train_y,sparse_Train_x_list,sparse_Train_y_list ],feed_dict={x_t: X_test, y_t: Y_test,x_t2: X_train,y_t2: Y_train})    


if sys.argv[2].find("raw-data") != -1:
	#raw
	ll = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2 ]
	#rlist = [1e-3]
	rlist = [1]
	prefix += '_raw_'
else: 
	#normalized
	ll = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2 ]
	#rlist = [1e-1]
	rlist = [1e-3]
	prefix += '_norm_'
	#print ("This is NOT Working at the moment..... ")
	#exit ()	

for lmethod in [ sys.argv[1] ]:
    for r in rlist:
            final_cost = cost + r * regularization
        
            for learning_rate in ll: 

                outfile = open(prefix + lmethod  +"_" + str(index) + "_readings.txt", "w", 0)
                index += 1

                outfile.write("------------------------------------------\n")
                outfile.write("Method: " +  lmethod + "\n")
                outfile.write("Step Length: " +  str(learning_rate) + "\n")
                outfile.write("Regularization: " +  str(r) + "\n")
                outfile.write("Normalization: " + sys.argv[2] +"\n")
                outfile.write("Batch Size: "+ str(batch_size) +"\n")
            
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
                            correct_prediction = tf.equal( tf.argmax(pred,1), tf.argmax(tf.sparse_tensor_to_dense(y),1) )
                            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float64"))
                            outfile.write("%3d:%4.2f:%3.2f:%3.2f:%e:%e\n" % \
                                            ((0), \
                                            (0), \
                                            accuracy.eval({x:X_train, y:Y_train})*100., \
                                            accuracy.eval({x: X_test, y: Y_test})*100., \
                                            sess.run(final_cost, feed_dict={x: X_train, y: Y_train}), \
                                            sess.run(final_cost, feed_dict={x: X_test, y: Y_test})))

                            print ("%3d:%4.2f:%3.2f:%3.2f:%e:%e\n" % \
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
                        for i in range(len(batch_x_list)):
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
                            correct_prediction = tf.equal( tf.argmax(pred,1), tf.argmax(tf.sparse_tensor_to_dense(y),1) )
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
