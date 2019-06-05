#This script loads data from pickle files, build a LeNet CNN, and train the model
import logging
logging.getLogger().setLevel(logging.INFO)
import mxnet as mx
import numpy as np
import pickle

#loading data form pickle files
pickle_in = open("X.pickle","rb")
X_train = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y_train = pickle.load(pickle_in)

X_train = np.transpose(X_train, (0, 3, 1, 2))

X_train_set_as_float = X_train.astype('float32')

data = X_train_set_as_float
label = np.array(y_train)
#print(data[1])
#print(label.shape)

#generate training set and validation set
batch_size = 50
ntrain = int(data.shape[0]*0.8)  #0.8 means 20% validation set and 80% training set
train_iter = mx.io.NDArrayIter(data[:ntrain, :], label[:ntrain], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(data[ntrain:, :], label[ntrain:], batch_size)

#first conv layer
net = mx.sym.Variable('data')
net = mx.sym.Convolution(net, kernel=(3,3), num_filter=16, name="conv1")
net = mx.sym.Activation(net, act_type="relu", name= "relu1")
net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2),name="max_pool1")

#second conv layer
net = mx.sym.Convolution(net, kernel=(2,2), num_filter=32, name="conv2")
net = mx.sym.Activation(net, act_type="relu", name="relu2")
net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2),name="max_pool2")

#flatten layer
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
net = mx.sym.Activation(net, name='relu3', act_type="relu")
net = mx.sym.FullyConnected(net, name='fc2', num_hidden=5)
net = mx.sym.SoftmaxOutput(net, name='softmax')

mx.viz.plot_network(net)

mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])

# allocate memory given the input data and label shapes
mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)

# initialize parameters by uniform random numbers
mod.init_params(initializer=mx.init.Uniform(scale=.5))

# use SGD with learning rate 0.1 to train
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.02), ))

# use accuracy as the metric
metric = mx.metric.create('acc')

# reset train_iter to the beginning
train_iter.reset()

# fit the module
model_prefix = 'mynet'
checkpoint = mx.callback.do_checkpoint(model_prefix)
mod.fit(train_iter,
        eval_data=val_iter,
        optimizer='sgd',
        optimizer_params={'learning_rate':0.02},
        eval_metric='acc',
        epoch_end_callback=checkpoint,
        num_epoch=10)
