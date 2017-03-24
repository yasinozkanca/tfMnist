from tensorflow.examples.tutorials.mnist import input_data
from scipy import ndimage
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf
import numpy as np
test = mnist.test
labels = test.labels

img = test.images
train_images = img[0:9000]
train_labels = labels[0:9000]
test_images = img[9000:10000]
test_labels = labels[9000:10000]

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 196])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx/fact, sy/fact)
    return res

def downSampleImages(trainImages,labels):
    imgArray = np.ndarray(shape=(28, 28))
    downSampled = np.ndarray(shape=(len(trainImages), 14*14))

    for k in range(len(trainImages)):

        imgArray = trainImages[k].reshape((28, 28))
        imgArray = block_mean(imgArray, 2)

        imgArray = imgArray.reshape((1, 196))

        downSampled[k, :] = imgArray

    return downSampled

def augmentImages(downSampled):
    rotated = np.ndarray(shape=(len(downSampled), 14*14))
    zoomed = np.ndarray(shape=(len(downSampled), 14 * 14))
    shifted = np.ndarray(shape=(len(downSampled), 14 * 14))

    for i in range(len(downSampled)):
        tempImg = downSampled[i]
        tempImg = tempImg.reshape([14, 14])

        temp1Img = ndimage.interpolation.rotate(tempImg, 20)
        rotated[i, :] = temp1Img
        temp2Img = ndimage.interpolation.zoom(tempImg, 0.75)
        zoomed[i, :] = temp2Img
        temp3Img = ndimage.interpolation.shift(tempImg, [5, 3])
        shifted[i,:] = temp3Img

    rotated = rotated.reshape((1, 196))
    zoomed = zoomed.reshape((1, 196))
    shifted = shifted.reshape((1, 196))

    return shifted,zoomed,rotated

def incTrainSamples(trainImages,labels):
    downSampled = downSampleImages(trainImages,labels)
    #trainImages = np.concatenate((trainImages, downSampled))
    #labels = np.concatenate((labels, downSampled))
    newBatch = (downSampled, labels)
    return newBatch

def batchGetter(start,finish):
    trainImages = mnist.test.images[start:finish]
    trainLabels = mnist.test.labels[start:finish]
    return trainImages,trainLabels

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x,strides):
  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                        strides=strides, padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,14,14,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

h_pool1 = max_pool_2x2(h_conv1,[1,2,2,1])

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2,[1,1,1,1])

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())


for k in range(12):
    for i in range(180):
        batch_image,batch_labels = batchGetter(i*50,(i+1)*50)
        batch = incTrainSamples(batch_image, batch_labels)

        if i % 90 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
downSampledTest = downSampleImages(mnist.test.images[9000:10000], mnist.test.labels[9000:10000])
print("test accuracy %g" % accuracy.eval(feed_dict={
    x: downSampledTest, y_: mnist.test.labels[9000:10000], keep_prob: 1.0}))
