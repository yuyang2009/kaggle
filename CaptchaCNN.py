import matplotlib.pyplot as plt
import numpy as np
import random
import os
from captcha.image import ImageCaptcha
import glob
import string
from PIL import Image
import tensorflow as tf



"""
This class does some initial training of a neural network for predicting drawn
digits based on a data set in data_matrix and data_labels. It can then be used to
train the network further by calling train() with any array of data or to predict
what a drawn digit is by calling predict().

The weights that define the neural network can be saved to a file, NN_FILE_PATH,
to be reloaded upon initilization.
"""
class CaptchaCNN:
    width_images = 100
    height_images = 40
    num_charmap = 36
    num_channel = 1
    num_char = 4
    char_map = string.ascii_uppercase + string.digits
    model_path = './model/model.ckpt'

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, CaptchaCNN.height_images, CaptchaCNN.width_images])
        self.y_ = tf.placeholder(tf.float32, [None, CaptchaCNN.num_char, CaptchaCNN.num_charmap])

        #todo: find out why reshape doesn.t work
        self.x_image = tf.reshape(self.x, [-1, CaptchaCNN.height_images, CaptchaCNN.width_images, 1])

        #Dropout
        self.keep_prob = tf.placeholder(tf.float32)

        #First Convolutional Layer
        self.W_conv1 = self._weight_variable([5, 5, 1, 32])
        self.b_conv1 = self._bias_variable([32])
        self.h_conv1 = tf.nn.relu(self._conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = self._max_pool_2x2(self.h_conv1)

        #Second Convolutional Layer
        self.W_conv2 = self._weight_variable([5, 5, 32, 64])
        self.b_conv2 = self._bias_variable([64])
        self.h_conv2 = tf.nn.relu(self._conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self._max_pool_2x2(self.h_conv2)

        #Third Convolutional Layer
        # W_conv3 = self._weight_variable([5, 5, 64, 128])
        # b_conv3 = self._bias_variable([128])
        # h_conv3 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
        # h_pool3 = self._max_pool_2x2(h_conv2)

        #First densely Connected Layer
        self.W_fc1 = self._weight_variable([10 * 25 * 64, 1024])
        self.b_fc1 = self._bias_variable([1024])
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 10*25*64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        #Readout Layer
        self.W_fc2 = self._weight_variable([1024, 144])
        self.b_fc2 = self._bias_variable([144])

        self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

        self.y_conv = tf.reshape(self.y_conv, [-1, CaptchaCNN.num_char, CaptchaCNN.num_charmap])

        self.cross_entropy = tf.reduce_mean(tf.concat(0, [tf.nn.softmax_cross_entropy_with_logits(
                self.y_conv[:, _i, :], self.y_[:, _i, :]) for _i in range(CaptchaCNN.num_char) ]))

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.max_idx_p = tf.argmax(self.y_conv, 2)
        self.max_idx_l = tf.argmax(self.y_, 2)
        self.correct_prediction = tf.cast(tf.equal(self.max_idx_p, self.max_idx_l), tf.float32)
        self.accuracy = tf.reduce_mean(tf.reduce_min(self.correct_prediction, axis=1))

        self.saver = tf.train.Saver()
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.config.gpu_options.allow_growth = True


    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def img_preprocess(self, img):
        img_resize = np.array(img.resize((CaptchaCNN.width_images, CaptchaCNN.height_images)))
        img_gray = np.mean(img_resize, -1)
    #     img_gray = np.multiply(img_resize, [0.2989, 0.5870, 0.1140])
        img_scale = np.multiply(img_gray, 1/255.0)
        return img_scale

    def dataset_generator(self, num_images=10000):
    #     image_generator = ImageCaptcha(fonts=['./font/AntykwaBold.ttf'])
        image_generator = ImageCaptcha(fonts=['./font/AntykwaBold.ttf'])
        # X = np.empty((num_images, CaptchaCNN.height_images, CaptchaCNN.width_images))
        # y = np.empty((num_images, CaptchaCNN.num_charmap * CaptchaCNN.num_char))
        # 未初始化，内存泄漏
        X = np.zeros((num_images, CaptchaCNN.height_images, CaptchaCNN.width_images))
        y = np.zeros((num_images, CaptchaCNN.num_charmap * CaptchaCNN.num_char))
        for i in range(num_images):
            text = ''.join(random.sample(CaptchaCNN.char_map, CaptchaCNN.num_char))
            img = image_generator.generate_image(text)
            X[i, :, :] = self.img_preprocess(img)
            y_index = [ CaptchaCNN.char_map.find(text[_i])+_i*CaptchaCNN.num_charmap for _i in range(CaptchaCNN.num_char)]
            y[i, y_index] = 1

            #灰度值求解
            #np.sum(np.array([0.2989, 0.5870, 0.1140]) * img, axis=2)
            # if (i+1) % (num_images/5) == 0:
            #     print("generating captcha " , i)
        return X, y

    def get_next_batch(self, X, y, batch_size=128):
        index_in_epoch = 0
        num_images = X.shape[0]
    #     assert num_images <= X.shape[0]
        while True:
            start = index_in_epoch
            index_in_epoch += batch_size
            if index_in_epoch > num_images:
                perm = np.arange(num_images)
                np.random.shuffle(perm)
                X = X[perm]
                y = y[perm]
                start = 0
                index_in_epoch = batch_size
                assert batch_size <= num_images
            end = index_in_epoch
            yield X[start:end], y[start:end]

    def train(self, X_train, y_train):
        train_batch = self.get_next_batch(X_train, y_train, batch_size= 128)

        with tf.Session(config=self.config) as sess:
            self._load(sess)
            while True:
                X_batch, y_batch = next(train_batch)
                y_batch = y_batch.reshape(-1, CaptchaCNN.num_char, CaptchaCNN.num_charmap)
                if (step+1)%500 == 0:
                    train_accuracy = accuracy.eval(feed_dict={self.x:X_batch, self.y_: y_batch, self.keep_prob: 1.0})
                    print(time.ctime() ,": step %d, training accuracy %g"%(step, train_accuracy))
                    if train_accuracy >= 0.95:
                        print("* "*40)
                        print(time.ctime() ,"CNN trianing done, total %d steps, training accuracy %g"%(step, train_accuracy))
                        break
                step += 1
                sess.run(train_step, feed_dict={x: X_batch, y_: y_batch, keep_prob: 0.5})

    def predict(self, X):
        with tf.Session(config=self.config) as sess:
            self._load(sess)
            pred = self.max_idx_p.eval(feed_dict={self.x: X, self.keep_prob: 1.0})
            pred_char_lst = [''.join([CaptchaCNN.char_map[_i] for _i in pred_ind]) for pred_ind in pred]
            return pred_char_lst

    def _save(self, sess):
        self.saver.save(sess, CaptchaCNN.model_path)

    def _load(self, sess):
        if not os.path.exists("./model"):
            sess.run(tf.global_variables_initializer())
            print("No model used!!!")
        else:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, CaptchaCNN.model_path)

if __name__ == '__main__':
    captcha_cnn = CaptchaCNN()
    X_test, y_test = captcha_cnn.dataset_generator(10)
    y_test = y_test.reshape(-1, captcha_cnn.num_char, captcha_cnn.num_charmap)
    pred = captcha_cnn.predict(X_test)
    print(pred)
    y_ind = np.argmax(y_test, axis=2)
    text = [''.join([CaptchaCNN.char_map[_i] for _i in _ind]) for _ind in y_ind]
    print(text)
    for i in range(10):
        plt.imsave('./img/'+str(i)+'.jpg', X_test[i], cmap='gray')


