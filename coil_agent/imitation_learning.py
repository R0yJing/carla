from __future__ import print_function

import os
from scipy import misc
import scipy
from PIL import Image
import tensorflow as tf
import numpy as np
import time
slim = tf.contrib.slim
import sys
sys.path.append(r"C:\Users\autpucv\WindowsNoEditor\PythonAPI\carla")
from carla import VehicleControl
from .imitation_learning_network import load_imitation_learning_network
import cv2 as cv

class ImitationLearning():

    def __init__(self, city_name, avoid_stopping, memory_fraction=0.25, image_cut=[115, 510]):


        self.dropout_vec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5

        config_gpu = tf.ConfigProto()

        # GPU to be selected, just take zero , select GPU  with CUDA_VISIBLE_DEVICES

        # config_gpu.gpu_options.visible_device_list = '0'

        # config_gpu.gpu_options.per_process_gpu_memory_fraction = memory_fraction

        self._image_size = (88, 200, 3)
        self._avoid_stopping = avoid_stopping

        self._sess = tf.Session(config=config_gpu)
        with tf.device('/cpu:0'):
            self._input_images = tf.placeholder("float", shape=[None, self._image_size[0],
                                                                self._image_size[1],
                                                                self._image_size[2]],
                                                name="input_image")

            self._input_data = []

            self._input_data.append(tf.placeholder(tf.float32,
                                                   shape=[None, 4], name="input_control"))

            self._input_data.append(tf.placeholder(tf.float32,
                                                   shape=[None, 1], name="input_speed"))

            self._dout = tf.placeholder("float", shape=[len(self.dropout_vec)])

        with tf.name_scope("Network"):
            self._network_tensor = load_imitation_learning_network(self._input_images,
                                                                   self._input_data,
                                                                   self._image_size, self._dout)
        self.start_time = time.time()
        import os
        dir_path = os.path.dirname(__file__)

        self._models_path = os.path.join(dir_path, "model")
        
        # tf.reset_default_graph()
        self._sess.run(tf.global_variables_initializer())
        #print_weights(self._sess)

        self.load_model()

        self._image_cut = image_cut
        
    def load_model(self):

        variables_to_restore = tf.global_variables()

        saver = tf.train.Saver(variables_to_restore, max_to_keep=0)

        if not os.path.exists(self._models_path):
            raise RuntimeError('failed to find the models path')

        ckpt = tf.train.get_checkpoint_state(self._models_path)
        if ckpt:
            print('Restoring from ', ckpt.model_checkpoint_path)
            saver.restore(self._sess, ckpt.model_checkpoint_path)
        else:
            ckpt = 0

        return ckpt
    def get_action(self, sensor_data, measurements, directions):
        control = self.run_step(measurements, sensor_data, directions)
        return control.steer, control.throttle, control.brake
    def run_step(self, measurements, sensor_data, directions, target=None):

        control = self._compute_action(sensor_data,
                                       measurements, directions)

        return control
    def _compute_action(self, rgb_image, speed, direction=None):

        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]
        #img = Image.fromarray(rgb_image)
        print(self._image_size)
        #img = img.resize((self._image_size[0], self._image_size[1]), Image.BILINEAR)
        #image_input = np.array(img).astype(np.float)
        #print(image_input.shape)
        image_input = misc.imresize(rgb_image, [self._image_size[0],
                                                       self._image_size[1]])
        image_input = np.multiply(image_input, 1.0 / 255.0)

        steer, acc, brake = self._control_function(image_input, speed, direction, self._sess)
        print(steer, acc, brake)
        # This a bit biased, but is to avoid fake breaking

        if brake < 0.1:
            brake = 0.0

        if acc > brake:
            brake = 0.0

        # We limit speed to 35 km/h to avoid
        if speed > 10.0 and brake == 0.0:
            acc = 0.0

        control = VehicleControl()
        steer = steer.item() if type(steer) != float else steer
        acc = acc.item() if type(acc) != float else acc
        brake = brake.item() if type(brake) != float else brake
        control.steer = steer
        control.throttle = acc
        control.brake = brake

        control.hand_brake = False 
        control.reverse = False

        return control
    def _compute_action2(self, rgb_image, speed, direction=None):

        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :, :]

        # image_input = scipy.misc.imresize(rgb_image, [self._image_size[0],
        #                                               self._image_size[1]])

        image_input = rgb_image.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)

        steer, acc, brake = self._control_function(image_input, speed, direction, self._sess)
        steer = steer.item() if type(steer) != float else steer
        acc = acc.item() if type(acc) != float else acc
        brake = brake.item() if type(brake) != float else brake
        # This a bit biased, but is to avoid fake breaking

        if brake < 0.1:
            brake = 0.0

        if acc > brake:
            brake = 0.0

        # We limit speed to 35 km/h to avoid
        if speed > 10.0 and brake == 0.0:
            acc = 0.0

        control = VehicleControl(steer=steer, throttle=acc, brake=brake)
        #control.hand_brake = 0
        #control.reverse = 0

        return control

    def _control_function(self, image_input, speed, control_input, sess):

        branches = self._network_tensor
        x = self._input_images
        dout = self._dout
        input_speed = self._input_data[1]
        cv.imshow("reshaped", (image_input.reshape((88, 200, 3)) * 255).astype('uint8'))
        cv.waitKey(1)
        #image_input = image_input[self._image_cut[0] : self._image_cut[1], :]
        image_input = image_input.reshape(
            (1, self._image_size[0], self._image_size[1], self._image_size[2]))
        
        # Normalize with the maximum speed from the training set ( 90 km/h)
        speed = np.array(speed / 25.0)

        speed = speed.reshape((1, 1))
      
        if control_input == 2 or control_input == 0.0:
            all_net = branches[0]
        elif control_input == 3:
            all_net = branches[2]
        elif control_input == 4:
            all_net = branches[3]
        else:
            all_net = branches[1]

        feedDict = {x: image_input, input_speed: speed, dout: [1] * len(self.dropout_vec)}

        output_all = sess.run(all_net, feed_dict=feedDict)

        predicted_steers = (output_all[0][0])

        predicted_acc = (output_all[0][1])

        predicted_brake = (output_all[0][2])

        if self._avoid_stopping:
            predicted_speed = sess.run(branches[4], feed_dict=feedDict)
            predicted_speed = predicted_speed[0][0]
            real_speed = speed * 25.0

            real_predicted = predicted_speed * 25.0
            if real_speed < 2.0 and real_predicted > 3.0:
                # If (Car Stooped) and
                #  ( It should not have stopped, use the speed prediction branch for that)

                predicted_acc = 1 * (5.6 / 25.0 - speed) + predicted_acc

                predicted_brake = 0.0

                predicted_acc = predicted_acc[0][0]
        predicted_brake = 0
        return predicted_steers, predicted_acc, predicted_brake