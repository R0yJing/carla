

from matplotlib import pyplot as plt 
from constants import MAX_REPLAY_BUFFER_SIZE, IM_HEIGHT, IM_WIDTH, MINI_BATCH_SIZE, NUM_AGENT_TRAIN_STEPS_PER_ITER, NUM_EPOCHS, STORAGE_LIMIT, TARGET_SPEED

from random import shuffle
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Flatten, Conv2D, Reshape, concatenate
from keras import initializers
import numpy as np
import os
import cv2
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
import carla
#https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout

#source that recommend he_uniform kernel initializer, instead of the default glorot uniform initializer (when using relu as activation)

#https://stackoverflow.com/questions/54011173/is-there-any-documentation-about-default-weight-initializer-in-keras

#source that recommend a bias of 0.1 
#https://github.com/carla-simulator/imitation-learning/blob/62f93c2785a2452ca67eebf40de6bf33cea6cbce/agents/imitation/imitation_learning_network.py#L13

from keras.layers import Activation
#dog_dir = os.listdir(r".\PetImages\Dog")
CHECKPT_FOLDER_DIR = r"C:\Users\autpucv\Desktop\my scripts\imitation_learning\checkpoints"#r".\checkpoints"
def add_fc_block(base_layer, num_units, name, dropout=0.5):
    #first need to add wx+b layer
    #default weight initializer is glorot uniform
    temp = Dense(num_units, activation="linear", bias_initializer=initializers.constant(0.1), name=name)(base_layer)
    
    #temp = tf.keras.layers.Dropout(dropout)(temp)
    temp = tf.keras.layers.Activation('relu')(temp)
    return temp

def add_test_fc_block(base_layer, num_units, name):
    temp = Dense(num_units, activation='relu', name=name)(base_layer)
    return temp
def add_conv_block(base_layer, filters, kern_size, strides, name, dropout=0.2):
    #if input_shape is not None:
    #print("input layer")
    temp = Conv2D(filters, kernel_size=kern_size, strides=strides, padding="valid", bias_initializer=initializers.constant(0.1), name=name)(base_layer)
     #   base_layer.add(tf.keras.layers.Conv2D(filters, kernel_size=kern_size, strides=strides, padding="valid", activation='relu'))
    temp = tf.keras.layers.BatchNormalization()(temp)
    #temp = tf.keras.layers.Dropout(dropout)(temp) 
    temp = tf.keras.layers.Activation('relu')(temp) 
    
    return temp
def get_img_array(idx):
    #print(dog_dir[idx])
    img_array = cv2.imread(os.path.join(r".\PetImages", "Dog", dog_dir[idx]), cv2.IMREAD_GRAYSCALE)
    #if idx ==0:
        #cv2.imshow("", img_array)
    #cv2.imshow("",img_array)
    img_array.resize((IM_HEIGHT, IM_WIDTH))
    
    #img_array = img_array.reshape(-1)
    #print("shpape of image")
    #print(img_array.shape)
    return img_array / 255.0
class test_image_module:
    def eval(self):
        _, mse = self.model.evaluate(self.x, self.y, verbose='1')
        print(mse * 100.0)

    def print_info(self):
        weights = self.model.get_weights()
        print("=============weights==============")
        print()
        for i, w in enumerate(weights):
            print(f"layer {i}")
            print(w.shape)
            print(w)
            print("--------------")
        print()
        print("=============summary==============")
        print(self.model.summary())
    def __init__(self):
        x = [self.add_noise(250) for i in range(1000)] + [self.add_noise(255/2.0) for i in range(1000)] + [self.add_noise(0) for i in range(1000)]
        y = [random.random() / 10.0 for i in range(1000)] + [1 + random.random() / 10.0 for i in range(1000)] + [2 + random.random() / 10.0 for one in range(1000)]
        for i in range(3000 - 1):
            ind = random.randint(i + 1, 3000 - 1)
            x[i], x[ind] = x[ind], x[i]
            y[i], y[ind] = y[ind], y[i]
        self.x = np.array(x).reshape(len(x), 88,200,3)
        self.y = np.array(y)
        #add 1) to denote working with a grayscale image =
        im_module = image_module()
        self.model = Model(im_module.image_model_in, Dense(1)(im_module.image_model_out))

        # temp = add_conv_block(temp, 64, 3, 2, "image_module_l3")
        # temp = add_conv_block(temp, 64, 3, 1, "image_module_l4")
        # temp = add_conv_block(temp, 128, 3, 2, "image_module_l5")
        # temp = add_conv_block(temp, 128, 3, 1, "image_module_l7")
        # temp = add_conv_block(temp, 256, 3, 1, "image_module_l8")
        #flatten the input so the output is flat as well
     
        self.print_info()
        try:
            self.model.load_weights(r".\checkpoints\test_image_weights.hdf5")
        except:
            pass
        self.print_info()
        from keras.optimizers import Adam
        opt = Adam(lr=1e-4, decay=1e-4/NUM_EPOCHS)
        self.model.compile(optimizer=opt, loss='mse', metrics=['mse'])

    def add_noise(self, base):
        mat = np.random.uniform(0, 5, (88,200,3))
        mat += base 

        return mat / 255.0
    def train(self): #mse should use linear activation
        #sigmoid for classfication problems 
        print(self.model.summary())
        
        
        chkpt = ModelCheckpoint(os.path.join(CHECKPT_FOLDER_DIR, "test_image_weights.hdf5"), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        history = self.model.fit(x=self.x, y=self.y, batch_size=120, validation_split=0.2, epochs=NUM_EPOCHS, verbose='auto', shuffle=True, callbacks=[chkpt])
        try:
            show_accuracy_graph([history])
        except:
            print("cannot show mse graph")
class image_module:
    def __init__(self):
        #add 1) to denote working with a grayscale image =
        self.image_model_in = Input((IM_HEIGHT, IM_WIDTH,3))
        temp = add_conv_block(self.image_model_in, 32, 5, 2, "image_module_l1")
        temp = add_conv_block(temp, 32, 5, 1, "image_module_l2")
        temp = add_conv_block(temp, 64, 3, 2, "image_module_l3")
        temp = add_conv_block(temp, 64, 3, 1, "image_module_l4")
        temp = add_conv_block(temp, 128, 3, 2, "image_module_l5")
        temp = add_conv_block(temp, 128, 3, 1, "image_module_l7")
        temp = add_conv_block(temp, 256, 3, 1, "image_module_l8") 
        #flatten the input so the output is flat as well
        length = np.array(temp.shape.as_list()[1:])
        length = np.prod(length)
        temp = Reshape((length,))(temp)
        # temp = Flatten()(temp)
        temp = add_fc_block(temp, 512, "image_module_fc_l9")
        self.image_model_out = add_fc_block(temp, 512, "image_module_fc_l10")
        #image_model_out = Dense(512, activation='relu', name='layer_1')(image_model_in)
        
        
#model1 = Model(model1_in, model1_out)
#print(model1_in.shape)

class mlp:
    def __init__(self, input_size, name):
        
        self.module_in = Input((input_size,))
        layer1 = add_fc_block(self.module_in, 128, f"{name}_l1")
        self.module_out = add_fc_block(layer1, 128, f"{name}_l2")

        #speed_model_out = Dense(1, activation='relu', name='layer_2')(speed_model_in)


class action_module:
    def __init__(self, base_layer :Activation):
        print("aciton module input")
        temp = add_fc_block(base_layer, 256, "action_l1")
        self.ac_in = temp
        temp = add_fc_block(temp, 256, "action_l2")
        #TODO implement properly as wx_plus_b
        self.action_module_out = Dense(3, activation="linear", use_bias=True, bias_initializer=initializers.constant(0.1))(temp)

#0.5

# shared_layer1 = Dense(518)(concatenated)
# shared_layer2 = Dense(518)(shared_layer1)


#print(merged_model.summary())


# checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc',
# save_best_only=True, verbose=2)
# early_stopping = EarlyStopping(monitor="val_loss", patience=5)

import random
#image = get_img_array(0)
import math
from keras.optimizers import Adam
#print(image)

    #print(y)
MAX_BRANCH_BUFFER_SIZE = 53760
def show_accuracy_graph(histories):
    accumulated_accuracies = [point for history in histories for point in history.history["mse"]]
    accumulated_val_accuracies = [point for history in histories for point in history.history["val_mse"]]
    plt.plot(accumulated_accuracies)
    plt.plot(accumulated_val_accuracies)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
# X = np.random.uniform(0,1,size=(100,IM_HEIGHT,IM_WIDTH)) 
# print(X.shape)
# X2 = np.random.uniform(0,1,size=(100,1))
# print(X2.shape)
#the model epects two inputs
import pickle
import sys
sys.path.insert(0, r"C:\Users\autpucv\Downloads\CARLA_0.8.2\PythonClient\carla\agent")
from agent import Agent
class agent(Agent): 
    
    def __init__(self, fake_training=False, training=True, simulating=False):
        self.histories = []
        self.suggested_actions = []
        
        self.num_inputs_added = 0
        self.test_actions = []
        self.test_images = []
        self.test_cmds = []
        self.test_speeds = []

        self.train_actions = []
        self.train_images = []
        self.train_cmds = []
        self.train_speeds = []
        split = int(math.ceil(0.2 * MAX_REPLAY_BUFFER_SIZE))
        if not fake_training and not simulating:

            try:
                if training:
                    
                        #self.train_images, self.train_speeds, self.train_cmds = self.normalise_samples(d[0], d[1], d[2])
                        #self.train_actions = np.array(d[3])
                        
                        self.split = split
                        #self.load_training_data(r'.\recordings\training\recording-p1.pkl')    
                        #self.load_training_data(r'.\recordings\training\recording-p2.pkl') 
                        #self.load_data(r'.\recordings\training\recording-4.pkl')
                        #self.shuffle([self.train_images, self.train_speeds, self.train_cmds, self.train_actions])   
                        #print(f"there are {len(self.replay_buffer_left)} number of data")
                else:
                    with open(r'.\recordings\testing\recording-1-p1.pkl', 'wb') as f:
                        d = pickle.load(f)
                        self.test_images, self.test_speeds, self.test_cmds = self.normalise_samples(d[0], d[1], d[2])
                        self.test_actions = np.array(d[3])

            except Exception as e:
                raise e
        elif fake_training: pass
            # self.train_images = [np.random.uniform(0, 255, (88, 200, 3)) for i in range(500)]
            # self.train_actions = [[0, 0.5, 0] for i in range(500)]
            # self.train_cmds = ["straight" for i in range(500)]
            # self.train_speeds = [30 for i in range(500)]
        self.checkpoint = ModelCheckpoint(os.path.join(CHECKPT_FOLDER_DIR, "weights.best.testing.225epochs.patience3.batch_size180.validation0.33_1.hdf5"), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        self.early_stopping = EarlyStopping(monitor ="val_loss",
                                        mode ="min", patience = 3, 
                                        restore_best_weights = True)
        self.model = self.create_model()
        print("loading weights...")
        self.try_load_weights()
        
        opt = Adam(lr=1e-4, decay=1e-4/NUM_EPOCHS)
        #large errors are heavily penalised with the mean squared error
        self.model.compile(loss='mse', optimizer=opt, 
metrics=['mse'])
        self.split = math.ceil(MAX_REPLAY_BUFFER_SIZE * 0.2)

    def dump(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump([self.train_images,
                        self.train_speeds,
                        self.train_cmds,
                        self.train_actions], f)
    
    def load_n_images(self, n, filename=r'.\recordings\training\recording-p1.pkl'):
        with open(filename, 'rb') as f:
            d = pickle.load(f)
            self.shuffle(d)
            normalised_imgs, normalised_speeds, normalised_cmds = self.normalise_samples(d[0][:n], d[1][:n], d[2][:n])
            normalised_actions = np.array(d[3][:n])
            self.test_images += [img for img in normalised_imgs]
            self.test_speeds += [spd for spd in normalised_cmds]
            self.test_cmds += [act for act in normalised_actions]
            self.test_actions += [spd for spd in normalised_speeds]
        
    def shuffle(self, data):
        
       #last is -2
        for i in range(len(data[0]) - 1):
            idx = random.randint(i + 1, len(data[0]) - 1)

            data[0][i], data[0][idx] = data[0][idx], data[0][i]
            data[1][i], data[1][idx] = data[1][idx], data[1][i]
            data[2][i], data[2][idx] = data[2][idx], data[2][i]
            data[3][i], data[3][idx] = data[3][idx], data[3][i]
    def load_data(self,filename, training=True):

        with open(filename, 'rb') as f:
            d = pickle.load(f)
            normalised_imgs, normalised_speeds, normalised_cmds = self.normalise_samples(d[0], d[1], d[2])
            normalised_actions = np.array(d[3])
            if training:
                self.train_images += [img for img in normalised_imgs]
                self.train_cmds += [spd for spd in normalised_cmds]
                self.train_actions += [act for act in normalised_actions]
                self.train_speeds += [spd for spd in normalised_speeds]
            else:
                self.test_images += [img for img in normalised_imgs]
                self.test_cmds += [spd for spd in normalised_cmds]
                self.test_actions += [act for act in normalised_actions]
                self.test_speeds += [spd for spd in normalised_speeds]
    def try_load_weights(self):

        files = os.listdir(CHECKPT_FOLDER_DIR)
        if len(files) != 1:
            print("no checkpoints saved!")
        else:

            print(f"found checkpoint : {files[0]}")
            checkpt = os.path.join(CHECKPT_FOLDER_DIR, files[0])

            self.model.load_weights(checkpt)
    
    def create_model(self):
        spd_module = mlp(1, "speed")
        cmd_module = mlp(3, "command")
        self.cmd_module = cmd_module
        self.spd_module = spd_module
        
        img_module = image_module()
        self.img_module = img_module
        concatenated = concatenate([img_module.image_model_out, spd_module.module_out, cmd_module.module_out]) #TODO fix!!!
        intermediate_layer = add_fc_block(concatenated, 512, "intermediate_layer") 
        ac_module = action_module(intermediate_layer)
        self.ac_module = ac_module
        out = ac_module.action_module_out
        merged_model = Model([img_module.image_model_in , spd_module.module_in, cmd_module.module_in], out)
        return merged_model

    def _pop_from_buffer(self):
        self.train_images.remove(self.train_images[0])
        self.train_cmds.remove(self.train_cmds[0])
        self.train_speeds.remove(self.train_speeds[0])
        self.train_actions.remove(self.train_actions[0])

    def _add_to_buffer(self, img, speed, cmd, action):
        '''assuming normalised'''
        
        self.train_images.insert(MAX_REPLAY_BUFFER_SIZE - self.split, img)
        self.train_speeds.insert(MAX_REPLAY_BUFFER_SIZE - self.split, speed)
        self.train_cmds.insert(MAX_REPLAY_BUFFER_SIZE - self.split, cmd)
        self.train_actions.insert(MAX_REPLAY_BUFFER_SIZE - self.split, action)


    def insert_input(self, image, speed, cmd, action):
                    
        if len(self.test_images) == MAX_REPLAY_BUFFER_SIZE:
            self._pop_from_buffer()
        self._add_to_buffer(image, speed, cmd, action)
    
    

    def calculate_min_buffer_size(self, per_sample_size = 62e3):
        return round(STORAGE_LIMIT * 1e9 / per_sample_size / 3)
    
    def evaluate(self):
        files = os.listdir(r'.\recordings\testing')
        self.load_data(r'.\recordings\testing\recording-1-p2.pkl', training=False)

        l = len(self.test_images)
        acc = self.model.evaluate(x=[np.array(self.test_images), np.array(self.test_speeds), np.array(self.test_cmds)], y=np.array(self.test_actions))

        return
        
    def show_plots(self, history):
        #accuracy
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='left')
        #loss
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='right')
        plt.show()
    def train(self):
        # num_samples = round(TRAIN_BATCH_SIZE / 3)
        
        split = math.ceil(len(self.train_cmds) / 3)
        start =  0
        end=start+split
        history_count = 0
        for i in range(3):
            imgs = np.array(self.train_images[start:end])
            spds = np.array(self.train_speeds[start:end])
            cmds = np.array(self.train_cmds[start:end])
            acts = np.array(self.train_actions[start:end])
            history = self.model.fit(x=[imgs, spds, cmds], y=acts, epochs=NUM_EPOCHS, batch_size=150, validation_split=0.2, shuffle=True, callbacks=[self.early_stopping, self.checkpoint])
            start = end
            end = min(MAX_REPLAY_BUFFER_SIZE, end + split)

            self.histories.append(history)
           
        # for i in range(500):
            
        #     indices = random.sample(range(0, MAX_REPLAY_BUFFER_SIZE - self.split), int(TRAIN_BATCH_SIZE * 0.8))
        #     indices_validation = random.sample(range(MAX_REPLAY_BUFFER_SIZE - self.split, MAX_REPLAY_BUFFER_SIZE), int(TRAIN_BATCH_SIZE * 0.2))
        #     imgs = np.array([self.train_images[idx] for idx in indices])
        #     speeds = np.array([self.train_speeds[idx] for idx in indices])
        #     cmds = np.array([self.train_cmds[idx] for idx in indices])
        #     actions = np.array([self.train_actions[idx] for idx in indices])
        
        #     validate_images = np.array([self.train_images[idx] for idx in indices_validation])
        #     validate_speeds = np.array([self.train_speeds[idx] for idx in indices_validation])
        #     validate_cmds = np.array([self.train_cmds[idx] for idx in indices_validation])
        #     validate_actions = np.array([self.train_actions[idx] for idx in indices_validation])
        #     history = self.model.fit(x=[imgs, speeds, cmds], y=actions, validation_data=([validate_images, validate_speeds, validate_cmds], validate_actions), epochs=1)
        #     self.histories.append(history)
        #use a validation split of 0.33
            #model accuracy
        
        #10 epochs
        # for _ in range(10):
        #     history= None
        #     start = 0
        #     step = int(MAX_REPLAY_BUFFER_SIZE / 3)

        #     stop = step
            # for i in range(3):
                
            #     imgs, speeds, cmds = self.normalise_samples(self.train_images[start : stop], self.train_speeds[start : stop], self.train_cmds[start : stop])
            #     history = self.model.fit([imgs, speeds, cmds], y=np.array(self.train_actions[start : stop]), validation_split=0.33, batch_size=TRAIN_BATCH_SIZE, epochs=1, callbacks=[self.checkpoint, self.early_stopping])

            #     start += step
            #     stop += step if i < 1 else step - 1
            # self.histories.append(history)
        #each iteration o
        show_accuracy_graph(self.histories)
    
        
    # def show_loss_graph(self, histories):
    #     accumulated_losses = [point for history in histories for point in history["loss"]]
    #     accumulated_val_losses = [point for history in histories for point in history["val_loss"]]
    #     plt.plot(accumulated_losses)
    #     plt.plot(accumulated_val_losses)
    #     plt.title('model loss')
    #     plt.ylabel('loss')
    #     plt.xlabel('epoch')
    #     plt.legend(['train', 'val'], loc='upper left')
    def normalise_single_sample(self, image, speed, cmd, grayscale= False):
        image = np.reshape(image, (IM_HEIGHT, IM_WIDTH, 1 if grayscale else 3)) / 255
        if cmd == "left":
            cmd = [0,0,1]
        elif cmd == "straight":
            cmd = [0,1,0]
        else:
            cmd = [1,0,0]
        speed /= TARGET_SPEED

        return image, speed, cmd

    def normalise_samples(self,  images, speeds, commands, grayscale=False):
       
        normalised_cmds = []
        
        images = np.array(images) / 255 if images[0].dtype != 'float' else np.array(images)
        speeds = np.array(speeds) / TARGET_SPEED
        left_cmd = [0,0,1]
        right_cmd = [1,0,0]
        straight_cmd = [0,1,0]
        left = 0
        right = 0
        straight = 0
        normalised_cmds = [left_cmd if cmd == "left" else straight_cmd if cmd == "straight" else right_cmd for cmd in commands]
        return images, speeds, np.array(normalised_cmds)
        # for img,speed,command in zip(images, speeds, commands):
        #     img, speed, command = self.normalise_single_sample(img, speed, command)
        #     normalised_imgs.append(img)
        #     normalised_speeds.append(speed)
        #     normalised_cmds.append(command)
        #return np.array(normalised_imgs), np.array(normalised_speeds), np.array(normalised_cmds)


    def get_actions(self, images, speeds, commands, grayscale = False):
        #images is still not an ndarray at this time
        
        images, speeds, commands = self.normalise_samples(images,speeds,commands)
        
        predictions = self.model.predict([images, speeds, commands], len(images), verbose='0')
        return predictions

    def run_step(self, measurements, sensor_data, directions, target):

        s,t,b = self.get_single_action(sensor_data['CameraRGB'].data,
                                       measurements.player_measurements.forward_speed, directions)

        return carla.VehicleControl(steer=s, throttle=t,brake=s)

    def get_single_action(self, image,speed, command):
        steer, throttle, brake = self.get_actions([image], [speed], [command])[0]
        return np.clip(steer, -1, 1), np.clip(throttle, 0, 1), np.clip(brake, 0, 1)
def test_evaluate(agent : agent):
    # agent.test_images = np.random.uniform(0, 1, (300, 88, 200, 3))
    # agent.test_speeds = np.random.uniform(0, 1, (300,1))
    # agent.test_cmds = np.random.uniform(0, 1, (300, 3))
    # agent.test_actions = np.random.uniform(0, 1, (300, 3))
    agent.evaluate()
def test_train():
    agt = agent()
    agt.train()
    # agent.img_replay_buffer = np.random.uniform(0, 255, (TRAIN_BATCH_SIZE, IM_HEIGHT, IM_WIDTH))
    # agent.speed_replay_buffer = np.random.uniform(0, 90, (TRAIN_BATCH_SIZE,))
    # agent.cmd_replay_buffer = [[1,0,0] for  i in range(TRAIN_BATCH_SIZE)]
    # agent.suggested_actions = [[1,1,1] for i in range(TRAIN_BATCH_SIZE)]
    
    # agent.train_images = np.random.uniform(0, 1, (300, 88, 200, 3))
    # agent.train_speeds = np.random.uniform(0, 1, (300,1))
    # agent.train_cmds = np.random.uniform(0, 1, (300, 3))
    # agent.train_actions = np.random.uniform(0, 1, (300, 3))
def test_show_graph():
    histories = [{"accuracy":[i for i in range(100)], "val_accuracy" : [i + 3 for i in range(100, 200)]}, {"accuracy":[i for i in range(100, 200)], "val_accuracy" : [i + 4 for i in range(100, 200)]}]
    agt = agent(fake_training=True)
    show_accuracy_graph(histories)
def test_insert():
    agt = agent(fake_training=True)
    img = np.random.uniform(0, 255, (88, 200, 3))
    speed = 29
    cmd = "straight"
    action = [0, 1, 2]
    for i in range(100):
        agt.insert_input(img, speed, cmd, action)

    agt.train()
def test_load_data():
    pass
def print_weights(agent : agent, n):
    weights = agent.model.get_weights()
    
    with open(f"keras weights before", "w") as f:

        for i, w in enumerate(weights):
            f.write(f"layer {i}\n")
            f.write(str(w) + "\n")
            f.write("\n")

class test_module:
    def __init__(self):
        inLayer = Input((3,3,1))
    
        temp = add_conv_block(inLayer, 25, 2, 1, "test_conv" )
        temp = add_conv_block(temp, 25, 2, 1, "layer2")
        temp = Flatten()(temp)
        #temp = Dense(1)(temp)

        temp = Dense(1, activation=None, use_bias=False)(temp)
        self.model = Model(inLayer, temp)
        f=r"checkpoints\test_weights.hdf5"
        if os.path.exists(f): 

            self.model.load_weights(f)
        self.model.compile(loss='mse', optimizer='adam', 
metrics=['accuracy'])
    def train(self):
        print(self.model.summary())

        one = np.array([[0,0,1], [0,1,0], [1,0,0]])
        two = np.array([[0,1,0], [0,1,0], [0,1,0]])
        three = np.array([[1,0,0], [0,1,0], [0,0,1]])
        
        x = [one for i in range(5000)] + [two for i in range(5000)] + [three for one in range(5000)]
        y = [0 + random.random() / 10.0 for i in range(5000)] + [1 + random.random() / 10.0 for i in range(5000)] + [2 + random.random() / 10.0 for one in range(5000)]
        for i in range(15000 - 1):
            ind = random.randint(i + 1, 15000 - 1)
            x[i], x[ind] = x[ind], x[i]
            y[i], y[ind] = y[ind], y[i]
        x = np.array(x).reshape(len(x), 3,3,1)
        y = np.array(y)
        chkpt = ModelCheckpoint(os.path.join(CHECKPT_FOLDER_DIR, "test_weights.hdf5"), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        history = self.model.fit(x=x, y=y, batch_size=120, validation_split=0.2, epochs=100, verbose='0', shuffle=True, callbacks=[chkpt])
        show_accuracy_graph([history])
   
# #print(agt.get_single_action(image, speed,command))

#test_load_data()
# test_evaluate(agt)
#test_evaluate(agt)
#test_show_graph()
#test_insert()
             #verbose=1, shuffle=True)
    

#callbacks=[early_stopping, checkpoint])