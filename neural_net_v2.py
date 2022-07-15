

from matplotlib import pyplot as plt
from pyrsistent import s 
from constants import BATCH_SIZE, MAX_REPLAY_BUFFER_SIZE, IM_HEIGHT, IM_WIDTH, MINI_BATCH_SIZE, NUM_AGENT_TRAIN_STEPS_PER_ITER, NUM_EPOCHS, STORAGE_LIMIT, TARGET_SPEED
import h5py
from collections import deque 
from random import sample, shuffle
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Flatten, Conv2D, Reshape, concatenate
from keras import initializers
import numpy as np
import os
import cv2
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
import carla
import glob
#https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
#apply dropout after activation https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
#source that recommend he_uniform kernel initializer, instead of the default glorot uniform initializer (when using relu as activation)

#https://stackoverflow.com/questions/54011173/is-there-any-documentation-about-default-weight-initializer-in-keras

#source that recommend a bias of 0.1 
#https://github.com/carla-simulator/imitation-learning/blob/62f93c2785a2452ca67eebf40de6bf33cea6cbce/agents/imitation/imitation_learning_network.py#L13

from keras.layers import Activation

from image_augmenter import image_augmenter
#dog_dir = os.listdir(r".\PetImages\Dog")
CHECKPT_FOLDER_DIR = r"C:\Users\autpucv\Desktop\my scripts\imitation_learning\checkpoints"#r".\checkpoints"
NUM_TRAIN_SAMPLES = 74800
def add_fc_block(base_layer, num_units, name, dropout=0.5):
    #first need to add wx+b layer
    #default weight initializer is glorot uniform
    temp = Dense(num_units, activation="linear", bias_initializer=initializers.constant(0.1), name=name)(base_layer)
    temp = tf.keras.layers.BatchNormalization()(temp)
    #temp = tf.keras.layers.Dropout(dropout)(temp)
    temp = Activation('relu')(temp)
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
    temp = Activation('relu')(temp)
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
        opt = Adam(lr=1e-4, beta_1=0.7,beta_2=0.85)
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
        #base_layer filter kernsize  strides
        temp = add_conv_block(self.image_model_in, 32, 5, 2, "image_module_l1")
        temp = add_conv_block(temp, 32, 3, 1, "image_module_l2")
        temp = add_conv_block(temp, 64, 3, 2, "image_module_l3")
        temp = add_conv_block(temp, 64, 3, 1, "image_module_l4")
        temp = add_conv_block(temp, 128, 3, 2, "image_module_l5")
        temp = add_conv_block(temp, 128, 3, 1, "image_module_l7")
        temp = add_conv_block(temp, 256, 3, 2, "image_module_l8") 
        #temp = add_conv_block(temp, 256, 3, 1, "image_module_l8") 
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
from keras.utils import Sequence
sys.path.insert(0, r"C:\Users\autpucv\Downloads\CARLA_0.8.2\PythonClient\carla\agent")
from agent import Agent
class Generator(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, dataset, batch_size=120):
        self.x = dataset
        self.batch_size = batch_size
        self.indices = [i for i in range(len(dataset))]
        random.shuffle(self.indices)
        self.augmenter = image_augmenter()
       
    def __len__(self):
        return len(self.indices) // 120

    def __getitem__(self, idx):
        
        batch_img = self.augmenter.aug(np.array([sample[0] for sample in self.x[idx*self.batch_size:(idx+1) * self.batch_size]]))/255
        batch_speeds = np.array([sample[1] for sample in self.x[idx*self.batch_size:(idx+1) * self.batch_size]])
        batch_cmds = np.array([sample[2] for sample in self.x[idx*self.batch_size:(idx+1) * self.batch_size]])
        
        batch_y = np.array([sample[3] for sample in self.x[idx*self.batch_size:(idx+1) * self.batch_size]])
        return ([batch_img, batch_speeds, batch_cmds], batch_y)
    
    def on_epoch_end(self):
      
        random.shuffle(self.indices)
class agent(Agent): 
    
    def __init__(self, fake_training=False, training=True, simulating=False):
        self.histories = []
        self.suggested_actions = []
        self.num_errors = 0
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
                        #self.load_data(r'.\recordings\training\recording-7.pkl')
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
        self.checkpoint = ModelCheckpoint(os.path.join(CHECKPT_FOLDER_DIR, "weights.best.testing.225epochs.patience3.batch_size180.validation0.33_1.hdf5"), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        self.early_stopping = EarlyStopping(monitor ="val_loss",
                                        restore_best_weights = True,
                                        mode ="auto", patience = 3)
        self.augmenter = image_augmenter()                            
        self.cache = {'images':[], "speeds" : [], "commands" : [], "actions" : []}
        self.val_cache = {'images':[], "speeds" : [], "commands" : [], "actions" : []}
        self.model = self.create_model()
        print("loading weights...")
    
        self.try_load_weights()
        
        opt = Adam(lr=1e-4)
        #large errors are heavily penalised with the mean squared error
        self.model.compile(loss='mean_squared_error', optimizer=opt, 
metrics=['mse', 'accuracy'])
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
            self.test_cmds += [spd for spd in normalised_cmds]
            self.test_speeds += [spd for spd in normalised_speeds]  
            self.test_actions += [act for act in normalised_actions]
        
    def shuffle(self, data):
        
       #last is -2
        for i in range(len(data[0]) - 1):
            idx = random.randint(i + 1, len(data[0]) - 1)

            data[0][i], data[0][idx] = data[0][idx], data[0][i]
            data[1][i], data[1][idx] = data[1][idx], data[1][i]
            data[2][i], data[2][idx] = data[2][idx], data[2][i]
            data[3][i], data[3][idx] = data[3][idx], data[3][i]
    def test_batch_gen(self, filenames):
        for _ in self.generator():
            pass
        
    def get_samples(self, filenames, upper_lim, searched_indices : dict):
        sub_lim = upper_lim / 3
        vectorised_commands = np.eye(3).astype('uint8')
    
        follow_lane_ct = 0
        left_ct = 0
        right_ct = 0
        i = 0
        samples = []
    
        while follow_lane_ct < sub_lim or left_ct < sub_lim or right_ct < sub_lim:
            
            try:
                #if not searched_indices.get(i):
                    #searched_indices[i] = True
                with h5py.File(filenames[i//200], 'r') as f:
                    #searched_indices[i] = True
                    cmd = f['targets'][i%200][24]
                    if cmd == 5:
                        i+=1; 
                        continue
                    unpacked_sample = (f['rgb'][i% 200], f['targets'][i%200][10] / TARGET_SPEED, vectorised_commands[int(cmd - 2)], f['targets'][i%200][:3] )
                    if cmd == 2 and follow_lane_ct < sub_lim:
                        samples.append(unpacked_sample)
                        follow_lane_ct += 1
                    elif cmd == 3 and left_ct < sub_lim:
                        samples.append(unpacked_sample)
                        left_ct += 1
                    elif cmd == 4 and right_ct < sub_lim:
                        samples.append(unpacked_sample)
                        right_ct += 1
                    #straight is omitted due to scarcity

            except Exception as e:
                print(e)
            i+=1
    
        return samples
    def generator(self):
        for i in range(2):
            for j in range(2):
                
                print("generate")
                yield None
    def batch_generator(self, is_validation, upperlim, num_data_loads=2):
        filenames = glob.glob('dataset\SeqVal\*.h5' if is_validation else 'dataset\SeqTrain\*.h5')
        train_searched_indices = dict()
        num_samples_per_load = upperlim / num_data_loads
        ct = 0
        #the generator function does not reset after an epoch
        while True:
            print("epoch>>>>>>>>>>>..."+str(ct))
            ct+=1
            # if (len(self.cache['images']) != 0 and not is_validation) or (len(self.val_cache['images']) != 0 and is_validation):
            #     print('cache')

            #     cache = self.val_cache if is_validation else self.cache
            #     random_indices = [i for i in range(upperlim)]
            #     random.shuffle(random_indices)
                
            #     for i in range(0, upperlim, 120):
            #         yield ([self.augmenter.aug(np.array(cache['images'][i:i+120]))/255, np.array(cache['speeds'][i:i+120]), np.array(cache['commands'][i:i+120])], np.array(cache['actions'][i:i+120]))
                
            # else:
            print("no cache")
            for i in range(2):
                samples = self.get_samples(filenames, num_samples_per_load, train_searched_indices)
                random.shuffle(samples)
                for i in range(0, len(samples), 120):
                
                    batch = samples[i:i+120]
                    imgs = [sample[0] for sample in batch]
                    spds = [sample[1] for sample in batch]
                    cmds = [sample[2] for sample in batch]
                    actions = [sample[3] for sample in batch]
                    augmented_imgs = self.augmenter.aug(np.array(imgs))/255
                    spds = np.array(spds) / TARGET_SPEED
                    cmds = np.array(cmds)
                    actions = np.array(actions)
                    data_tuple = ([augmented_imgs, spds, cmds], actions)
                    
                    #self.add_to_cache(([imgs, spds, cmds], actions), is_validation)
                    yield data_tuple
    def add_to_cache(self, data_tuple, is_validation):
        cache = self.val_cache if is_validation else self.cache
        cache['images'] += [sample for sample in data_tuple[0][0]]
        cache['speeds'] += [sample for sample in data_tuple[0][1]]
        cache['commands'] += [sample for sample in data_tuple[0][2]]
        cache['actions'] += [sample for sample in data_tuple[1]]
    def load_data(self,filename, training=True):

        with open(filename, 'rb') as f:
            from pympler.asizeof import asizeof
            print(asizeof(f))
            d = pickle.load(f)
            trim(d)
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
        checkpt = os.path.join(CHECKPT_FOLDER_DIR, "weights.best.testing.225epochs.patience3.batch_size180.validation0.33_1.hdf5")
    
        if len(files) != 1:
            print("no checkpoints saved!")
        else:

            print(f"found checkpoint : {files[0]}")
            checkpt = os.path.join(CHECKPT_FOLDER_DIR, "weights.best.testing.225epochs.patience3.batch_size180.validation0.33_1.hdf5")

            #self.model.load_weights(checkpt)
    
    def create_model(self):
        spd_module = mlp(1, "speed")
        self.spd_module = spd_module
        
        img_module = image_module()
        self.img_module = img_module

        cmd_module = mlp(3, 'command')
        concatenated = concatenate([img_module.image_model_out, spd_module.module_out, cmd_module.module_out]) #TODO fix!!!
        intermediate_layer = add_fc_block(concatenated, 512, "intermediate_layer") 
        #one for each command type
       
        ac_module = action_module(intermediate_layer)
        return Model([self.img_module.image_model_in, self.spd_module.module_in, cmd_module.module_in], ac_module.action_module_out)

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
        predictions = self.model.predict([np.array(self.test_images), np.array(self.test_speeds), np.array(self.test_cmds)])
        acc = self.model.evaluate(x=[np.array(self.test_images), np.array(self.test_speeds), np.array(self.test_cmds)], y=np.array(self.test_actions))

        return
        
    def show_plots(self, history):
        #accuracy
        history = history.history
        plt.plot(history['mse'])
        plt.plot(history['val_mse'])
        plt.title('model mean sq err')
        plt.ylabel('mse')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        #loss
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='right')
        plt.show()
    def get_train_samples(self, filenames, size=1):
    
        numSamplesPerFile = 200
        images = []
        actions = []
        commands = []
        speeds = []
        vectorised_commands = np.eye(4).astype('uint8')
        #follow lane [1,0,0,0]
        #left [0,1,0,0]
        #straight [0,0,1,0]
        #right [0,0,0,1]
        num_train_samples = len(self.train_images)

        i = 0
        while i < int(BATCH_SIZE * 200 * size):
            
            sample_idx = random.randint(0, numSamplesPerFile - 1)
            try:
                data = h5py.File(random.choice(filenames), 'r') 
                images.append(data['rgb'][sample_idx])
                idx = int(data['targets'][sample_idx][24] - 2)
                cmd = vectorised_commands[idx]
                commands.append(cmd)
                speeds.append(data["targets"][sample_idx][10])
                act = data["targets"][sample_idx][:3]
                actions.append(data["targets"][sample_idx][:3])

            except Exception as e:

                self.num_errors += 1
                print(self.num_errors)
                continue
            finally:
                data.close()
            i += 1

                #batch_x is image
                #batch_y steer throttle brake 
                #batch_s speed
        images= np.array(self.augmenter.aug(images))/255
        return ([images, np.array(speeds)/TARGET_SPEED, np.array(commands)], np.array(actions))
    
    def train(self):
        # num_samples = round(TRAIN_BATCH_SIZE / 3)
        
        valPath =  "dataset\SeqVal\\"
        trainPath = "dataset\SeqTrain\\"
        trainFiles = glob.glob(trainPath + "*.h5")
        valFiles = glob.glob(valPath + "*.h5")
        # self.samples = self.get_samples(trainFiles,600)#131520)
        # self.val_samples = self.get_samples(valFiles,600)#15520)
        n_train_samples=131520 * 3
        n_val_samples =15520#2880
        train_search_indices = []
        
        val_data = self.get_samples(valFiles, n_val_samples, dict())
        val_x = [np.array([sample[0] for sample in val_data]), np.array([sample[1] for sample in val_data]), np.array([sample[2] for sample in val_data])]
        val_y = np.array([sample[3] for sample in val_data])
        histories = []
        import time
        s = time.time()
        samples = self.get_samples(trainFiles, n_train_samples, dict())
    
        history = self.model.fit(Generator(samples), steps_per_epoch=n_train_samples//120, batch_size=120, epochs=10, shuffle=True, callbacks=[self.checkpoint, self.early_stopping], validation_data=(val_x, val_y))
           
    
        self.show_graph([history], time.time() - s)
        # history = self.model.fit_generator(self.batch_generator(False, n_train_samples), steps_per_epoch=n_train_samples / 120,
        #     validation_steps=n_val_samples / 120, epochs=5, validation_data=self.batch_generator(True, n_val_samples),
        #     callbacks=[self.early_stopping, self.checkpoint])
        # self.show_plots(history)
        
        #self.model.fit(x=train_samples[0], y=train_samples[1], batch_size=120, shuffle=True, validation_data=val_samples, epochs=50, callbacks=[self.early_stopping])
        print("num errors" + str(self.num_errors))
       
    
    def _show_graph(self, history, name):
        acc = [point for point in history.history[name]]
        acc_val = [point for point in history.history["val_"+name]]
        plt.plot(acc)
        plt.plot(acc_val)
        plt.title('model ' + name)
        plt.ylabel(name)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        
    def show_graph(self, histories, duration):
        plt.title(f'{round(duration/3600, 2)} hrs' )
        self._show_graph(histories[0], 'accuracy')
        plt.figure()
        plt.title(f'{round(duration/60, 2)} hrs' )
        self._show_graph(histories[0], 'loss')
        plt.figure()
        plt.title('hrs' )
        self._show_graph(histories[0], 'mse')
        plt.show()
    def normalise_single_sample(self, image, speed, cmd, grayscale= False):
        image = np.reshape(image, (IM_HEIGHT, IM_WIDTH, 1 if grayscale else 3)) / 255
        vec_commands = np.eye(3).astype('uint8')
        speed /= TARGET_SPEED

        return image, speed, vec_commands[cmd-2]

    def normalise_samples(self,  images, speeds, commands, grayscale=False):
       
        normalised_cmds = []
        
        images = np.array(images) / 255 if images[0].dtype != 'float' else np.array(images)
        speeds = np.array(speeds) / TARGET_SPEED
        left_cmd = [0,1,0]
        right_cmd = [0,0,1]
        straight_cmd = [1,0,0]
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


    def get_action(self, image, speed, command, grayscale = False):
        #images is still not an ndarray at this time
        
        image, speed, cmd = self.normalise_single_sample(image,speed,command)
        image = np.reshape(image, (1,88,200,3))
        speed = np.reshape(speed, (1,1))
        cmd = np.reshape(cmd, (1,3))
       
        s,t,b= self.model.predict([image, speed, cmd], 1, verbose='0')[0]
        return (s.item(), t.item(), b.item())

    def run_step(self, measurements, sensor_data, directions, target):

        s,t,b = self.get_action(sensor_data['CameraRGB'].data,
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
        temp2  = Dense(1, activation=None, use_bias=False)(temp)
        self.model1 = Model(inLayer, temp)
        f=r"checkpoints\test_weights.hdf5"
        self.model2 = Model(inLayer, temp2)
        self.model1.compile(loss='mse', optimizer='adam', 
metrics=['accuracy'])
        self.model2.compile(loss='mse', optimizer='adam', metrics='accuracy')
        
    def train(self):
        print(self.model1.summary())

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
        chkpt = ModelCheckpoint(os.path.join(CHECKPT_FOLDER_DIR, "test_weights.hdf5"), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        print(self.model2.get_weights()[0])

        self.model1.fit(x=x, y=y, batch_size=120, validation_split=0.2, epochs=1, verbose='0', shuffle=True, callbacks=[chkpt])
        
        print(self.model2.get_weights()[0])
        print(self.model1.get_weights()[0])
        print()

def trim(data):
    for i in range(4):
        data[i] = data[i][:8404]

# #print(agt.get_single_action(image, speed,command))

#test_load_data()
# test_evaluate(agt)
#test_evaluate(agt)
#test_show_graph()
#test_insert()
             #verbose=1, shuffle=True)
    

#callbacks=[early_stopping, checkpoint])