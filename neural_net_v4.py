


from matplotlib import pyplot as plt
from constants import BATCH_SIZE, DEBUG_MAX_REPLAY_BUFFER_SIZE, MAX_REPLAY_BUFFER_SIZE, IM_HEIGHT, IM_WIDTH, MINI_BATCH_SIZE, NUM_AGENT_TRAIN_STEPS_PER_ITER, NUM_EPOCHS, STORAGE_LIMIT, TARGET_SPEED
import h5py
from collections import deque 
from random import sample, shuffle
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Flatten, Conv2D, Reshape, concatenate, Multiply, Activation, Lambda
from keras.losses import MeanSquaredError
from keras import initializers
import numpy as np
import os
import cv2
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import carla
import glob
from load_data_v2_mask import *
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
#https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
#apply dropout after activation https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
#source that recommend he_uniform kernel initializer, instead of the default glorot uniform initializer (when using relu as activation)

#https://stackoverflow.com/questions/54011173/is-there-any-documentation-about-default-weight-initializer-in-keras

#source that recommend a bias of 0.1 
#https://github.com/carla-simulator/imitation-learning/blob/62f93c2785a2452ca67eebf40de6bf33cea6cbce/agents/imitation/imitation_learning_network.py#L13


from image_augmenter import image_augmenter

#dog_dir = os.listdir(r".\PetImages\Dog")
CHECKPT_FOLDER_DIR = r"C:\Users\autpucv\Desktop\my scripts\imitation_learning\checkpoints"#r".\checkpoints"
NUM_TRAIN_SAMPLES = 74800


def custom_mse(y_true, y_pred):
    # mask = tf.zeros(3)
    # l_and = tf.logical_and
    # l_not = tf.logical_not
    # reduce = tf.math.reduce_all
    
    # m=l_not(l_and(reduce(tf.equal(y_true,mask),axis=1), reduce(K.equal(y_pred, mask),axis=1)))
    # mse = MeanSquaredError()(y_true[m], y_pred[m])
    return MeanSquaredError()(y_true, y_pred) * 3

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
        temp = add_conv_block(temp, 256, 3, 1, "image_module_l8") 
        temp = add_conv_block(temp, 256, 3, 1, "image_module_l9") 
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
    def __init__(self, base_layer :Activation, idx):
        print("aciton module input")
        temp = add_fc_block(base_layer, 256, f"action_l1_{idx}")
        self.ac_in = temp
        temp = add_fc_block(temp, 256, f"action_l2_{idx}")
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
import keras.backend as K
sys.path.insert(0, r"C:\Users\autpucv\Downloads\CARLA_0.8.2\PythonClient\carla\agent")
#from agent import Agent

class LossHistory(Callback):
    
    def __init__(self, best=np.inf):
        super().__init__()
        self.best_val_loss = best
    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        if logs['val_loss'] < self.best_val_loss:
            self.best_val_loss = logs['val_loss']
    
class Generator(Sequence):
    
    # Class is a dataset wrapper for better training performance
    def __init__(self, dataset, batch_size=BATCH_SIZE, is_val=False, num_train_samples=None):
        self.is_val = is_val
        self.dataset = dataset
        self.batch_size = batch_size // 3 
        self.indices = [i for i in range(len(dataset[0]))]
        random.shuffle(self.indices) # tuples containing (img, ..., act)
        self.augmenter = image_augmenter()
        self.num_steps = 0

        self.num_train_samples = num_train_samples
        if is_val:
            self.min_subset_size = min([len(subset ) for subset in self.dataset])
        
    def __len__(self):
        #each branch will fit 120, and there are 4 batches
        if not self.is_val:
            return len(self.indices) // self.batch_size
        else:
            #the entire val set counts as batch
            return 1
    def __getitem__(self, idx):
        ##########
        zero_mask = [0.0, 0.0, 0.0]
        one_mask = [1.0, 1.0,1.0]
        mask = [[one_mask, zero_mask, zero_mask],
                [zero_mask, one_mask, zero_mask],
                [zero_mask, zero_mask, one_mask]]
      
        ##################
        samples = []
        
        for subset in self.dataset:
            if self.is_val:
                
                samples += subset[:min(int(self.num_train_samples * 0.33 // 3), self.min_subset_size)]
            else:
                
                samples += subset[idx*self.batch_size : (idx + 1) * self.batch_size]
                random.shuffle(samples)
        #normlisation
        batch_img = self.augmenter.aug(np.array([sample[0] for sample in samples]))/255
        batch_speeds = np.array([sample[1] for sample in samples]) / TARGET_SPEED
        batch_cmds = [sample[2] for sample in samples]
        batch_masks = np.array([mask[cmd - 2] for cmd in batch_cmds])

        # actions = [batch_masks[:,i,:].reshape((-1,3)) for i in range(4) ]
            
        batch_actions = [sample[3] for sample in samples]
        # x = []
        # follow_lane_actions = np.array([action[0] if cmd == 2 else zero_mask for action, cmd in zip(batch_actions, batch_cmds)])
        # left_actions = np.array([action[1] if cmd == 3 else zero_mask for action, cmd in zip(batch_actions, batch_cmds)])
        # right_actions = np.array([action[2] if cmd == 4 else zero_mask for action, cmd in zip(batch_actions, batch_cmds)])
        return ([batch_img, batch_speeds, batch_masks], 
        [*(np.array([action if cmd == current_cmd else zero_mask 
        for action, cmd in zip(batch_actions, batch_cmds)]) 
        for current_cmd in range(2, 5))])
    
    def on_epoch_end(self):        
        for subset in self.dataset:
            random.shuffle(subset)

class agent: 
    
    def __init__(self, debug_level=None, train_init_policy=False, rl=False):
        self.histories = []
        self.debug_level = debug_level
        self.suggested_actions = []    
        self.num_errors = 0
        self.num_inputs_added = 0
        self.train_samples = [[], [], []]
        self.val_data = []
        self.minimum_loss = np.inf
        self.loss_hist = LossHistory(self.minimum_loss)

        self.n_times_val_loss_no_update = 0
        self.patience = 2
        if not rl:
            self.val_data = load_data_2(False, debug_level)#self.get_samples(valFiles, n_val_samples, dict())
            if train_init_policy:
                self.train_samples = load_data_2(debug_level=debug_level)
        
        self.checkpoint = ModelCheckpoint(os.path.join(CHECKPT_FOLDER_DIR, ("weights_multi_branch.hdf5" if not self.debug_level else "weights_multi_branch_debug.hdf5")), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        self.early_stopping = EarlyStopping(monitor ="val_loss", restore_best_weights = True, mode ="min", patience = 3)
                 
                    
        self.augmenter = image_augmenter()                            
        self.model = self.create_model()
        
        print("loading weights...")
    
        self.try_load_model()
        
        opt = Adam(lr=1e-4)
        #large errors are heavily penalised with the mean squared error
        #tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()
        self.model.compile(loss=custom_mse, optimizer=opt, metrics=[custom_mse])
        self.split = math.ceil(MAX_REPLAY_BUFFER_SIZE * 0.2)
    
    
        
        
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
        vectorised_commands = np.eye(4).astype('uint8')
    
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
    def try_load_model(self):
        checkpt = os.path.join(CHECKPT_FOLDER_DIR, "weights_multi_branch.hdf5")
        
        if os.path.exists(checkpt):
            try:
                f = open("min_loss.txt", 'r')
                self.minimum_loss = int(f.readline())
            except:
                print("cannot read minloss")
            finally:
                f.close()
            print(f"found file {checkpt}")
            self.model.load_weights(checkpt)
        else:
            print("cannot find" + checkpt)
        #self.model = tf.keras.models.load_model(CHECKPT_FOLDER_DIR +r"\weights_multi_branch")
        #self.model.save_weights(CHECKPT_FOLDER_DIR + r"\weights_multi_branch.hdf5")
        #self.model.load_weights(CHECKPT_FOLDER_DIR + r"\weights_multi_branch")
    def try_load_weights(self):

        files = os.listdir(CHECKPT_FOLDER_DIR)
        checkpt = os.path.join(CHECKPT_FOLDER_DIR, "weights.best.testing.225epochs.patience3.batch_size180.validation0.33_1.hdf5")
    
        if len(files) != 4:
            print("no checkpoints saved!")
        else:
            
            print(f"found checkpoint : {files}")
            for file, model in zip(sorted(files, key=lambda item: glob_sorter(item, 1)), model):
                model.load_weights(os.path.join(CHECKPT_FOLDER_DIR, file))
    
    def create_model(self):
        masks = Input((3, 3))
        spd_module = mlp(1, "speed")
        self.spd_module = spd_module
        
        img_module = image_module()
        self.img_module = img_module

        #cmd_module = mlp(4, 'command')
        concatenated = concatenate([img_module.image_model_out, spd_module.module_out]) #TODO fix!!!
        intermediate_layer = add_fc_block(concatenated, 512, "intermediate_layer") 
        #one for each command type
        models = []
        masked_outputs = []
        for i in range(3):
            ac_module = action_module(intermediate_layer, i)
            print(ac_module.action_module_out.shape)
            print(masks.shape)
            
            branch_masks = Flatten()(masks[:, i, :])
            print(branch_masks.shape)
            masked_outputs.append(Multiply()([ac_module.action_module_out, branch_masks]))

        return Model([img_module.image_model_in, spd_module.module_in, masks], masked_outputs)

    def _pop_from_buffer(self):
        self.train_images.remove(self.train_images[0])
        self.train_cmds.remove(self.train_cmds[0])
        self.train_speeds.remove(self.train_speeds[0])
        self.train_actions.remove(self.train_actions[0])

    @property
    def MAX_REPLAY_BUFFER_SIZE(self):
        return MAX_REPLAY_BUFFER_SIZE if not self.debug_level else DEBUG_MAX_REPLAY_BUFFER_SIZE

    def _add_to_replay_mem(self, img, speed, cmd, action):
        
        self.train_samples[cmd - 2].append((img, speed, cmd, action))
        
    
    def insert_input(self, image, speed, cmd, action):
    
        if len(self.train_samples[cmd - 2]) == self.MAX_REPLAY_BUFFER_SIZE // 3:
            
            self.train_samples[cmd - 2].remove(self.train_samples[cmd -2][0])
    
        self._add_to_replay_mem(image, speed, cmd, action)
    
    

    def calculate_min_buffer_size(self, per_sample_size = 62e3):
        return round(STORAGE_LIMIT * 1e9 / per_sample_size / 3)
    
    def evaluate(self):
        files = os.listdir(r'.\recordings\testing')
        self.load_data(r'.\recordings\testing\recording-1-p2.pkl', training=False)
        
        l = len(self.test_images)
        predictions = self.model.predict([np.array(self.test_images), np.array(self.test_speeds), np.array(self.test_cmds)])
        acc = self.model.evaluate(x=[np.array(self.test_images), np.array(self.test_speeds), np.array(self.test_cmds)], y=np.array(self.test_actions))

        return
    def _show_plots(self, histories, metric):
        num_plots = len(histories) * 2
        for i in range(3):
            plt.figure()
            plt.title(f"branch {i} metric")

            for history in histories:
                history = history.history
                if i > 0:
                    plt.plot(history[f'multiply_{i}_{metric}'])
                    plt.plot(history[f'val_multiply_{i}_{metric}'])
                else:
                    plt.plot(history[f'multiply_{metric}'])
                    plt.plot(history[f'val_multiply_{metric}'])
    
            plt.ylabel(metric)
            plt.xlabel('epoch')
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.xticks(np.arange(0, self.NUM_EPOCHS, 1.0))
            plt.legend([f'train{i // 2}' if i % 2 == 0 else f'val{i // 2}' for i in range(num_plots)], loc='upper left')
            plt.savefig(f"saved_graphs_multi_branch\\branch_{i}_{metric}.png", )
    @property
    def NUM_EPOCHS(self): return 3
    def show_average_plots(self, histories, metric):
        plt.figure()
        plt.title(f"average {metric}")
        num_plots = len(histories) * 2
        for i, history in enumerate(histories):
            history = history.history
            val_histories = [history[f'val_multiply_{metric}']] + [history[f'val_multiply_{i}_{metric}'] for i in range(1,3)]
            train_histories = [history[f'multiply_{metric}']] + [history[f'multiply_{i}_{metric}'] for i in range(1,3)]
            val_average = np.average(val_histories, axis=0)
            train_average = np.average(train_histories, axis=0)
            plt.plot(val_average)
            plt.plot(train_average)
        plt.xticks(np.arange(0, self.NUM_EPOCHS, 1.0))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend([f'train{i // 2}' if i % 2 == 0 else f'val{i // 2}' for i in range(num_plots)], loc='upper left')
        plt.savefig(f"saved_graphs_multi_branch\\average_{metric}.png")

    
    def show_plots(self, histories):
        #accuracy
        self._show_plots(histories, "custom_mse")
        self.show_average_plots(histories, 'custom_mse')
        #plt.figure()
       # self._show_plots(histories, "accuracy")
        #plt.show()
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
    @property
    def BATCH_SIZE(self):
        if self.debug_level == 1:
            return DEBUG_BATCH_SIZE
        elif self.debug_level == 0:
            return 9
        else:
            return BATCH_SIZE
    def train(self):
        
        best_val_loss_so_far = self.minimum_loss
        history = self.model.fit(Generator(self.train_samples, self.BATCH_SIZE), batch_size=self.BATCH_SIZE, validation_data=Generator(self.val_data, is_val=True, num_train_samples=sum([len(samples) for samples in self.train_samples])), epochs=self.NUM_EPOCHS,
         callbacks=[self.checkpoint, self.loss_hist, self.early_stopping])
        self.minimum_loss = self.loss_hist.best_val_loss
        if self.model.stop_training:
            
            print(f"stopped epoch: {self.early_stopping.stopped_epoch}")
            print(f"current best: {self.early_stopping.best}")
            print(f"baseline: {self.early_stopping.baseline}")
            print(f"current best: {self.loss_hist.best_val_loss}")
            
            
            self.model.stop_training = False
        self.early_stopping = EarlyStopping(monitor ="val_loss",
                                        restore_best_weights = True,
                                        mode ="auto", patience=3, baseline=self.minimum_loss)
        #if two values are the same it means the model has not improved at all
        if self.minimum_loss == best_val_loss_so_far:
            self.n_times_val_loss_no_update += 1
            if self.n_times_val_loss_no_update == self.patience:
                return False
        else:
            if not self.debug_level:
                with open('min_loss.txt', 'w') as f:
                    f.write(str(self.minimum_loss))
                    
        self.histories.append(history)
        #self.show_plots(history) 

        #self.show_graph([history], time.time() - s)
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
        # plt.title(f'{round(duration/3600, 2)} hrs' )
        # self._show_graph(histories[0], 'accuracy')
        # plt.figure()
        self._show_graph(histories[0], 'loss')
        
        self._show_graph(histories[0], 'mse')
        plt.show()
    def normalise_single_sample(self, image, speed, cmd, grayscale= False):
        ones_mask = [1.0,1.0,1.0]
        zeros_mask = [0.0,0.0,0.0]
        mask_types = [[ones_mask, zeros_mask, zeros_mask, zeros_mask], 
                    [zeros_mask, ones_mask, zeros_mask, zeros_mask],
                    [zeros_mask, zeros_mask, ones_mask, zeros_mask],
                    [zeros_mask, zeros_mask, zeros_mask, ones_mask]]

        image = np.reshape(image, (1, IM_HEIGHT, IM_WIDTH, 1 if grayscale else 3)) / 255
        speed /= TARGET_SPEED
        speed = np.array([speed])
        mask = np.array([mask_types[cmd - 2]])
        return image, speed, mask

    def normalise_samples(self,  images, speeds, commands, grayscale=False):
       
        normalised_cmds = []
        #######masks##########
        full_mask = np.ones((3,))
        empty_mask = np.zeros((3,))
        mask_types = [[full_mask, empty_mask, empty_mask, empty_mask], 
                    [empty_mask, full_mask, empty_mask, empty_mask],
                    [empty_mask, empty_mask, full_mask, empty_mask],
                    [empty_mask, empty_mask, empty_mask, full_mask]]
        images = np.array(images) / 255 if images[0].dtype != 'float' else np.array(images)
        speeds = np.array(speeds) / TARGET_SPEED
       
        left = 0
        right = 0
        straight = 0
        masks = np.array([mask_types[cmd - 2] for cmd in commands])
        return images, speeds, masks
        # for img,speed,command in zip(images, speeds, commands):
        #     img, speed, command = self.normalise_single_sample(img, speed, command)
        #     normalised_imgs.append(img)
        #     normalised_speeds.append(speed)
        #     normalised_cmds.append(command)
        #return np.array(normalised_imgs), np.array(normalised_speeds), np.array(normalised_cmds)


    def get_action(self, image, speed, command, grayscale = False):
        #images is still not an ndarray at this time
        
        image, speed, mask = self.normalise_single_sample(image,speed,command)
        
        s,t,b = self.model.predict([image, speed, mask], 1, verbose='0')[command - 2][0]
        #s,t,b = pred[command-2]
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
    agt = agent(debug_level=0, train_init_policy=False)
    
    for i in range(81):
        
        img = np.random.uniform(0, 255, (88, 200, 3)).astype('uint8')
        speed = 29
        cmd = i % 3 + 2 
      
        action = np.random.uniform(-1, 1, (3,)).tolist()

        agt.insert_input(img, speed, cmd, action)

    agt.train()
def test_load_data():
    pass
#agent(1, True).train()

test_insert()


