from keras.layers import Dense, Conv2D, Input,Flatten
from keras.models import Model
from keras.optimizers import Adam
import math
import numpy as np
import pickle
from constants import IM_WIDTH, IM_HEIGHT

#input dim = 88 x 200

class agent:
    def __init__(self):
    
        net_in = Input((IM_HEIGHT, IM_WIDTH, 3))
        temp = Conv2D(24, 5, (2,2))(net_in)
        temp = Conv2D(36, 5, (2,2))(temp)
        temp = Conv2D(48, 3, (2,2))(temp)
        temp = Conv2D(64, 3, (2,2))(temp)
    
        temp = Flatten()(temp)
        temp = Dense(1164)(temp)
        temp = Dense(100)(temp)
        temp = Dense(50)(temp)
        temp = Dense(10)(temp)
        temp = Dense(1, activation='linear')

        self.model = Model(net_in, temp)
        opt = Adam()
        self.model.compile()
    
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
    
    def load_data(self, filename, training=True):
        

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