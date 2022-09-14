import os
import tensorflow as tf
from constants import IM_HEIGHT, IM_WIDTH

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Reshape, concatenate
import numpy as np
from neural_net_v2 import *

TARGET_SPEED = 25
def normalize(state):
    
    
    all_cmds = tf.convert_to_tensor(np.eye(4), tf.float32)
    images, speeds, cmd_indices = state
    cmd_indices = tf.cast(cmd_indices, tf.int32)
    images /= 255
    speeds /= TARGET_SPEED
    
    vec_cmds = tf.stack([all_cmds[idx - 2] for idx in cmd_indices], 0)

    return images, speeds, vec_cmds

def create_model(critic, fc1_dims=512, fc2_dims=512):
    img_module = image_module()

    cmd_module = mlp(4, 'cmd')
    spd_module = mlp(1, 'speed')
   
    concat_layer = concatenate([img_module.module_out, spd_module.module_out, cmd_module.module_out])
    act_module = None

    if critic:
        act_module = mlp(3, "action")
        concat_layer = concatenate([concat_layer, act_module.module_out])
    out = add_fc_block(concat_layer, fc1_dims, 'out0')
    out = add_fc_block(out, fc2_dims, 'out1')

    #all outputs are non-negative real numbers at this point
    if not critic:
        steer = Dense(1, activation='tanh')(out) #tf.tanh(out[:, :1])
        throttle_brake = Dense(2, activation='sigmoid')(out)
        out = tf.concat([steer, throttle_brake], axis=1)
        return Model([img_module.module_in, spd_module.module_in, cmd_module.module_in], out)
    else:
        out = Dense(3, activation='linear')(out)
        return Model([img_module.module_in, spd_module.module_in, cmd_module.module_in, act_module.module_in], out)
    

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512,
            name='critic', chkpt_dir='DDPG\checkpoints'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_ddpg.h5')
        
        self.model = create_model(True)
    def call(self, state, action):
        state = normalize(state)
        return self.model([*state, action])
        # action_value = concatenate([self.image_model(state[0]), self.spd_module(state[1]), self.cmd_module(state[2]), action])
        # action_value = add_fc_block(action_value, 512, 'fc_1')
        # action_value = add_fc_block(action_value, 512, 'fc_2')
        # q = self.q(action_value)

        # return q

class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, n_actions=2, name='actor',
            chkpt_dir='DDPG\checkpoints'):
        super(ActorNetwork, self).__init__()
        self.model = create_model(False)#agent(train_initial_policy=True, rl=True).create_model()
        
        # self.fc1_dims = fc1_dims
        # self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_ddpg.h5')

        # self.fc1 = Dense(self.fc1_dims, activation='relu')
        # self.fc2 = Dense(self.fc2_dims, activation='relu')
        # self.mu = Dense(self.n_actions, activation='tanh')
    # def set_weights(self, weights):
    #     self.model.set_weights(weights)
    # def get_weights(self):
    #     return self.model.get_weights()
     
    def call(self, state):
        state = normalize(state)
        
        return self.model([*state ])