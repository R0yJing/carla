import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from .buffer import ReplayBuffer
from .networks import ActorNetwork, CriticNetwork
from .utils import *
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.0001,
                 gamma=0.99, n_actions=3, max_size=1000000, tau=0.005,
                 fc1=400, fc2=300, batch_size=128, noise=0.1, load_checkpoint=False):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        #steer throttle brake 
        self.n_actions = n_actions
        self.noise = noise
        
        self.actor = ActorNetwork(n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(name='critic')

        self.target_actor = ActorNetwork(n_actions=n_actions,
                                         name='target_actor')
        self.target_critic = CriticNetwork(name='target_critic')
        #actor should learn faster than critic as more accurate critic network
        #should be used later during training

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.built = True
        self.actor.built = True
        self.target_critic.built = True
        self.critic.built = True
        self.r_plot = []
        self.update_network_parameters(tau=1)
        if load_checkpoint:
            self.load_models()
        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.get_weights()
        print("actor weight set")
        for i, weight in enumerate(self.actor.get_weights()):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)
        weights = []
        targets = self.target_critic.weights
        print("critic weight set")
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        
        self.target_critic.set_weights(weights)
       
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        try:
            self.actor.load_weights(self.actor.checkpoint_file)
            self.target_actor.load_weights(self.target_actor.checkpoint_file)
            self.critic.load_weights(self.critic.checkpoint_file)
            self.target_critic.load_weights(self.target_critic.checkpoint_file)
        except:
            print("cannot load model!")
    def get_action(self, img, spd, cmd):
        
        act = self.choose_action((img, spd, cmd), True)
        
        return act[0].item(), act[1].item(), act[2].item()
    
   

    def choose_action(self, observation, evaluate=False):
        #state = tf.convert_to_tensor([observation], dtype=tf.float32)
        #call method will autom normalize the ob
        img, spd, cmd = observation
        actions = self.actor((tf.expand_dims(img, 0), tf.expand_dims(spd, 0), tf.expand_dims(cmd, 0)))
        action = actions[0].numpy()
        action[2] = 0
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        print("learning")
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)
        img, speed, cmd = state
        new_img, new_speed, new_cmd = new_state
        states = (tf.convert_to_tensor(img, dtype=tf.float32), tf.convert_to_tensor(speed, dtype=tf.float32), tf.convert_to_tensor(cmd, dtype=tf.float32))
        states_ = (tf.convert_to_tensor(new_img, dtype=tf.float32), tf.convert_to_tensor(new_speed, dtype=tf.float32), tf.convert_to_tensor(new_cmd, dtype=tf.float32))
        rewards = tf.reshape(tf.convert_to_tensor(reward, dtype=tf.float32), (-1, 1))
        
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        dones = done.reshape((-1, 1))
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = self.target_critic(
                                states_, target_actions)
            critic_value = self.critic(states, actions)

            target = rewards + self.gamma*critic_value_*(1-dones)
            critic_loss = tf.reduce_mean(keras.losses.MSE(target, critic_value))
         
    
        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        print(critic_network_gradient)
        #trainable variables is the set of weights and biases
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(state)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()
# def is_num(o):
#     return type(o) == int
# import keras.backend as K
# import numpy as np
# import tensorflow as tf
# def mse(X, Y):
#     if is_num(X[0]):
#         return sum([(x - y)**2 for x, y in zip(X, Y)]) / len(X)
#     else:

#         return K.mean(np.array([mse(x, y) for x, y in zip(X, Y)]))
    
# def mse(X, Y):
#     if is_num(X[0]):
#         return sum([(x - y)**2 for x, y in zip(X, Y)]) / len(X)
#     else:

#         return tf.reduce_mean(np.array([mse(x, y) for x, y in zip(X, Y)]))
# # x=[[[[3,2],[2, 4],[1, 5]], [[7,6],[7,2],[7,1]]]] 
# # y=[[[[8,5],[2, 8],[8, 1]], [[7,6],[3,8],[9,-1]]]]
# # print(mse(x, y))

