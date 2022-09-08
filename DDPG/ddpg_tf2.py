import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from .buffer import ReplayBuffer
from .networks import ActorNetwork, CriticNetwork
from .utils import *
class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None,
                 gamma=0.99, n_actions=3, max_size=1000000, tau=0.005,
                 fc1=400, fc2=300, batch_size=66, noise=0.1, load_checkpoint=False):
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
        
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.built = True
        self.actor.built = True
        self.target_critic.built = True
        self.critic.built = True

        self.update_network_parameters(tau=1)
        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.get_weights()
        for i, weight in enumerate(self.actor.get_weights()):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)
        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)
        w = self.target_actor.get_weights()
        w1 = self.target_actor.model.get_weights()
        for a0, a1 in zip(w, w1):
            if (a0 != a1).any():
                print("weights not equal")
    
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
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)
    def get_action(self, img, spd, cmd):

        act = (self.choose_action((img, spd, cmd), True)).numpy()
        
        return act[0].item(), act[1].item(), act[2].item()
        

    def choose_action(self, observation, evaluate=False):
        #state = tf.convert_to_tensor([observation], dtype=tf.float32)
        #call method will autom normalize the ob
        img, spd, cmd = observation
        actions = self.actor((tf.expand_dims(img, 0), tf.expand_dims(spd, 0), tf.expand_dims(cmd, 0)))
        
        _, _, cdir = observation
        if not evaluate:
            pass
            # actions += tf.random.normal(shape=[self.n_actions],
            #                             mean=0.0, stddev=self.noise)
        
        # note that if the env has an action > 1, we have to multiply by
        # max action at some point

        min_action = (-1, 0, 0)
        max_action = (1, 1, 0)
        # if cdir == 3:
        #     #left 
        #     min_action = (0, 0, 0)
        #     max_action = (1, 1, 0)
        # elif cdir == 4:
        #     #right
        #     min_action = (0, 0, 0)
        #     max_action = (-1, 1, 0)
        # else:
        #     #straight
        #     min_action = (-1, 0, 0)
        #     max_action = (1, 1, 0)
        
        actions = tf.clip_by_value(actions, min_action, max_action )
    

        return actions[0]

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
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(
                                states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma*critic_value_*(1-done)
            critic_loss = keras.losses.MSE(target, critic_value)
    
        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(state)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()