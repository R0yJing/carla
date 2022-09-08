import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_dims, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.image_state_memory =[] #np.zeros((self.mem_size, *input_dims[0]))
        self.speed_state_memory = [] #np.zeros((self.mem_size, *input_dims[1]))
        self.command_state_memory = [] #np.zeros((self.mem_size, *input_dims[2]))
        self.new_image_state_memory = [] #np.zeros((self.mem_size, *input_dims[0]))
        self.new_speed_state_memory = [] #np.zeros((self.mem_size, *input_dims[1]))
        self.new_command_state_memory = [] #np.zeros((self.mem_size, *input_dims[2]))
        
        self.action_memory = []#np.zeros((self.mem_size, n_actions))
        self.reward_memory = []#np.zeros(self.mem_size)
        self.terminal_memory = []#np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        #index = self.mem_cntr % self.mem_size
        self.image_state_memory.append(state[0])
        self.new_image_state_memory.append(state_[0])
        self.speed_state_memory.append(state[1])
        self.new_speed_state_memory.append(state_[1])
        
        self.command_state_memory.append(state[2])
        self.new_command_state_memory.append(state_[2])
        
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.terminal_memory.append(done)

        self.mem_cntr += 1
    def get_samples(self, memory, batch):
        return np.array([memory[idx] for idx in batch])
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = (self.get_samples(self.image_state_memory, batch), self.get_samples(self.speed_state_memory, batch), self.get_samples(self.command_state_memory, batch))
        states_ = (self.get_samples(self.image_state_memory, batch), self.get_samples(self.speed_state_memory, batch), self.get_samples(self.command_state_memory, batch))
        actions = np.array([self.action_memory[idx] for idx in batch])
        rewards = np.array([self.reward_memory[idx] for idx in batch])
        dones = np.array([self.terminal_memory[idx] for idx in batch])

        return states, actions, rewards, states_, dones