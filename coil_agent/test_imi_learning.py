
import numpy as np
import imitation_learning
agent = imitation_learning.ImitationLearning('', False)
img = np.random.uniform(0, 255, (600, 800, 3)).astype('uint8')
speed = 20

dir = 2
print(agent.get_action(img, speed, dir))
