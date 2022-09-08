datadir = r"coiltrain_dataset"
import glob
import json
#get left
import numpy as np
import os
from lateral_augmentations import *
import random
import imageio
from send2trash import send2trash
from copy import copy
# num_train_files = 22110 * 3
# num_val_files = 29160
# 

folders = os.listdir(datadir)
DEBUG_BATCH_SIZE = 120
vectorised_commands = np.eye(4).astype('uint8')
import numpy as np
errors = 0
#TOTAL_NUM_SAMPLES =  * 4 # 11670 * 4
full_mask = [1.0,1.0,1.0]
empty_mask = [0.0,0.0,0.0]
mask_types = [[full_mask, empty_mask, empty_mask], 
                    [empty_mask, full_mask, empty_mask],
                    [empty_mask, empty_mask, full_mask]]
def glob_sorter(item:str, len=5):
    idx = item.index('.')
    return int(item[idx-len:idx])
def get_cmd(data):
    if int(data['directions']) == 5:
        return 2
    else: return int(data['directions'])

def load_data(load_train=True, debug_level=1):
    commands_ct = [0,0,0,0]

    images = []
    measurements = []
    commands = []
    actions = []
    fin = False
    global errors, num_train_files, num_val_files, DEBUG_BATCH_SIZE, mask_types
    num_train_files = 22116 * 3
    num_val_files = np.inf
 
    if debug_level == 0:
        num_train_files = 9
        num_val_files = 9
    elif debug_level == 1:
        num_samples  =DEBUG_BATCH_SIZE * 3
        images = [np.random.uniform(0, 255, (88, 200, 3)).astype('uint8') for _ in range( num_samples)]
        speeds = np.random.uniform(0, 30, (num_samples,)).tolist()
        commands = [2 for i in range(DEBUG_BATCH_SIZE)] + [3 for i in range(DEBUG_BATCH_SIZE)] + [4 for i in range(DEBUG_BATCH_SIZE)]
        actions = np.random.uniform((DEBUG_BATCH_SIZE, 3)).tolist()
        return images, speeds, commands, actions

    folder_names = []
    if not load_train:
        folder_names = folders[1:]
    else:
        folder_names = folders[:1]
    
    for folder in folder_names:
        if fin: break
        subfolders = os.listdir(datadir + "\\" + folder)

        for f in subfolders:
           
            measurement_files = sorted(glob.glob(f"{datadir}\\{folder}\\{f}\\measurements*.json"),key=glob_sorter)
            centre_image_files = sorted(glob.glob(f"{datadir}\\{folder}\\{f}\\CentralRGB*.png"),key=glob_sorter)
            left_image_files = sorted(glob.glob(f"{datadir}\\{folder}\\{f}\\LeftRGB*.png"),key=glob_sorter)
            right_image_files = sorted(glob.glob(f"{datadir}\\{folder}\\{f}\\RightRGB*.png"),key=glob_sorter)
           
            
            for measurement, left, centre, right in zip(measurement_files, centre_image_files, left_image_files, right_image_files):
                
                
            
                im_left = imageio.imread(left)
                im_centre = imageio.imread(centre)
                im_right = imageio.imread(right)
                
                try:
                    with open(measurement, 'rb') as fp:
                        
                        data = json.load(fp)
                        
                        steer = data['steer']
                        speed = data['playerMeasurements']['forwardSpeed'] 
                        throttle = data['throttle']
                        brake = data['brake']
                 
                       
                        if load_train and sum(commands_ct) == num_train_files:
                             
                             fin = True
                             break
                        if not load_train and sum(commands_ct) == num_val_files:
                             #collected enough validation samples
                             fin = True
                             break
    
                        if not load_train and commands_ct[get_cmd(data) - 2] >= num_val_files // 3:#TOTAL_NUM_SAMPLES //4:
                            continue
                        elif load_train and commands_ct[get_cmd(data) - 2] >= num_train_files // 3:#TOTAL_NUM_SAMPLES //4:
                            continue
                        
                        commands_ct[get_cmd(data) - 2] += 3
                    
                        images.append(im_centre)
                        images.append(im_left)
                        images.append(im_right) 
                        
                        for i in range(3):
                            measurements.append(speed)
                            commands.append(get_cmd(data))
                        
                        # for i in range(3):
                        #     action = copy(mask_types[int(data["directions"] - 2)])
                            
                        #     actions.append(action)
                        dir = get_cmd(data)
                        actions.append([steer, throttle, brake])
                        actions.append([augment_steering(-45, steer, speed), throttle, brake])
                        actions.append([augment_steering(45, steer, speed), throttle, brake])

                        #with right, steer to the left -ve
                        #steer to the right +ve
                        
                            # print(steer)
                        # print("######################")
                except Exception as e:

                    send2trash(measurement)
                    send2trash(left)
                    send2trash(centre)
                    send2trash(right)
                    errors += 1
                    print(e)

            if fin:
                break
    #clip
    indices = [i for i in range(sum(commands_ct))]
    
        #int(num_train_files * 0.33 // (4 * BATCH_SIZE) * 4 * BATCH_SIZE)

    random.shuffle(indices)
    shuffled_images = [0 for i in indices]
    shuffled_measurements = [0 for i in indices]
    shuffled_cmds =[0 for i in indices]
    shuffled_actions = [0 for i in indices]
    j = 0
    for i in indices:
        shuffled_images[j] = images[i]
        shuffled_measurements[j] = measurements[i]
        shuffled_cmds[j] = commands[i]
        shuffled_actions[j] = actions[i]

        j += 1
    print(commands_ct)
    return shuffled_images, shuffled_measurements, shuffled_cmds, shuffled_actions
def load_data_2(load_train=True, debug_level=1):
    shuffled_images, shuffled_measurements, shuffled_cmds, shuffled_actions = load_data(load_train, debug_level)
    # full_mask = np.ones((3,))
    # empty_mask = np.zeros((3,))
    # mask_types = [[full_mask, empty_mask, empty_mask], 
    #                 [empty_mask, full_mask, empty_mask],
    #                 [empty_mask, empty_mask, full_mask]]
    f_samples = [(img, spd, cmd, act) for img, spd, cmd, act in zip(shuffled_images, shuffled_measurements, shuffled_cmds, shuffled_actions)]
    l_samples = [(img, spd, cmd, act) for img, spd, cmd, act in zip(shuffled_images, shuffled_measurements, shuffled_cmds, shuffled_actions)]
    r_samples = [(img, spd, cmd, act) for img, spd, cmd, act in zip(shuffled_images, shuffled_measurements, shuffled_cmds, shuffled_actions)]
    #s_samples = [(img, spd, mask_types[3], act) for img, spd, cmd, act in zip(shuffled_images, shuffled_measurements, shuffled_cmds, shuffled_actions) if cmd == 3]
#    min_l = min(len(f_samples), len(l_samples), len(r_samples), len(s_samples))
    return f_samples, l_samples, r_samples

