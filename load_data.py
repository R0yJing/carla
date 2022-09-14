datadir = r"coiltrain_dataset"
from constants import BATCH_SIZE, TARGET_SPEED
import glob
import json
#get left
import numpy as np
import os
from lateral_augmentations import *
import random
import imageio
from send2trash import send2trash
folders = os.listdir(datadir)
DEBUG_BATCH_SIZE = 24
vectorised_commands = np.eye(4).astype('uint8')
import numpy as np
errors =0
#TOTAL_NUM_SAMPLES =  * 4 # 11670 * 4
def _get_min_type_sample(data):
    return min([len(d) for d in data])
def glob_sorter(item:str, len=5):
    idx = item.index('.')
    return int(item[idx-len:idx])
num_train_files = 0
num_val_files = 0
def load_data(load_train=True, debug=False, max_val_lim=None):
    global errors, num_train_files, num_val_files, DEBUG_BATCH_SIZE

    num_train_files = 22116 * 3 if not debug else 9
    if max_val_lim is not None:
        num_val_files = max_val_lim
    else:
        num_val_files = 24000 if not debug else 9

    commands_ct = [0,0,0,0]

    images = []
    measurements = []
    commands = []
    actions = []
    fin = False

    ct = 0
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
                if not load_train:
                    im_centre = im_centre.astype('float32') /255
                    im_left = im_left.astype('float32')/255
                    im_right = im_right.astype('float32')/255
                
                  
                try:
                    with open(measurement, 'rb') as fp:
                        
                        data = json.load(fp)
                        
                        command = vectorised_commands[int(data['directions']) - 2]
                        steer = data['steer']
                        speed = data['playerMeasurements']['forwardSpeed'] 
                        throttle = data['throttle']
                        brake = data['brake']
                        # if int(data['directions']) == 3 and steer <= -0.1 or (int(data['directions']) == 4 and steer >= 0.1):
                        #     print(int(data['directions']))
                        # else:
                        #     continue
                            #print(agt.get_action(im_centre, speed, int(data['directions'])))
                        
                        
                        if not load_train and sum(commands_ct) >= num_val_files:
                             fin = True
                             break
                        elif load_train and debug and sum(commands_ct) == DEBUG_BATCH_SIZE * 4:
                            fin = True
                            break
                        #only 3 types of commands needs to validated
                        if not load_train and commands_ct[int(data['directions']) - 2] >= num_val_files // 3:#TOTAL_NUM_SAMPLES //4:
                            continue

                        elif load_train and debug and commands_ct[int(data['directions']) - 2] >= DEBUG_BATCH_SIZE:
                            continue
                        commands_ct[int(data['directions']) - 2] += 3
                    
                        images.append(im_centre)
                        images.append(im_left)
                        images.append(im_right) 
                        for i in range(3):
                            measurements.append(speed/TARGET_SPEED)
                            commands.append(command)
                        actions.append(np.array([steer, throttle, brake]))
                        actions.append(np.array([augment_steering(-45, steer, speed), throttle, brake]))
                        #with right, steer to the left -ve
                        #steer to the right +ve

                        actions.append(np.array([augment_steering(45, steer, speed), throttle, brake]))
                    
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
def load_data_2(load_train=True, debug=False):
    shuffled_images, shuffled_measurements, shuffled_cmds, shuffled_actions = load_data(load_train, debug)
    cmds = np.eye(4).astype('uint8')
    f_samples = [(img, spd, cmd, act) for img, spd, cmd, act in zip(shuffled_images, shuffled_measurements, shuffled_cmds, shuffled_actions) if (cmd == cmds[0]).all()]
    l_samples = [(img, spd, cmd, act) for img, spd, cmd, act in zip(shuffled_images, shuffled_measurements, shuffled_cmds, shuffled_actions) if (cmd == cmds[1]).all()]
    r_samples = [(img, spd, cmd, act) for img, spd, cmd, act in zip(shuffled_images, shuffled_measurements, shuffled_cmds, shuffled_actions) if (cmd == cmds[2]).all()]
    s_samples = [(img, spd, cmd, act) for img, spd, cmd, act in zip(shuffled_images, shuffled_measurements, shuffled_cmds, shuffled_actions) if (cmd == cmds[3]).all()]
    return f_samples, l_samples, r_samples, s_samples