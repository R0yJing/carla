from neural_net import *
from matplotlib import pyplot as plt
agt = agent()
#agt.load_n_images(100)
agt.evaluate()
#agt.train()
# test = test_module()
# test.train()
#im_module = test_image_module()
#im_module.train()
# im_module.eval()
# one = np.array([np.ones((88,200,3))])
# # for i in range(100):
# #     one[i][i] = 1
# #one.resize((1,100,100,1))
# two = np.array([np.zeros((88,200,3)) + 0.5])

# #two.resize((1,100,100,1))

# three = np.array([np.zeros((88,200,3))])
# #three.resize((1,100,100,1))
# x=im_module.model.predict(one, batch_size=1)
# y=im_module.model.predict(two, batch_size=1)#[np.array([[0,1,0], [0,1,0], [0,1,0]]).reshape((1,3,3,1))], batch_size=1)
# z=im_module.model.predict(three, batch_size=1)
#print(x, y,z)
agt.load_n_images(100)
agt.evaluate()
# #agt.train()
# # expert_actions = [action for action in agt.train_actions[:100]]
# # steers = [act[0] for act in expert_actions]
# # throttles = [act[1] for act in expert_actions]
# # brakes = [act[2] for act in expert_actions]
# # p_actions = []
# # for i in range(100):
# #     action = agt.get_single_action(agt.train_images[i], agt.train_speeds[i], agt.train_cmds[i])
# #     p_actions.append(action)

# # p_steer = [act[0] for act in p_actions]
# # p_throttle = [act[1] for act in p_actions]
# # p_brakes = [act[2] for act in p_actions]
# print("trainable weights")
# print(len(agt.model.trainable_weights))
# print("untrainable weights")
# print(len(agt.model.non_trainable_weights))
# print("weights")
# print(len(agt.model.weights))



# # plt.plot(steers)
# # plt.plot(throttles)
# # plt.plot(brakes)
# # plt.plot(p_steer)
# # plt.plot(p_throttle)
# # plt.plot(p_brakes)


# # model.compile(loss='mean_squared_error',
# #             optimizer='adam',
# #             metrics='mean_squared_error')

# # speed = agt.spd_module
# # speed_model = Model(speed.module_in, speed.module_out)
# # # preds = speed_model.predict([speed for speed in agt.train_speeds])
def show_plot(preds, name):

    print("name")
    preds_dict = {}
    for i in range(len(preds[0])):
        preds_dict[i] = []
    for pred in preds:
        for i in range(len(pred)):
            preds_dict[i].append(pred[i])
    plt.figure()
    for p in preds_dict.values():
        plt.plot(p)
    plt.title(name )
    

img_mod = agt.img_module
img_model = Model(img_mod.image_model_in, img_mod.image_model_out)
images = [img for img in agt.train_images]
preds_img = img_model.predict(np.array(agt.train_images), 100 )

show_plot(preds_img, "image")


cmd_mod = agt.cmd_module
cmd_model = Model(cmd_mod.module_in, cmd_mod.module_out)
preds_cmd = cmd_model.predict(np.array(agt.train_cmds), 100 )

show_plot(preds_cmd, "cmd")

spd_mod = agt.spd_module
spd_model = Model(spd_mod.module_in, spd_mod.module_out)
preds_spd = spd_model.predict(np.array(agt.train_speeds), 100 )

show_plot(preds_spd, "speed")

preds = agt.get_actions(agt.train_images, agt.train_speeds, agt.train_cmds)
show_plot(preds, "actions")
plt.show()


#     # cmd_mod = agt.cmd_module
#     # cmd_model= Model(cmd_mod.module_in, cmd_mod.module_out)
#     # preds = cmd_model.predict(agt.train_cmds.tolist())
#     # plt.plot(preds)

#     # img_model = Model(agt.img_module.image_model_in, agt.img_module.image_model_out)
#     # preds = img_model.predict(agt.train_images.tolist())

#     # plt.plot(preds)


# # for i in range(4):
# #     print(model.predict([np.random.uniform(0, 1, (1,3))], batch_size=1)[0])
# img_mod = image_module()
# img_model = Model(img_mod.image_model_in, img_mod.image_model_out)
# preds = img_model.predict([np.random.uniform(0, 1, (10, 88, 200, 3))], batch_size=10)

# # # #agt = agent(fake_training=True)
# # # # for i in range(10):
# # # #     print(agt.get_single_action(np.random.uniform(0, 1, (88, 200, 3)), np.random.randint(1, 30), "straight"))


