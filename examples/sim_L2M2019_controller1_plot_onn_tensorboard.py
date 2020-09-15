from osim.env import L2M2019Env
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl


"""
imported package dir: E:\\miniconda3_64\\envs\\opensim-rl-clone\\lib\\site-packages\\osim'

"""


from onn.OnlineNeuralNetwork import ONN

print ('onn imported')

from sklearn.datasets import make_classification, make_circles

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score,mean_squared_error

import numpy as np


from torch.utils.tensorboard import SummaryWriter

import argparse

# Construct the argument parser
ap = argparse.ArgumentParser()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Add the arguments to the parser
ap.add_argument("-train",type=str2bool,required=True,
   help="specify mode in training or test, expecting True or False")
ap.add_argument("-load", type=str2bool,required=True,
   help="specify load mode from saved model or not, expecting True or False")

args = vars(ap.parse_args())



writer = SummaryWriter('runs/L2M2019_onn')


mode = '2D'
difficulty = 3
visualize = False
seed = None
sim_dt = 0.01
sim_t = 100
timstep_limit = int(round(sim_t/sim_dt))


INIT_POSE = np.array([
    1.699999999999999956e+00,  # forward speed
    .5,  # rightward speed
    9.023245653983965608e-01,  # pelvis height
    2.012303881285582852e-01,  # trunk lean
    0*np.pi/180,  # [right] hip adduct
    -6.952390849304798115e-01,  # hip flex
    -3.231075259785813891e-01,  # knee extend
    1.709011708233401095e-01,  # ankle flex
    0*np.pi/180,  # [left] hip adduct
    -5.282323914341899296e-02,  # hip flex
    -8.041966456860847323e-01,  # knee extend
    -1.745329251994329478e-01])  # ankle flex

if mode is '2D':
    params = np.loadtxt('params_2D.txt')
elif mode is '3D':
    params = np.loadtxt('params_3D_init.txt')

locoCtrl = OsimReflexCtrl(mode=mode, dt=sim_dt)
locoCtrl.set_control_params(params)

env = L2M2019Env(visualize=visualize, seed=seed, difficulty=difficulty)
env.change_model(model=mode, difficulty=difficulty, seed=seed)
obs_dict = env.reset(project=True, seed=seed,
                     obs_as_dict=True, init_pose=INIT_POSE)
env.spec.timestep_limit = timstep_limit

total_reward = 0
t = 0
i = 0


# initiate onn network




onn_network = ONN(features_size=4, max_num_hidden_layers=5,
                  qtd_neuron_per_hidden_layer=40, n_classes=2,loss_fun = 'mse')

PATH = "state_dict_model.pt"

if args['load'] == True:
    load_file = "state_dict_model_0.13.pt"
    onn_network.load_state_dict(torch.load(load_file))
    print('%s loaded'%load_file)

#hip_angle = np.zeros((timstep_limit, 1))
#knee_angle = np.zeros((timstep_limit, 1))
#ankle_angle = np.zeros((timstep_limit, 1))
#r_foot_force = np.zeros((timstep_limit, 1))
#l_foot_force = np.zeros((timstep_limit, 1))
hip_abd = np.zeros((1, 1))
hip_angle = np.zeros((1, 1))
knee_angle = np.zeros((1, 1))
ankle_angle = np.zeros((1, 1))

r_foot_force = np.zeros((1, 1))
l_foot_force = np.zeros((1, 1))

y_pred_list = []
force_ind = []
X_list = []
y_list = []
acc_list = []

# timestep = 300


running_loss = 0.0



for i in range(timstep_limit):

    t += sim_dt
    # locoCtrl.set_control_params(params)
    action = locoCtrl.update(obs_dict)
    # done if either the pelvis of the human model falls below 0.6 meters or when it reaches 10 seconds (i=1000)
    obs_dict, reward, done, info = env.step(
        action, project=True, obs_as_dict=True)
		

    # hip_angle.append(-obs_dict['r_leg']['joint']['hip'])
    # knee_angle.append(-obs_dict['r_leg']['joint']['knee'])
    # ankle_angle.append(-obs_dict['r_leg']['joint']['ankle'])

#    hip_angle[i, :] = -obs_dict['r_leg']['joint']['hip']
#    knee_angle[i, :] = -obs_dict['r_leg']['joint']['knee']
#    ankle_angle[i, :] = -obs_dict['r_leg']['joint']['ankle']
    hip_abd[0, :] = -obs_dict['r_leg']['joint']['hip_abd']
    hip_angle[0, :] = -obs_dict['r_leg']['joint']['hip']
    knee_angle[0, :] = -obs_dict['r_leg']['joint']['knee']
    ankle_angle[0, :] = -obs_dict['r_leg']['joint']['ankle']

    r_foot_force[0, :] = obs_dict['r_leg']['ground_reaction_forces'][2]
    l_foot_force[0, :] = obs_dict['l_leg']['ground_reaction_forces'][2]

#    if obs_dict['r_leg']['ground_reaction_forces'][2] > 0:
#        y = np.array([1])
#    else:
#        y = np.array([0])

    X = np.array([hip_abd[0, :],hip_angle[0, :] ,knee_angle[0, :] ,ankle_angle[0, :]]).T
    y = np.array([r_foot_force[0,:],l_foot_force[0,:]]).T

#    X_list.append(X)
#    y_list.append(y)


    predictions = onn_network.predict(X)


#    acc = balanced_accuracy_score(y, predictions)
    loss = mean_squared_error(y, predictions)
    running_loss += loss
#    writer.add_scalar('training loss lr=0.001',loss,i)
    if i % 200 == 0:
        writer.add_scalar('training loss',
                           running_loss / 200,
                            i)
        print("Online error on %d steps: %.4f"%(i,running_loss / 200))
        running_loss = 0.0
    if args['train'] == True:
        print (args['train'])
        break
        onn_network.partial_fit(X, y)
        torch.save(onn_network.state_dict(), PATH)

#    if  len(X_list) % 10 == 0:
#        # X,y = split_sequences(X_list,y_list,n_steps = 10)
#
#        X = np.array(X_list).reshape((1,n_steps_in,3))
#
#        #X = np.array(X_list).reshape((1,n_steps_in*3))
#
#        X = tf.convert_to_tensor(X, dtype=tf.float32)
#
#
#        with tf.GradientTape() as tape:
#
#            # Run the forward pass of the layer.
#            # The operations that the layer applies
#            # to its inputs are going to be recorded
#            # on the GradientTape.
#            logits = model(X, training=True)  # Logits for this minibatch
#
#            # Compute the loss value for this minibatch.
#            loss_value = loss_fn(y, logits)
#
#
#
#        # Use the gradient tape to automatically retrieve
#        # the gradients of the trainable variables with respect to the loss.
#        grads = tape.gradient(loss_value, model.trainable_weights)
#
#        # Run one step of gradient descent by updating
#        # the value of the variables to minimize the loss.
#        optimizer.apply_gradients(zip(grads, model.trainable_weights))
#
#        y_pred= model.predict(X)
#
#        predictions = onn_network.predict(X_test)
#
#        if i % 20 == 0:
#            print(
#                "Training loss at step %d: %.4f"
#                % (i, float(loss_value))
#            )
#
#
#
#
#        if loss_value < 0.01:
#            print ('loss = ', loss_value.numpy())
#            force_ind.append(i)
#            obs_dict['r_leg']['ground_reaction_forces'][2] = y_pred[0,0]
#            obs_dict['l_leg']['ground_reaction_forces'][2] = y_pred[0,1]
#
#        y_pred_list.append(y_pred[0])
#
#        X_list.pop(0)
#        y_list.pop(0)

    total_reward += reward
    if done:
        break
print('    score={} time={}sec'.format(total_reward, t))

#print('toatal_acc : ', np.mean(np.array(acc_list)))

# concat_arr = np.concatenate((hip_angle,knee_angle,ankle_angle,r_foot_force,l_foot_force),axis= 1)



# fig,ax=plt.subplots(4,1)
# ax[0].plot(np.arange(timstep_limit),hip_angle)
# ax[1].plot(np.arange(timstep_limit),knee_angle)
# ax[2].plot(np.arange(timstep_limit),ankle_angle)
# ax[3].plot(np.arange(timstep_limit),r_foot_force)

# plt.show()