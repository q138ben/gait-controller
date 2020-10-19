from osim.env import L2M2019Env
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl


"""
imported package dir: E:\\miniconda3_64\\envs\\osim_onn\\lib\\site-packages\\osim'

"""


from onn_torch_gd import Neural_Network

print ('onn imported')

from sklearn.datasets import make_classification, make_circles

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score,mean_squared_error

import numpy as np


from torch.utils.tensorboard import SummaryWriter

import argparse

import datetime

import torch.nn as nn

import torch.optim as optim

from statsmodels.tsa.stattools import adfuller, kpss

import pandas as pd

from fireTS.models import NARX, DirectAutoRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


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


now_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
PATH = 'runs/L2M2019_narx/'+now_time
writer = SummaryWriter(PATH )


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


# initiate narx network

mdl_r = NARX(RandomForestRegressor(), auto_order=1, exog_order=[1,1,1], exog_delay=[0,0,0])
mdl_l = NARX(RandomForestRegressor(), auto_order=1, exog_order=[1,1,1], exog_delay=[0,0,0])





load_file = 'state_dict_model.pt'

if args['load'] == True:
    onn_network.load_state_dict(torch.load(PATH +'/'+ load_file))
    print('%s loaded'%load_file)
else:
    print('trained from scratch')


if args['train'] == True:
    print ('Performing traning mode')
else:
    print('Not traing mode')
   
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
y_list_l = []
y_list_r = []
acc_list = []

# timestep = 300

window_size = 250


running_loss = 0.0

#df = pd.DataFrame(index=range(timstep_limit),columns=['grf_r','grf_l'])

for i in range(timstep_limit):

    t += sim_dt
    # locoCtrl.set_control_params(params)
    action = locoCtrl.update(obs_dict)
    print ('action = ',action)
    break

    # done if either the pelvis of the human model falls below 0.6 meters or when it reaches 10 seconds (i=1000)
    obs_dict, reward, done, info = env.step(
        action, project=True, obs_as_dict=True)
    print ('pelvis_vel_0 = ',obs_dict['pelvis']['vel'][0])

		

    hip_abd[0, :] = -obs_dict['r_leg']['joint']['hip_abd']
    hip_angle[0, :] = -obs_dict['r_leg']['joint']['hip']
    knee_angle[0, :] = -obs_dict['r_leg']['joint']['knee']
    ankle_angle[0, :] = -obs_dict['r_leg']['joint']['ankle']

    r_foot_force[0, :] = obs_dict['r_leg']['ground_reaction_forces'][2]
    l_foot_force[0, :] = obs_dict['l_leg']['ground_reaction_forces'][2]

#   df.loc[i,['grf_r']] = r_foot_force[0, :]
#   df.loc[i,['grf_l']] = l_foot_force[0, :]




    X = np.array([hip_angle[0, :] ,knee_angle[0, :] ,ankle_angle[0, :]]).T
    y_r = np.array(r_foot_force[0,:])
    y_l = np.array(l_foot_force[0,:])
#    print('X_shape :', np.shape(X))
#    print('y_r_shape :', np.shape(y_r))

    if i <= window_size:
        X_list.append(X)
        y_list_r.append(y_r)
        y_list_l.append(y_l)





    if  i == window_size:

        X_train = np.concatenate(X_list[-200:],axis= 0)
        y_train_r = np.concatenate(y_list_r[-200:],axis= 0)
        y_train_l = np.concatenate(y_list_l[-200:],axis= 0)		
        print ('X_train_shape :', np.shape(X_train))
        print('y_train_shape :', np.shape(y_train_r))
#       print('X_train:',X_train)
#       print('y_train:',y_train)
        mdl_r.fit(X_train, y_train_r)
        mdl_l.fit(X_train, y_train_l)
        print ('model finishes fitting')


#    if  len(X_list) <= window_size & len(X_list) >= 2:
#
#        X_train = np.concatenate(X_list[-200:],axis= 0)
#        y_train_r = np.concatenate(y_list_r[-200:],axis= 0)
#        y_train_l = np.concatenate(y_list_l[-200:],axis= 0)
##        print('X_train:',X_train)
##        print('y_train:',y_train_r)		
#        mdl_r.fit(X_train, y_train_r)
#        mdl_l.fit(X_train, y_train_l)
##        print ('model finishes fitting')

    if i > window_size:
        X_test_w = np.append(X, [[7, 8, 9]], axis=0)

        y_test_w_r = np.append(y_r,100)
        y_test_w_l = np.append(y_l,100)

        y_pred_w_r = mdl_r.predict(X_test_w,y_test_w_r,step=1)
        y_pred_w_l = mdl_l.predict(X_test_w,y_test_w_l,step=1)		
#        y_pred_list.append(y_pred_w[-1])
        loss = np.mean((y_pred_w_r[-1] - r_foot_force[0,:])**2)**.5 
#        print ('loss = ', loss)
        running_loss += loss
	    
        writer.add_scalars(f'right ground reaction force', {'true':r_foot_force[0,:],'predicted': y_pred_w_r[-1],},  i)
        writer.add_scalars(f'left ground reaction force', {'true':l_foot_force[0,:],'predicted': y_pred_w_l[-1],},  i)

        if i % 200 == 0:
            writer.add_scalar('rmse_loss',
                               running_loss / 200,
                                i)
            print("Online rmse error on %d steps: %.4f"%(i,running_loss / 200))
            running_loss = 0.0

	    
#       if loss_value < 0.01:
#           print ('loss = ', loss_value.numpy())
#           force_ind.append(i)
#           obs_dict['r_leg']['ground_reaction_forces'][2] = y_pred[0,0]
#           obs_dict['l_leg']['ground_reaction_forces'][2] = y_pred[0,1]

        obs_dict['r_leg']['ground_reaction_forces'][2] = y_pred_w_r[-1]
        obs_dict['l_leg']['ground_reaction_forces'][2] = y_pred_w_l[-1]
    total_reward += reward
    if done:
        break
print('    score={} time={}sec'.format(total_reward, t))

#df.to_csv('ground_reaction_force.csv',float_format='%.4f')

# ADF Test
#result = adfuller(df['grf_r'].values, autolag='AIC')
#print(f'ADF Statistic: {result[0]}')
#print(f'p-value: {result[1]}')
#for key, value in result[4].items():
#    print('Critial Values:')
#    print(f'   {key}, {value}')
#
## KPSS Test
#result = kpss(df['grf_l'].values, regression='c')
#print('\nKPSS Statistic: %f' % result[0])
#print('p-value: %f' % result[1])
#for key, value in result[3].items():
#    print('Critial Values:')
#    print(f'   {key}, {value}')

#print('toatal_acc : ', np.mean(np.array(acc_list)))

# concat_arr = np.concatenate((hip_angle,knee_angle,ankle_angle,r_foot_force,l_foot_force),axis= 1)



# fig,ax=plt.subplots(4,1)
# ax[0].plot(np.arange(timstep_limit),hip_angle)
# ax[1].plot(np.arange(timstep_limit),knee_angle)
# ax[2].plot(np.arange(timstep_limit),ankle_angle)
# ax[3].plot(np.arange(timstep_limit),r_foot_force)

# plt.show()