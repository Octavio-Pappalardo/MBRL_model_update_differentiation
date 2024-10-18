import wandb


# 2: Define the search space
parameters_dict = {
'seed' : {'values': [1,2,3]},
'seeding' : {'value': False},
'device_real_world' : {'value': 'cpu'},
'device_virtual_world' : {'value': 'cpu'},

 'std_factor': {'values': [1,1.5,2.5] },
 'constistency_loss_coef': {'values': [ 0,0.1,1]},
 'stochastic_encoder': {'values': [ True, False] , "probabilities":[0.2,0.8]},
                        

'total_timesteps' : {'value': 200000},
'num_real_mdp_steps_per_update' : {'value': 50},
'num_real_mdp_steps_per_adaptation' : {'values': [5,10]},

'num_updates_in_adaptation' : {'value': 1},
'num_virtual_trayectories_per_adaptation_update' : {'values': [35,20],"probabilities":[0.2,0.8]},  
'len_virtual_trayectories' : {'values': [5,10,15]},

'num_steps_to_start_updating_after' : {'value': 1000},
'num_initial_random_steps' : {'value': 10000},

'policy_lr' : {'value': 3e-4},
'q_func_lr' : {'value': 1e-3},
'model_lr' : {'value': 3e-4},
'encoder_lr' : {'value': 3e-4},

'adaptation_lr' : {'value': 0.05},


'replay_buffer_size' : {'value': int(1e5)},
'num_batches_of_consecutive_elements_per_sac_update' : {'value': 30},
'updates_to_steps_ratio' : {'value': 1},

'virtual_state_dim' : {'value': 30},
'action_encoding_dim' : {'value': None},

 'gamma': {'value': 0.99 },
 'alpha': {'value': 0.2 },
 'polyak': {'value': 0.995 },

 'num_test_episodes': {'value': 15 },
 'test_every_steps_num': {'value': 5000 },
 
 }


# Define sweep config
sweep_configuration = {
    "method": "random", # method fro searching . grid ,random or bayes
    "name": "sac_plus_implicit_model_sweep",
    "metric": {"goal": "maximize", "name": "best_achieved_performance"},
}
sweep_configuration['parameters']=parameters_dict


#Ensure that you log (wandb.log) the exact metric name that you defined the sweep to optimize within your Python script or Jupyter Notebook.
#Defining the metric in the sweep configuration is only required when using the bayes method for the sweep.

# 3: Initialize a sweep job with the specified configurations
sweep_id = wandb.sweep(sweep=sweep_configuration, project='state_conditioned_model')

print(sweep_id)