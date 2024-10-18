import gymnasium as gym
import numpy as np
import torch
import ray
from torch.optim import Adam
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchopt
import wandb
from copy import deepcopy



from data_collection import ReplayBuffer ,compute_loss_in_virtual_MDP , take_M_steps_in_env
from neural_nets import State_encoder_stochastic, State_encoder_deterministic, SquashedGaussianMLPActor , MLPQFunction , Model 
from updates import distributed_sac_update ,update_model_target_networks ,update_agent_with_virtual_data_v1,update_agent_with_virtual_data_v2 
from test_and_logs import policy_evaluation , test_policy_with_adaptations ,Logger
from config import get_config

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def train():

    wandb.init()

    

    config=wandb.config
    print(config)
    config=config.as_dict()
    config=DictToObject(config) 
    print(config)

    if config.seeding==True:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
    device_real=config.device_real_world
    device_virtual=config.device_virtual_world

    #---------------
    env_id='Ant-v4'  # "Hopper-v4"  ,  "Walker2d-v4"  , "HalfCheetah-v4"
    def make_env(env_id):
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if config.seeding==True:
            env.action_space.seed(config.seed)
            env.observation_space.seed(config.seed)

        return env

    #-------------------
    #---------------

    env, test_env = make_env(env_id), make_env(env_id)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Replay buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, max_size=config.replay_buffer_size)


    # build policy and value functions

    if not config.stochastic_encoder:
        state_encoder=State_encoder_deterministic(obs_dim=obs_dim[0] ,obs_encoding_size=config.virtual_state_dim)
    else:
        state_encoder=State_encoder_stochastic(obs_dim=obs_dim[0] ,obs_encoding_size=config.virtual_state_dim)


    policy = SquashedGaussianMLPActor(input_dim=config.virtual_state_dim, act_dim=act_dim[0],
                                        hidden_sizes=(256,256), activation=nn.ReLU, act_limit=act_limit) #(256,256)
    q1 = MLPQFunction(obs_dim=obs_dim[0], act_dim=act_dim[0],
                        hidden_sizes=(256,256), activation=nn.ReLU)
    q2 = MLPQFunction(obs_dim=obs_dim[0], act_dim=act_dim[0],
                        hidden_sizes=(256,256), activation=nn.ReLU)

    #target networks
    target_q1 =  deepcopy(q1)  
    target_q2 =  deepcopy(q2)  
    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in target_q1.parameters(): 
        p.requires_grad = False  
    for p in target_q2.parameters(): 
        p.requires_grad = False  


    # List of parameters for both Q-networks
    q_params = list(q1.parameters()) + list(q2.parameters()) # itertools.chain(q1.parameters(), q2.parameters())

    # Set up optimizers for policy and q-functions
    policy_optimizer = Adam(policy.parameters(), lr=config.policy_lr)
    qs_optimizer = Adam(q_params, lr=config.q_func_lr)
    state_encoder_optimizer=  Adam(state_encoder.parameters(), lr=config.encoder_lr)



    model=Model( virtual_state_dim=config.virtual_state_dim ,
                env=env , len_virtual_trayectories=config.len_virtual_trayectories)

    model_optimizer= Adam(model.parameters(), lr=config.model_lr)

    logger=Logger()
    #----------------------

    #------------------


    logger.prepare_for_wyb_logging()

    if ray.is_initialized:
        ray.shutdown()
    ray.init()


    c=0 
    update_number=0


    new_state=torch.tensor(env.reset()[0],dtype=torch.float32).to(device_virtual) #get a first observation from environment

    total_performed_steps=0
    while total_performed_steps< config.total_timesteps:
        #each loop makes use of [config.num_real_mdp_steps_per_update] new steps of real data
        base_policy_state_dict = torchopt.extract_state_dict(policy)

        ######## ############## ------ REAL MDP DATA COLLECTION ----------############## ##############
        update_current_step_num=0

        while (update_current_step_num + config.num_real_mdp_steps_per_adaptation)<config.num_real_mdp_steps_per_update:
            #perofrm random actions in the first steps
            if total_performed_steps < config.num_initial_random_steps:
                new_state, num_steps_taken = take_M_steps_in_env(random_actions=True, state_encoder=state_encoder,policy=policy,env=env, replay_buffer=replay_buffer , 
                                            m=config.num_real_mdp_steps_per_adaptation ,current_state=new_state ,
                                            finish_if_episode_ends=True, logger=logger,device=device_real)
            
            #adapt the base policy and perform actions with it for data collection
            else:
                adaptation_optimizer =torchopt.MetaSGD(policy, lr=config.adaptation_lr) #aca no tengo porque usar un differetiable optimizer
                #adapt base policy using model data 
                for l in range(config.num_updates_in_adaptation):
                    policy_loss_for_adaptation=compute_loss_in_virtual_MDP(state_encoder=state_encoder,policy=policy, model=model , current_real_state=new_state ,
                                                                            config=config,device=device_virtual)
                    adaptation_optimizer.step(policy_loss_for_adaptation)
                #steps in real world - collect data with adapted policy in real world for m steps and add them to the replay_buffer . 
                new_state, num_steps_taken = take_M_steps_in_env(state_encoder=state_encoder,policy=policy,env=env, replay_buffer=replay_buffer , 
                                                            m=config.num_real_mdp_steps_per_adaptation ,current_state=new_state ,
                                                            finish_if_episode_ends=True,logger=logger, device=device_real)
                torchopt.recover_state_dict(policy, base_policy_state_dict)

            update_current_step_num+=num_steps_taken
            total_performed_steps+=num_steps_taken

        print(f'return at {total_performed_steps} steps taken = {np.mean(logger.episodes_returns[-1:])}' )

        logger.log_training_performance(total_real_steps=total_performed_steps)

        ######## ############## ############## ############## ############## ##############

        ######## ############## ------   UPDATE MODELS ---------- ############## ##############

        if total_performed_steps< config.num_steps_to_start_updating_after:
            continue
        else:
            for j in range(update_current_step_num * config.updates_to_steps_ratio):  #perform as many update iterations as num of steps collected times a ration defined in config #range(config.num_real_mdp_steps_per_update):
                
                #sac_update()
                distributed_sac_update(replay_buffer =replay_buffer,state_encoder=state_encoder, policy=policy , model=model ,
                        q1=q1 ,q2=q2 , target_q1=target_q1, target_q2=target_q2  , state_encoder_optimizer=state_encoder_optimizer,
                        policy_optimizer=policy_optimizer ,qs_optimizer=qs_optimizer, model_optimizer=model_optimizer ,
                            config=config , logger=logger)
                
                #update_model_target_networks(model,config)

                #update_agent_with_virtual_data_v2(replay_buffer=replay_buffer , num_states_to_consider=config.num_states_for_estimating_virtual_loss,
                #                                  state_encoder=state_encoder, policy=policy  , model=model , policy_optimizer=policy_optimizer ,config=config, logger=logger)
                #update_agent_with_virtual_data_v1(replay_buffer=replay_buffer , num_states_to_consider=config.num_states_for_estimating_virtual_loss
                #                                   , state_encoder=state_encoder, policy=policy  , model=model , policy_optimizer=policy_optimizer
                #                                   , config=config, logger=logger)

                
                logger.num_real_steps_at_time_of_update.append(total_performed_steps)
                logger.log_update_metrics(total_real_steps=total_performed_steps)
                update_number+=1

        if total_performed_steps//4000 > c:
            logger.num_real_steps_at_time_of_test.append(total_performed_steps)
            base_policy_returns=policy_evaluation(state_encoder=state_encoder,policy=policy, env=test_env, num_episodes=config.num_test_episodes, deterministic=True,device=device_real) 
            logger.test_base_params_episodes_returns.append(base_policy_returns)
            test_policy_with_adaptations(state_encoder, policy ,model, env, num_episodes=config.num_test_episodes, config=config ,logger=logger )

            logger.log_test_performance(total_real_steps=total_performed_steps)
            c+=1

            wandb.log({"best_achieved_performance":np.max(logger.test_episodes_returns)} ,step=total_performed_steps)
            

    wandb.finish()



if __name__=='__main__':
    sweep_id='octaviopappalardo/state_conditioned_model/h6yzwzwo'
    wandb.agent(sweep_id, function=train, count=1)
