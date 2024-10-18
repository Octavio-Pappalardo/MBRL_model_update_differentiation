import numpy as np
import torch
import numpy as np
import torch
import torchopt
import wandb

from data_collection import compute_loss_in_virtual_MDP


##########------------   LOGGER -----################
class Logger:
    def __init__(self):
        self.episodes_returns=[0]
        self.episodes_lengths=[0]
        self.num_episodes=0

        #metrics during updates
        self.num_updates=0
        self.num_real_steps_at_time_of_update=[]
        self.policy_loss = []
        self.q_loss= []
        self.actions_logprobs=[]
        self.entropies=[]
        self.virtual_loss=[0]
        self.kl_regu_loss=[0]

        self.q1_means= []
        self.q1_stds = []
        self.q2_means= []
        self.q2_stds = []
        self.model_consistency_loss=[]

        #tests metrics
        self.num_real_steps_at_time_of_test=[]
        self.test_episodes_returns = []
        self.test_episodes_lengths = []
        self.test_base_params_episodes_returns=[]

    def prepare_for_wyb_logging(self):
        # define our custom x axis metric
        wandb.define_metric("num updates")
        # define which metrics will be plotted against it
        wandb.define_metric("q1 means", step_metric="num updates")
        wandb.define_metric("q2 means", step_metric="num updates")
        wandb.define_metric("q1 stds", step_metric="num updates")
        wandb.define_metric("q2 stds", step_metric="num updates")
        wandb.define_metric("policy loss", step_metric="num updates")
        wandb.define_metric("model_consistency_loss", step_metric="num updates")
        wandb.define_metric("q loss", step_metric="num updates")
        wandb.define_metric("actions logprobs", step_metric="num updates")
        wandb.define_metric("entropy", step_metric="num updates")
        wandb.define_metric("virtual loss", step_metric="num updates")
        wandb.define_metric("kl regularization loss",step_metric="num updates")
        wandb.define_metric("real steps at time of update", step_metric="num updates")
        

    def log_update_metrics(self,total_real_steps):
        self.num_updates+=1
        wandb.log({'q1 means': self.q1_means[-1] ,'q2 means':self.q1_means[-1] ,
                    'q1 stds': self.q1_stds[-1] ,'q2 stds':self.q2_stds[-1] , 
                    'policy loss': self.policy_loss[-1] , 'model_consistency_loss':self.model_consistency_loss[-1],
                    'q loss':self.q_loss[-1] ,
                    'actions logprobs': self.actions_logprobs[-1] , 'entropy': self.entropies[-1] ,
                    'virtual loss':self.virtual_loss[-1] ,'kl regularization loss':self.kl_regu_loss[-1],
                    'real steps at time of update': self.num_real_steps_at_time_of_update[-1],
                     'num updates' : self.num_updates},step=total_real_steps)


    def log_training_performance( self,total_real_steps):
        if len(self.episodes_returns) > self.num_episodes:
            wandb.log({'episodes returns': np.mean(self.episodes_returns[-1]),
                    'episodes lengths':np.mean(self.episodes_lengths[-1]) } ,step=total_real_steps)
        self.num_episodes= len(self.episodes_returns) 
            

    def log_test_performance(self ,total_real_steps):
        wandb.log({'test_base_params_episodes_returns': np.mean(self.test_base_params_episodes_returns[-1]),
                'test_episodes_returns':np.mean(self.test_episodes_returns[-1]) ,
                'test_episodes_lengths':np.mean(self.test_episodes_lengths[-1]),
                 'real steps at time of test': self.num_real_steps_at_time_of_test[-1] } ,step=total_real_steps  )

#######  --------  TEST POLICY  ----------  ##########

# a utility
def _test_take_M_steps_in_env(state_encoder ,policy ,env, 
                        m,current_state, finish_if_episode_ends=True,
                        logger=None, deterministic=False, device='cpu'):

    num_steps_taken=0
    #set starting state
    next_obs=current_state 

    for step in range(m):

        #prepare for new step: next_obs becomes the new step's observation , the done flag which indicates wether the episode finished in the
        #last step becomes this episodes prev_done.
        obs= next_obs

        # get action
      
        with torch.no_grad():
            action, _ = policy.get_action(obs.unsqueeze(0), with_logprob=False, state_encoder=state_encoder,deterministic=deterministic) 
            action=action.squeeze(0).cpu().numpy()

        #execute the action and get environment response.
        next_obs, reward, terminated, truncated, info = env.step(action)  

        #preprocess and store data 
        reward= torch.as_tensor(reward,dtype=torch.float32).to(device)
        next_obs=torch.as_tensor(next_obs, dtype=torch.float32)

        #deal with the case where the episode ends 
        if terminated or truncated:
            #reset environment
            next_obs = torch.Tensor(env.reset()[0]).to(device)
            #save metrics
            assert 'episode' in info , 'problem with recordeEpisodeStatistics wrapper'
            if logger:
                logger.episodes_returns.append(info['episode']['r'][0])
                logger.episodes_lengths.append(info['episode']['l'][0])

            #if episode reaches ends reduce the buffer size to the number of steps taken and break from loop
            if finish_if_episode_ends==True:
                break

    return  next_obs 



            

def test_policy_with_adaptations(state_encoder, policy ,model, env, num_episodes, config ,logger ):
    test_logger=Logger()
    base_policy_state_dict = torchopt.extract_state_dict(policy)
    adaptation_optimizer =torchopt.MetaSGD(policy, lr=config.adaptation_lr)

    new_state=torch.tensor(env.reset()[0],dtype=torch.float32) #get a first observation from environment
    # Test the performance of the deterministic version of the agent.
    while len(test_logger.episodes_returns)<num_episodes:

        for l in range(config.num_updates_in_adaptation):
            policy_loss_for_adaptation=compute_loss_in_virtual_MDP(state_encoder=state_encoder,policy=policy, model=model , current_real_state=new_state ,
                                                                    config=config)
            adaptation_optimizer.step(policy_loss_for_adaptation)
        #steps in real world - collect data with adapted policy in real world for m steps and add them to the replay_buffer . 
        new_state = _test_take_M_steps_in_env(state_encoder=state_encoder,policy=policy,env=env,
                                                    m=config.num_real_mdp_steps_per_adaptation ,current_state=new_state ,
                                                    finish_if_episode_ends=True,logger=test_logger, deterministic=True)
        torchopt.recover_state_dict(policy, base_policy_state_dict)


    logger.test_episodes_returns.append(np.mean(test_logger.episodes_returns))
    logger.test_episodes_lengths.append(np.mean(test_logger.episodes_lengths))



#for evaluating the base policy
def policy_evaluation(state_encoder ,policy, env, num_episodes, deterministic=True, device='cpu'):
    episodes_lengths=[]
    episodes_returns=[]
    episode_num=0

    #get an initial state from the environment 
    next_obs = torch.tensor(env.reset()[0],dtype=torch.float32).to(device)
    done = torch.zeros(1).to(device)

    while episode_num < num_episodes:

        #prepare for new step: next_obs becomes the new step's observation , the done flag which indicates wether the episode finished in the
        #last step becomes this episodes prev_done.
        obs, prev_done = next_obs, done

        # get actionA action predictions and state value estimates
        with torch.no_grad():
            action, _=  policy.get_action(obs.unsqueeze(0) ,deterministic=deterministic ,state_encoder=state_encoder)  

        #execute the action and get environment response.
        next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy()) 
        done= terminated or truncated

        #prepare for next step
        next_obs=torch.as_tensor(next_obs,dtype=torch.float32).to(device)

        #deal with the case where the episode ends
        if done:
            #reset environment
            next_obs = torch.tensor(env.reset()[0],dtype=torch.float32).to(device) 
            episode_num+=1
            #save metrics
            assert 'episode' in info , 'problem with recordeEpisodeStatistics wrapper'
            episodes_returns.append(info['episode']['r'][0])
            episodes_lengths.append(info['episode']['l'][0])

    return np.array(episodes_returns).mean() 