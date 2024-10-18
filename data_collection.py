import numpy as np
import torch
import numpy as np



#######  --------  REPLAY BUFFER  ----------  ##########

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size):
        self.obs_buf = np.zeros( ((max_size,)+ obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros( ((max_size,)+ obs_dim), dtype=np.float32)
        self.act_buf = np.zeros( ((max_size,)+ act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, max_size

    def store_step_data(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_standard_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def sample_list_of_batches(self , num_elements , max_steps_per_element):
        '''Returns a list of num_elements batches where each batch contains consecutive steps taken in the environment.
        For this:
        Selects num_elements random indices of the buffer. For each index it creates a batch
        of data that starts at that timestep and contains the following max_steps_per_element-1 elements ; 
        or less if a termination episode (or the buffer's pointer) is encountered . All the batches are combined into a list'''
        total_states=num_elements #total states will be in between num_elements and num_elements x max_steps_per_element
        list_of_batches=[]
        idxs = np.random.randint(0, self.size, size=num_elements)

        for idx in idxs:
            batch_indices=[idx]
            for i in range(idx+1 ,idx+max_steps_per_element):
                if (i>self.max_size-1) or (self.done_buf[i-1]==True) or  (i==self.ptr):
                    break
                else:
                    batch_indices.append(i)
                    total_states+=1

            list_of_batches.append( self._batch_by_indices(batch_indices) )  

        return list_of_batches , total_states
            
    def _batch_by_indices(self ,indices):
        batch = dict(obs=self.obs_buf[indices],
                     obs2=self.obs2_buf[indices],
                     act=self.act_buf[indices],
                     rew=self.rew_buf[indices],
                     done=self.done_buf[indices])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}



##########################################################


#######  --------  Collect data and compute loss in virtual data  ----------  ##########

def compute_loss_in_virtual_MDP( state_encoder, policy , model , current_real_state,config, for_policy_update=False,device='cpu'):

    num_trayectories= config.num_virtual_trayectories_per_adaptation_update
    model.num_parallel_trayectories=num_trayectories

    virtual_observations=torch.zeros(model.max_episode_steps , num_trayectories  ,model.virtual_state_dim)
    virtual_actions= torch.zeros(model.max_episode_steps , num_trayectories  ,model.action_dim)
    virtual_actions_logprobs= torch.zeros(model.max_episode_steps , num_trayectories  ,1)

    #let the model know what the current state in the real world is 
    model.current_real_state=current_real_state.to(device)

    target=0

    #get an initial virtual state from the model (collects all the trayectories simultaneously)
    next_obs= model.reset(state_encoder)
    done=False

    for step in range(0, config.len_virtual_trayectories):

        assert done==False , 'code prepared to run the different tray parallely-not sequentially'
        if done:
            pass

        obs = next_obs
     
        action, logprob_a  = policy.get_action(obs,virtual_state=True ,with_logprob=True,std_factor=config.std_factor)

        virtual_observations[step]=obs
        virtual_actions[step] =action
        virtual_actions_logprobs[step] =logprob_a.unsqueeze(-1)
        next_obs,  done = model.step(action.squeeze(0))  


    trayectories=torch.cat((virtual_observations, virtual_actions, virtual_actions_logprobs) ,dim=2)

    if not for_policy_update:
        target1=model.trayectory_evaluator1(trayectories)
        target2=model.trayectory_evaluator2(trayectories)
    else: #if calculating loss to train policy to minimize ,then use target networks
        target1=model.target_trayectory_evaluator1(trayectories)
        target2=model.target_trayectory_evaluator2(trayectories)
    
    target=torch.min(target1, target2)


    target=torch.sum(target)/ num_trayectories
    #entropy_regularization= model.entropy_loss_scale* ((logprob_a * config.alpha).sum())/ num_trayectories
    #entropy_regularization= ((logprob_a * config.alpha).sum())/ (num_trayectories*model.max_episode_steps) #normalize by total num of acions taken - the same as in policy loss


    loss= -target #+ entropy_regularization
    
    return loss




##########################################################


#######  --------  Collect data in real MDP  ----------  ##########

def take_M_steps_in_env(state_encoder ,policy ,env, replay_buffer , 
                        m,current_state, finish_if_episode_ends=True,
                         random_actions=False,logger=None, device='cpu'):

    num_steps_taken=0
    #set starting state
    next_obs=current_state 

    for step in range(m):

        obs= next_obs

        if random_actions==True:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action, _ = policy.get_action(obs.unsqueeze(0), with_logprob=False, state_encoder=state_encoder) 
                action=action.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)  

        reward= torch.as_tensor(reward,dtype=torch.float32).to(device)
        next_obs=torch.as_tensor(next_obs, dtype=torch.float32)
        replay_buffer.store_step_data( obs=obs, act=action, rew=reward, next_obs=next_obs
                                        ,done= terminated)


        num_steps_taken+=1
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

    return  next_obs , num_steps_taken



###################################################




def compute_loss_in_virtual_MDP_v2( state_encoder, policy , model , observations,config, for_policy_update=False,device='cpu'):
    ''' hace lo mismo que la v1 pero lo computa para muchos real states en paralelo. Osea toma obs inputs the (num_obs , obs_dim) en ves de (obs_dim)'''

    num_trayectories= config.num_virtual_trayectories_per_adaptation_update
    model.num_parallel_trayectories=num_trayectories
    num_real_states=observations.shape[0]

    virtual_observations=torch.zeros( model.max_episode_steps ,num_trayectories  ,num_real_states, model.virtual_state_dim)
    virtual_actions= torch.zeros(model.max_episode_steps , num_trayectories  ,num_real_states,model.action_dim)
    virtual_actions_logprobs= torch.zeros(model.max_episode_steps , num_trayectories  ,num_real_states,1)


    #let the model know what the current state in the real world is 
    model.current_real_state=observations.to(device)
    target=0

    #get an initial virtual state from the model (collects all the trayectories simultaneously)
    next_obs= model.reset(state_encoder) # (num_trayectories ,num_real_states ,  virtual_obs_dim)

    done=False

    for step in range(0, config.len_virtual_trayectories):

        assert done==False , 'code prepared to run the different tray parallely-not sequentially'
        if done:
            pass

        obs = next_obs

        action, logprob_a  = policy.get_action(obs,virtual_state=True ,with_logprob=True,std_factor=config.std_factor) # (num_trayectories ,num_real_states , action_dim) 

        virtual_observations[step,...]=obs
        virtual_actions[step,...] =action
        virtual_actions_logprobs[step,...] =logprob_a.unsqueeze(-1)
     
        next_obs,  done = model.step(action) 


    #Compute the model's evaluation of each trayectory (the target to be optimized) and normalize by the amount of trayectories

    trayectories=torch.cat((virtual_observations,virtual_actions,virtual_actions_logprobs) ,dim=-1)

    input_dim=trayectories.shape[-1]
    trayectories=trayectories.view(model.max_episode_steps,-1, input_dim)

    if not for_policy_update:
        target1=model.trayectory_evaluator1(trayectories)
        target2=model.trayectory_evaluator2(trayectories)
    else: #if calculating loss to train policy to minimize ,then use target networks
        target1=model.target_trayectory_evaluator1(trayectories)
        target2=model.target_trayectory_evaluator2(trayectories)
    
    target=torch.min(target1, target2)

    target=torch.sum(target)/ (num_trayectories*num_real_states)
    #entropy_regularization= model.entropy_loss_scale* ((logprob_a * config.alpha).sum())/ (num_trayectories*num_real_states)
    
    loss= -target #+ entropy_regularization
    
    return loss

