import numpy as np
import torch
import numpy as np
from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal



def layer_init(layer, std=np.sqrt(2), bias_const=0.0 ,bias=True):
    torch.nn.init.orthogonal_(layer.weight, std)
    if bias==True:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


#######  --------  ACTOR and Q function NETWORKS  ----------  ##########


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class State_encoder_stochastic(nn.Module):
    def __init__(self, obs_dim,obs_encoding_size):
        super(State_encoder_stochastic,self).__init__()

        self.network=nn.Sequential( 
            nn.Linear(obs_dim, 256) ,
            nn.ReLU(),
            nn.Linear(256, obs_encoding_size)
            )
        initial_std = 1.0
        self.transition_model_logstd = nn.Parameter(torch.full( (obs_encoding_size,),  np.log(initial_std)) )
   
    def forward(self, obs ,num_samples=None,return_distribution=False):
        '''when num samples not specified it returns a tensor of the original shape (except the last
        dimension: obs_dim -> obs_encoding_dim). When specified it adds an additional dim at the beggining and
        returns num_samples encodings stacked in that dimension'''
        obs_encoding_mean=self.network(obs)
        obs_encoding_logstd = self.transition_model_logstd.expand_as(obs_encoding_mean)
        obs_encoding_std = torch.exp(obs_encoding_logstd )
        distribution = Normal(obs_encoding_mean, obs_encoding_std)
        if num_samples:
            obs_encoding = distribution.rsample((num_samples,)) #(num_samples,obs_encoding_size)
        else:
            obs_encoding = distribution.rsample((1,)).squeeze(0) #(obs_encoding_size)
        
        if return_distribution==True:
            return obs_encoding , distribution
        else:
            return obs_encoding

class State_encoder_deterministic(nn.Module):
    def __init__(self, obs_dim,obs_encoding_size):
        super(State_encoder_deterministic,self).__init__()

        self.network=nn.Sequential( 
            nn.Linear(obs_dim, 256) ,
            nn.ReLU(),
            nn.Linear(256, obs_encoding_size),
            nn.ReLU(),
            )
        
    def forward(self, obs ,num_samples=None):
        obs_encoding=self.network(obs)

        if num_samples:
            obs_encoding=obs_encoding.unsqueeze(0)
            expand_shape = (num_samples,) + obs_encoding.shape[1:]
            return obs_encoding.expand(expand_shape)
        else:
            return obs_encoding


LOG_STD_MAX = 2
LOG_STD_MIN = -20 #-5

class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, input_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()


        self.net = mlp([input_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def get_action(self, obs, virtual_state=False, state_encoder=None,  deterministic=False, with_logprob=True,with_entropy=False , std_factor=1):
        if virtual_state==True:
            encoded_state= obs
        else:
            encoded_state = state_encoder(obs)


        net_out = self.net(encoded_state)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std) *std_factor

        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
        else:
            logp_pi = None
            
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        if with_entropy:
            return pi_action, logp_pi , pi_distribution.entropy().mean()
        else:
            return pi_action, logp_pi


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) 




##########################################################


#######  --------  MODEL NETWORK  ----------  ##########
    



class TrayEvaluator(nn.Module):
    def __init__(self,input_size, hidden_state_size=128):
        super(TrayEvaluator,self).__init__()

        self.rnn= nn.GRU(input_size, hidden_size=hidden_state_size, batch_first=False)
        self.head= nn.Sequential(
            nn.Linear(hidden_state_size,128),
            nn.ReLU(),
            nn.Linear(128,1,bias=False)
        )

    def forward(self, input_trayectories):
        # Given trayectories in a tensor of dim (len_sequence, batch_size ,input_size) , it returns a value that represent how good each one is considered by the model.
        rnn_hidden_states , last_hidden = self.rnn(input_trayectories)  
        #last_hidden has dim (1 , batch_size ,input_size) 
        output=self.head(last_hidden.squeeze(0)) # (batch_size ,1)

        return output





class Model(nn.Module):
    def __init__(self,env, len_virtual_trayectories=10, virtual_state_dim=None ):
        super(Model,self).__init__()

        ####learnable parameter  to scale the entropy loss over virtual trayectories
        ####self.entropy_loss_scale = nn.Parameter(torch.tensor(1.0))


        self.action_dim=env.action_space.shape[0] 
        self.virtual_state_dim=virtual_state_dim
        input_size_transition= virtual_state_dim + self.action_dim  
        #The transition model takes in a virtual state and an action and returns the next virtual state (it is also conditioned on the current real world state)
        self.transition_model= nn.Sequential(
            layer_init(nn.Linear(input_size_transition, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, virtual_state_dim), std=1.0),
            nn.ReLU(),
        )
        initial_std = 1.0
        self.transition_model_logstd = nn.Parameter(torch.full( (virtual_state_dim,),  np.log(initial_std)) )
   
        input_size_loss= virtual_state_dim + self.action_dim  +1  
        self.trayectory_evaluator1= TrayEvaluator(input_size=input_size_loss, hidden_state_size=128)
        self.trayectory_evaluator2= TrayEvaluator(input_size=input_size_loss, hidden_state_size=128)

        #target networks
        self.target_trayectory_evaluator1 =  deepcopy(self.trayectory_evaluator1)  
        self.target_trayectory_evaluator2 =  deepcopy(self.trayectory_evaluator2)  
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.target_trayectory_evaluator1.parameters(): 
            p.requires_grad = False  
        for p in self.target_trayectory_evaluator2.parameters(): 
            p.requires_grad = False 

        #keep track of the current state in the real and virtual MDP to better hide model's inner working from data collection function.
        #the virtual state is of shape (N, virtual_state_dim) when running N virtual tray paralelly - analogously for encoded real state
        self.current_real_state=None
        self.current_virtual_state= None
        
        #number that controls how many trayectories are run in parallel
        self.num_parallel_trayectories=None

        #some statistics to determine when to end virtual episodes
        self.num_episode_steps=0
        self.max_episode_steps=len_virtual_trayectories


    def step(self,action):

        current_virtual_state=self.current_virtual_state

        #input=torch.cat([current_virtual_state,action,encoded_curr_real_state],dim=1) 
        input=torch.cat([current_virtual_state,action],dim=-1) 

        delta_state_mean = self.transition_model(input)

        delta_state_logstd = self.transition_model_logstd.expand_as(delta_state_mean)
        delta_state_std = torch.exp(delta_state_logstd )
        distribution = Normal(delta_state_mean, delta_state_std)
        ##delta_state= distribution.rsample() 
        ##next_state= current_virtual_state + delta_state #SI NO predigo el delta (predigo el state directmente) pensar en concatenar el current state en la ultima layer del transition model
        next_state= distribution.rsample() 

        #next_state= delta_state
        #next_state=self.transition_model(input) + current_virtual_state
        

        self.current_virtual_state=next_state
        self.num_episode_steps+=1
        if self.num_episode_steps==self.max_episode_steps:
            done=True
        else:
            done=False

        return next_state , done
    
    def reset(self ,state_encoder):
        #create num_trayectories samples of the latent embedding of the real world state
        first_virtual_state=state_encoder(self.current_real_state,self.num_parallel_trayectories)
        self.current_virtual_state=first_virtual_state

        self.num_episode_steps=0

        return first_virtual_state

    def parameters(self,recurse=True):
        #override parameters() method to ignore target network parameters
        for name, param in self.named_parameters(recurse=recurse):
            if (not name.startswith('target_trayectory_evaluator1')) and (not name.startswith('target_trayectory_evaluator2')):
                yield param
    



















