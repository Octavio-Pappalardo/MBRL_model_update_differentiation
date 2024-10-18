import numpy as np
import torch
import numpy as np
import ray

import torchopt


from data_collection import compute_loss_in_virtual_MDP , compute_loss_in_virtual_MDP_v2
from torch.distributions.normal import Normal




#######  --------  SAC LOSSES AND UPDATE  ----------  ##########
    

# Set up function for computing SAC Q-losses
def compute_q_loss(data  ,state_encoder,  policy ,q1 ,q2 , target_q1, target_q2  , gamma , alpha):
    '''Given a batch of data from the replay buffer (structured as a dict) , it computes the Q loss 
    for each of the 2 Q networks
    data batch contains tansitions : data['obs'], data['act'], data['rew'], data['obs2'], data['done'] '''

    q1_on_obs_act = q1(data['obs'] , data['act'])
    q2_on_obs_act = q2(data['obs'] , data['act'])

    # Bellman backup for Q functions
    with torch.no_grad():
        # Target actions come from *current* policy
        act2, logp_a2 = policy.get_action(data['obs2'] , state_encoder=state_encoder)

        # Target Q-values
        target_q1_on_obs2_act2 = target_q1(data['obs2'], act2)
        target_q2_on_obs2_act2 = target_q2(data['obs2'], act2)

        min_q_on_obs2_act2 = torch.min(target_q1_on_obs2_act2, target_q2_on_obs2_act2)
        backup = data['rew'] + gamma * (1 - data['done']) * (min_q_on_obs2_act2 - alpha * logp_a2) 

    # MSE loss against Bellman backup
    loss_q1 = ((q1_on_obs_act - backup)**2).sum()#.mean()
    loss_q2 = ((q2_on_obs_act - backup)**2).sum()#mean()
    loss_q = loss_q1 + loss_q2

    # Useful info for logging
    q_info = dict(q1_mean= np.mean(q1_on_obs_act.detach().numpy()) , q1_std = np.std(q1_on_obs_act.detach().numpy()) ,
                    q2_mean= np.mean(q2_on_obs_act.detach().numpy()) , q2_std = np.std(q2_on_obs_act.detach().numpy())  )

    return loss_q, q_info




# Set up function for computing SAC pi loss
def compute_policy_loss(state_encoder, data , policy ,q1 ,q2 ,alpha):

    action_preds, logp_actions , entropy = policy.get_action(data['obs'] ,state_encoder=state_encoder ,with_entropy=True)
    q1_on_obs_action_preds = q1(data['obs'], action_preds)
    q2_on_obs_action_preds = q2(data['obs'], action_preds)
    q_on_policy_preds = torch.min(q1_on_obs_action_preds, q2_on_obs_action_preds)

    # Entropy-regularized policy loss
    policy_loss = (alpha * logp_actions - q_on_policy_preds).sum()

    # Useful info for logging
    pi_info = dict(logprob=np.mean(logp_actions.detach().numpy()) , entropy=entropy.detach().numpy())

    return policy_loss, pi_info




def consistency_loss_prob_encoder(batch,state_encoder,model,config,num_samples):

    #encode real state 
    with torch.no_grad():
        state_encoding=state_encoder(batch['obs'],num_samples=num_samples).view(-1,config.virtual_state_dim)  #(batch_size * num_samples , virtual_state_dim)

    #get actions executed 
    expand_shape = (num_samples,) + batch['act'].shape
    actions= batch['act'].unsqueeze(0).expand(expand_shape)
    actions=actions.reshape(-1,batch['act'].shape[1]) #(batch_size * num_samples , action_dim)

    # apply transition to get a distribution over the next state encoding
    input=torch.cat([state_encoding,actions],dim=1)
    transition_mean=model.transition_model(input)
    transition_logstd = model.transition_model_logstd.expand_as(transition_mean)
    transition_std = torch.exp(transition_logstd)
    transition_distribution = Normal(transition_mean, transition_std)  #--(batch_size * num_samples , action_dim)

    #encode real next state
    with torch.no_grad():
        next_state_encoding, encoder_distribution =state_encoder(batch['obs2'],num_samples=num_samples,return_distribution=True)
        # reshape steps
        mean = encoder_distribution.mean
        std = encoder_distribution.stddev
        expanded_mean = mean.repeat(num_samples, 1) 
        expanded_std = std.repeat(num_samples, 1)    
        encoder_distribution = Normal(expanded_mean, expanded_std)  #--(batch_size * num_samples , action_dim)

    # Calculate the KL divergence
    kl_divergence = torch.distributions.kl_divergence(encoder_distribution,transition_distribution)
    
    return kl_divergence.mean(dim=-1).sum()





def consistency_loss_deter_encoder(batch,state_encoder,model,config,loss_type='mle' ,encoder_std=0.3):

    #encode real state 
    with torch.no_grad():
        state_encoding=state_encoder(batch['obs']) #(batch_size , virtual_state_dim)

    # apply transition to get a distribution over the next state encoding
    input=torch.cat([state_encoding,batch['act']],dim=1)
    transition_mean=model.transition_model(input)
    transition_logstd = model.transition_model_logstd.expand_as(transition_mean)
    transition_std = torch.exp(transition_logstd)
    transition_distribution = Normal(transition_mean, transition_std)  #

    #encode real next state
    with torch.no_grad():
        next_state_encoding =state_encoder(batch['obs2'])
        #print(next_state_encoding.shape)
    if loss_type=='kl':
        std=torch.full_like(next_state_encoding, encoder_std) 
        encoder_distribution = Normal(next_state_encoding, std)
        # Calculate the KL divergence
        kl_divergence = torch.distributions.kl_divergence(encoder_distribution, transition_distribution)
        loss=kl_divergence.mean(dim=-1).sum()
    elif loss_type=='mle':
        nll = -transition_distribution.log_prob(next_state_encoding)
        loss=nll.mean(dim=-1).sum()

    return loss












#######  --------  DISTRIBUTED version of SAC UPDATE  ----------  ##########
    
        
def combine_gradients(gradient_list ,aggregation_method='mean'):
    """Combines gradients from a list by averaging them for each parameter.
    Args:
        gradient_list: A list of gradients, where each element is a list of
            tensors containing gradients for a set of parameters.
    Returns:
        A list of tensors containing the combined gradients, where each tensor
        has the same shape as the corresponding tensor in the input gradients.
    """
    combined_gradients = []
    for gradients_per_param in zip(*gradient_list):
        if aggregation_method=='mean':
            combined_gradient = torch.mean(torch.stack(gradients_per_param), dim=0)
        elif aggregation_method=='sum':
            combined_gradient = torch.sum(torch.stack(gradients_per_param), dim=0)
        combined_gradients.append(combined_gradient)

    return combined_gradients

#######################



def _losses_gradients_for_batch(batch , state_encoder,policy ,model ,q1 ,q2 , target_q1, target_q2, config):
    q_params = list(q1.parameters()) + list(q2.parameters())
    #adapt base policy conditioning on the first state of the batch
    adaptation_optimizer =torchopt.MetaSGD(policy, lr=config.adaptation_lr)
    for l in range(config.num_updates_in_adaptation):
        policy_loss_for_adaptation=compute_loss_in_virtual_MDP(state_encoder=state_encoder,policy=policy, model=model , current_real_state=batch['obs'][0] ,
                                                                 config=config, device='cpu')
        adaptation_optimizer.step(policy_loss_for_adaptation)

    #compute qs loss on the batch
    q_loss, q_info = compute_q_loss(data=batch ,state_encoder=state_encoder, policy=policy ,q1=q1 ,q2=q2 , target_q1=target_q1 ,
                    target_q2=target_q2  , gamma=config.gamma , alpha=config.alpha)
    q_loss_grad=torch.autograd.grad(q_loss, q_params)

    #compute policy loss on the batch (dont keep track of gradients for the q functions)
    for p in q_params:
        p.requires_grad = False
    policy_loss, pi_info = compute_policy_loss(data=batch, state_encoder=state_encoder, policy=policy ,q1=q1 ,q2=q2 ,alpha=config.alpha)
    for p in q_params:
        p.requires_grad = True

    #normalize losses  - this normalization makes every batch contribute the same (note not every step contributes the same because some batches have more steps than other (but the data in each batch is more correlated))
    ##q_loss= q_loss/ len(batch['done'])
    ##policy_loss= policy_loss / len(batch['done'])

    #for p in state_encoder.parameters():
    #    p.requires_grad = False
    if config.stochastic_encoder==True:
        model_consistency_loss= consistency_loss_prob_encoder(batch,state_encoder,model,config,num_samples=5)#!!!
    else:
        model_consistency_loss= consistency_loss_deter_encoder(batch,state_encoder,model,config,loss_type='mle')#!!!
    #for p in state_encoder.parameters():
    #    p.requires_grad = True


    loss2= policy_loss + config.constistency_loss_coef * model_consistency_loss

    policy_loss_grads=torch.autograd.grad(loss2, list(state_encoder.parameters()) + list(model.parameters()) + list(policy.parameters()) ) 

    info={'q loss':q_loss.item() , 'policy loss': policy_loss.item() , 'model_consistency_loss':model_consistency_loss.item(), 'q1 mean': q_info['q1_mean']  , 'q1 std' : q_info['q1_std'] , 
          'q2 mean': q_info['q2_mean']  , 'q2 std' : q_info['q2_std']  , 'act logp': pi_info['logprob'] , 'entropy': pi_info['entropy']}

    return q_loss_grad , policy_loss_grads , info



remote_losses_gradients_for_batch= ray.remote(_losses_gradients_for_batch)

def distributed_sac_update(replay_buffer,state_encoder, policy , model ,q1 ,q2 , target_q1, target_q2  ,state_encoder_optimizer, policy_optimizer ,qs_optimizer, model_optimizer ,
                config, logger=None):
    q_params = list(q1.parameters()) + list(q2.parameters())
    list_of_batches, total_states =replay_buffer.sample_list_of_batches(num_elements=config.num_batches_of_consecutive_elements_per_sac_update 
                                                         , max_steps_per_element=config.num_real_mdp_steps_per_adaptation)
    #each batch in the list contains data from consecutive steps

    state_encoder_ref=ray.put(state_encoder)
    policy_ref=ray.put(policy)
    model_ref=ray.put(model)
    config_ref=ray.put(config)
    q1_ref=ray.put(q1)
    q2_ref=ray.put(q2)
    target_q1_ref=ray.put(target_q1)
    target_q2_ref=ray.put(target_q2)


    inputs=[[batch,state_encoder_ref ,policy_ref ,model_ref , q1_ref, q2_ref, target_q1_ref, target_q2_ref, config_ref] for batch in list_of_batches ]

    q_loss_gradients , policy_loss_gradients ,infos = zip(*ray.get([remote_losses_gradients_for_batch.options(num_cpus=1).remote(*i) for i in inputs])) 

    q_loss_gradients= combine_gradients(q_loss_gradients,aggregation_method='sum')
    q_loss_gradients= tuple(grad / total_states for grad in q_loss_gradients)
    policy_loss_gradients=  combine_gradients(policy_loss_gradients,aggregation_method='sum')
    policy_loss_gradients= tuple(grad / total_states for grad in policy_loss_gradients)


    #update
    qs_optimizer.zero_grad()
    policy_optimizer.zero_grad() 
    state_encoder_optimizer.zero_grad()
    model_optimizer.zero_grad()
    for p, grad in zip(q_params, q_loss_gradients):
        p.grad = grad
    for p, grad in zip( list(state_encoder.parameters()) + list(model.parameters()) +  list(policy.parameters()) , policy_loss_gradients): 
        p.grad = grad
    qs_optimizer.step()
    policy_optimizer.step() 
    state_encoder_optimizer.step()
    model_optimizer.step()

    # update target networks by polyak averaging.
    with torch.no_grad():
        for p, p_targ in zip(q1.parameters(), target_q1.parameters()):
            p_targ.data.mul_(config.polyak)
            p_targ.data.add_((1 - config.polyak) * p.data)
        for p, p_targ in zip(q2.parameters(), target_q2.parameters()):
            p_targ.data.mul_(config.polyak)
            p_targ.data.add_((1 - config.polyak) * p.data)
 
    #log metrics
    if logger:
        logger.q_loss.append( np.sum( [info['q loss'] for info in infos])/total_states )
        logger.policy_loss.append( np.sum( [info['policy loss'] for info in infos])/total_states )
        logger.model_consistency_loss.append( np.sum( [info['model_consistency_loss'] for info in infos])/total_states )
        logger.q1_means.append( np.mean( [info['q1 mean'] for info in infos]) )
        logger.q1_stds.append( np.mean( [info['q1 std'] for info in infos]) )
        logger.q2_means.append( np.mean( [info['q2 mean'] for info in infos]) )
        logger.q2_stds.append( np.mean( [info['q2 std'] for info in infos]) )
        logger.actions_logprobs.append( np.mean( [info['act logp'] for info in infos]) )
        logger.entropies.append( np.mean( [info['entropy'] for info in infos]) )




##########################################################


#######  --------  For collecting virtual data and updating base model on it  ----------  ##########
    
def update_agent_with_virtual_data_v1(replay_buffer , num_states_to_consider, state_encoder, policy , model , policy_optimizer , config, logger=None):
    for p in model.parameters():
        p.requires_grad = False
    for p in state_encoder.parameters():
        p.requires_grad=False    

    data=replay_buffer.sample_standard_batch(batch_size=num_states_to_consider)
    '''loss=0
    for i in range(len(data['obs'])):
        loss_on_virtual_data=compute_loss_in_virtual_MDP(state_encoder=state_encoder, policy=policy, model=model , current_real_state=data['obs'][i], config=config, device='cpu',for_policy_update=True)
        loss+=loss_on_virtual_data
    loss=loss/num_states_to_consider'''
    virtual_loss=compute_loss_in_virtual_MDP_v2(state_encoder=state_encoder, policy=policy , model=model
                                    , observations=data['obs'], config=config,device='cpu',for_policy_update=True)

    #kl loss
    _ , distribution =state_encoder(data['obs'],return_distribution=True)
    if isinstance(distribution, Normal):
        old_distribution = Normal(loc=distribution.loc.detach(), scale=distribution.scale.detach())
    kl=torch.distributions.kl.kl_divergence(distribution, old_distribution).mean()
    loss=virtual_loss + config.kl_coef * kl

    policy_optimizer.zero_grad()
    loss.backward()
    policy_optimizer.step()
    policy_optimizer.zero_grad()

    for p in model.parameters():
        p.requires_grad = True
    for p in state_encoder.parameters():
        p.requires_grad=True
    if logger:
        logger.virtual_loss.append(virtual_loss.item())
        logger.kl_regu_loss.append(kl.item())










def _adapt_and_compute_virtual_loss_grads(state_encoder, policy, model , current_real_state, config):
    #adapt base policy
    adaptation_optimizer =torchopt.MetaSGD(policy, lr=config.adaptation_lr)
    for l in range(config.num_updates_in_adaptation):
        policy_loss_for_adaptation=compute_loss_in_virtual_MDP(state_encoder=state_encoder,policy=policy, model=model , current_real_state=current_real_state ,
                                                                config=config, device='cpu')
        adaptation_optimizer.step(policy_loss_for_adaptation)
    #loss for update
    loss_on_virtual_data=compute_loss_in_virtual_MDP(state_encoder=state_encoder,policy=policy, model=model , current_real_state=current_real_state ,
                                                                 config=config, device='cpu',for_policy_update=True)
    #params_to_optimize=[param for param in policy.parameters() if param not in policy.state_encoder.parameters()]
    #params_to_optimize= list(policy.net.parameters()) + list(policy.mu_layer.parameters()) + list(policy.log_std_layer.parameters())
    params_to_optimize=policy.parameters()
    virtual_loss_grads=torch.autograd.grad(loss_on_virtual_data, params_to_optimize )

    info={'virtual loss': loss_on_virtual_data.item() }

    return virtual_loss_grads , info



remote_adapt_and_compute_virtual_loss_grads= ray.remote(_adapt_and_compute_virtual_loss_grads)

def update_agent_with_virtual_data_v2(replay_buffer , num_states_to_consider,state_encoder, policy , model , policy_optimizer ,config, logger=None):
    for p in model.parameters():
        p.requires_grad = False
    for p in state_encoder.parameters():
        p.requires_grad=False

    
    data=replay_buffer.sample_standard_batch(batch_size=num_states_to_consider)
    state_encoder_ref=ray.put(state_encoder)
    policy_ref=ray.put(policy)
    model_ref=ray.put(model)
    config_ref=ray.put(config)
    inputs=[[state_encoder_ref, policy_ref ,model_ref , data['obs'][i], config_ref] for i in range(len(data['obs'])) ]
    virtual_loss_gradients , infos = zip(*ray.get([remote_adapt_and_compute_virtual_loss_grads.options(num_cpus=1).remote(*i) for i in inputs]))

    virtual_loss_gradients= combine_gradients(virtual_loss_gradients)


    policy_optimizer.zero_grad()
    #params_to_optimize=[param for param in policy.parameters() if param not in policy.state_encoder.parameters()]
    #params_to_optimize= list(policy.net.parameters()) + list(policy.mu_layer.parameters()) + list(policy.log_std_layer.parameters())
    params_to_optimize=policy.parameters()
    for p, grad in zip(params_to_optimize, virtual_loss_gradients):
        p.grad = grad
    policy_optimizer.step()
    policy_optimizer.zero_grad()


    for p in model.parameters():
        p.requires_grad = True
    for p in state_encoder.parameters():
        p.requires_grad=True
    if logger:
        virtual_loss=np.mean( [info['virtual loss'] for info in infos])
        logger.virtual_loss.append(virtual_loss)





#############

#model target networks update
def update_model_target_networks(model,config):
    # update target networks by polyak averaging.
    with torch.no_grad():
        for p, p_targ in zip(model.trayectory_evaluator1.parameters(), model.target_trayectory_evaluator1.parameters()):
            p_targ.data.mul_(config.polyak)
            p_targ.data.add_((1 - config.polyak) * p.data)
        for p, p_targ in zip(model.trayectory_evaluator2.parameters(), model.target_trayectory_evaluator2.parameters()):
            p_targ.data.mul_(config.polyak)
            p_targ.data.add_((1 - config.polyak) * p.data)




###############################################################


#not distributed version
def sac_update(replay_buffer,state_encoder, policy , model ,q1 ,q2 , target_q1, target_q2  , policy_optimizer ,qs_optimizer, model_optimizer ,
                config, logger=None):
    q_params = list(q1.parameters()) + list(q2.parameters())
    base_policy_state_dict = torchopt.extract_state_dict(policy)
    list_of_batches=replay_buffer.sample_list_of_batches(num_elements=config.num_batches_of_consecutive_elements_per_sac_update 
                                                         , max_steps_per_element=config.num_real_mdp_steps_per_adaptation)
    #each batch in the list contains data from consecutive steps

    total_q_loss=0
    total_policy_loss=0
    for batch in list_of_batches:

        #adapt base policy conditioning on the first state of the batch
        adaptation_optimizer =torchopt.MetaSGD(policy, lr=config.adaptation_lr)
        for l in range(config.num_updates_in_adaptation):
            policy_loss_for_adaptation=compute_loss_in_virtual_MDP(state_encoder=state_encoder,policy=policy, model=model , current_real_state=batch['obs'][0] ,
                                                                    config=config, device='cpu')
            adaptation_optimizer.step(policy_loss_for_adaptation)

        #compute qs loss on the batch
        q_loss, q_info = compute_q_loss(data=batch , state_encoder=state_encoder, policy=policy ,q1=q1 ,q2=q2 , target_q1=target_q1 ,
                        target_q2=target_q2  , gamma=config.gamma , alpha=config.alpha)

        #compute policy loss on the batch (dont keep track of gradients for the q functions)
        for p in q_params:
            p.requires_grad = False
        policy_loss, pi_info = compute_policy_loss(data=batch, state_encoder=state_encoder, policy=policy ,q1=q1 ,q2=q2 ,alpha=config.alpha)
        for p in q_params:
            p.requires_grad = True

        #normalize losses  - this normalization makes every batch contribute the same (note not every step contributes the same because some batches have more steps than other (but the data in each batch is more correlated))
        q_loss= q_loss/ len(batch['done'])
        policy_loss= policy_loss / len(batch['done'])

        #add losses to total loss
        total_q_loss+=q_loss
        total_policy_loss+=policy_loss

        #recover to base policy parameters
        torchopt.recover_state_dict(policy, base_policy_state_dict)


    #normalize loss by num batches
    total_q_loss=total_q_loss/ len(list_of_batches)
    total_policy_loss=total_policy_loss/ len(list_of_batches)

    #log metrics
    if logger:
        logger.q_loss.append(total_q_loss.item())
        logger.policy_loss.append(total_policy_loss.item())
    
    #perform the gradient steps
    qs_optimizer.zero_grad()
    policy_optimizer.zero_grad()
    model_optimizer.zero_grad()
    policy_loss.backward()
    total_q_loss.backward()
    qs_optimizer.step()
    policy_optimizer.step()
    model_optimizer.step()


    # update target networks by polyak averaging.
    with torch.no_grad():
        for p, p_targ in zip(q1.parameters(), target_q1.parameters()):
            p_targ.data.mul_(config.polyak)
            p_targ.data.add_((1 - config.polyak) * p.data)
        for p, p_targ in zip(q2.parameters(), target_q2.parameters()):
            p_targ.data.mul_(config.polyak)
            p_targ.data.add_((1 - config.polyak) * p.data)
