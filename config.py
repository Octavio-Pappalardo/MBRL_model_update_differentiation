class Config_custom:
    def __init__(self):
        self.stochastic_encoder=True
        self.constistency_loss_coef=0.1
        self.std_factor=1.5 #factor that increases the randomness when taking actions in the virtual mdp vs when taking them on the real world.
        ###self.kl_coef=0.1

        self.seeding=False
        self.seed=1
        self.device_real_world='cpu' 
        self.device_virtual_world='cpu'


        self.total_timesteps=200000
        self.num_real_mdp_steps_per_update= 50 # controls the total number of steps in the real mdp that are taken for each iteration of the algorithm
        self.num_real_mdp_steps_per_adaptation=10 #M - controls how often the base policy is readapted using the model

        self.num_updates_in_adaptation=1 #controls how many updates are done to a policy an adaptation phase (with virtual data from model)
        self.num_virtual_trayectories_per_adaptation_update=30
        self.len_virtual_trayectories= 10 

        self.num_steps_to_start_updating_after= 1000
        self.num_initial_random_steps= 10000

        self.policy_lr=3e-4 
        self.q_func_lr=1e-3
        self.model_lr=3e-4
        self.encoder_lr=3e-4

        self.adaptation_lr=0.05 

        self.replay_buffer_size=int(1e5) 
        self.num_batches_of_consecutive_elements_per_sac_update= 30 #total steps consider per update is aprox :   num_real_mdp_steps_per_adaptation * num_batches_of_consecutive_elements_per_sac_update
        self.updates_to_steps_ratio=1 #how many sac and policy updates are taken per each step in real world. must be int >=1

        self.virtual_state_dim=30
        self.action_encoding_dim=None 



        self.gamma=0.99
        self.alpha=0.2  #entropy coef
        self.polyak=0.995

        self.num_test_episodes=10    
        self.test_every_steps_num =4000  # every how many real steps to test the base policy and the adapted policy in a deterministic manner




def get_config(config_settings):
    if config_settings=='custom':
        return Config_custom()
    else:
        raise ValueError(f"Unsupported config_setting: {config_settings}")