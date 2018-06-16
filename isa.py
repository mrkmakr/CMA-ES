import numpy as np

class isa():
    #discrete state discrete action absolute
    def __init__(self, field_size = 5, reward_radius = 2, n_session = 0, n_move = 3, tmax_trial = 5, trial_max = 100):
        """
        field_size : width and height of field
        reward_radius : radius of target area
        n_session : how many times target area change
        n_move : width and height of range of move
        """
        self.field_size = field_size
        self.reward_radius = reward_radius
        
        self.n_session = n_session
        self.n_obs = self.field_size **2 + 1
        
        ###
        self.n_move = n_move
        self.center = int((n_move-1)/2)
        self.n_act = self.n_move**2
        ###
        
        self.tmax_trial = tmax_trial
        self.trial_max = trial_max
        self.tmax= self.tmax_trial * self.trial_max
        self.trial_session_change = np.random.choice(np.arange(self.trial_max),self.n_session)
        
        self.session = 0
        self.trial = 0
        self.trial_session = 0
        self.t_trial = 0
        self.t_cum = 0
        self.reward_size = 1.0 / self.trial_max
        self.trial_result=[]
            
    def reset(self):
        #session trial t_in_trial

        self.reset_reward()
        self.reset_position()
        self.trial_session_change = np.random.choice(np.arange(self.trial_max),self.n_session)
        
        self.session = 0
        self.trial = 0
        self.t_trial = 0
        self.t_cum = 0
        self.success_trial = 0
        self.fail_trial = 0
        self.trial_result=[]
        self.pre_position_rec = []
        self.post_position_rec = []
        self.action_rec = []
        self.transform_obs()
        
        return np.append(self.position_one_hot,0)
    
    def reset_test(self):
        #reset for test
        #make a situation easy to understand
        self.reset_reward()
        self.reset_position()
        self.trial_session_change = np.int32(np.linspace(0,self.trial_max,self.n_session+1,endpoint=False))
        
        self.session = 0
        self.trial = 0
        self.t_trial = 0
        self.t_cum = 0
        self.success_trial = 0
        self.fail_trial = 0
        self.trial_result=[]
        self.pre_position_rec = []
        self.post_position_rec = []
        self.action_rec = []
        self.transform_obs()
        
        return np.append(self.position_one_hot,0)
    
    def reset_session(self):
        #session reset
        self.reset_reward()
        self.reset_position()
        self.session += 1
        self.trial_session = 0
        self.t_trial = 0
    
    def reset_trial(self):
        self.reset_position()
        self.trial += 1
        self.trial_session += 1
        self.t_trial = 0
        
    def reset_reward(self):
        rx = np.random.randint(self.field_size)
        ry = np.random.randint(self.field_size)
        
        self.reward_place = np.array([int(rx),int(ry)])
        
    def reset_position(self):
        rx = np.random.randint(self.field_size)
        ry = np.random.randint(self.field_size)
        
        self.position = np.array([rx,ry])
        self.transform_obs()
        reward, dis = self.reward_check()
        if reward == 1:
            self.reset_position()
    
    def act_proc(self,action):
        act_y =int(action / self.n_move)
        act_x = int(action % self.n_move) 
        return(np.array([act_y-self.center,act_x-self.center]))
    
    def step(self,action):
        
        self.t_cum += 1
        self.t_trial += 1
        
        flag_trial_finish = False
        flag_session_finish = False
        flag_env_finish = False
#         print('raw action', action)
#         print('pre positon', self.position)
        action = self.act_proc(action)
#         print('action', action)
        self.pre_position_rec.append(self.position)
        self.position += action
        self.post_position_rec.append(self.position)
        self.action_rec.append(action)
#         print('post positon', self.position)
        self.check_position()
#         print('checked positon', self.position)

        self.transform_obs()
        obs = self.position_one_hot
        reward,dis = self.reward_check()
        info = {'distance' : dis,
                'trial'    : self.trial,
                't_trial'  : self.t_trial,
                'session'  : self.session,
                'reward'   : self.reward_place,
                'position' : self.position,
                'action'   : action}
        
        
        if self.t_trial == self.tmax_trial or reward >0:
            flag_trial_finish = True
            if reward > 0:
                self.success_trial += 1
                self.trial_result.append(self.t_trial)
            else:
                self.fail_trial += 1
                self.trial_result.append(self.tmax_trial * 2)
                
        if flag_trial_finish == True and self.trial in self.trial_session_change:
            flag_session_finish = True
        
        
        if flag_session_finish:
            self.reset_session()
            obs = self.position_one_hot
        if flag_trial_finish:
            self.reset_trial()
            obs = self.position_one_hot
            
        
        
        done = False
        if self.trial >= self.trial_max:
            done = True    
        
        return np.append(obs,reward),reward,done,info
            
    
    def check_position(self):
        for k in range(2):
            if self.position[k] >= self.field_size:
                self.position[k] = self.field_size - 1
            if self.position[k] < 0:
                self.position[k] = 0
    def reward_check(self):
        dis = np.sqrt(np.sum((self.position - self.reward_place)**2))
        if dis < self.reward_radius:
            reward = self.reward_size
        else:
            reward = 0
        return reward,dis
    def transform_obs(self):
        self.position_one_hot = np.zeros([self.field_size, self.field_size])
        self.position_one_hot[self.position[0], self.position[1]] = 1