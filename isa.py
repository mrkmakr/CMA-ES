
class isa():
    #discrete state discrete action absolute
    def __init__(self, field_size = 5, reward_radius = 1, n_session = 1, n_move = 3, tmax_trial = 5, trial_max = 100):
        """
        field_size : width and height of field
        reward_radius : radius of target area
        n_session : how many times target area change
        n_move : width and height of range of move
        """
        self.field_size = field_size
        self.reward_radius = reward_radius
        
        self.n_session = n_session
        self.n_obs = self.field_size * 2 + 1
        
        ###
        self.n_move = n_move
        self.center = int((n_move-1)/2)
        self.n_act = self.n_move**2
        ###
        
        self.tmax_trial = tmax_trial
        self.trial_max = trial_max
        self.tmax= self.tmax_trial * self.trial_max
        self.trial_session_change = np.random.choice(np.arange(self.trial_max),self.n_session - 1)
        
        self.session = 0
        self.trial = 0
        self.trial_session = 0
        self.t_trial = 0
        self.t_cum = 0
        self.reward_size = 1.0 / self.trial_max
        self.trial_result=[]
            
    def reset(self, test = False):
        #session trial t_in_trial

        self.reset_reward()
        self.reset_position()
        if test:
            self.trial_session_change = np.int32(np.linspace(0,self.trial_max,self.n_session + 1,endpoint=False))[1:]
        else:
            self.trial_session_change = np.random.choice(np.arange(self.trial_max),self.n_session - 1)
        
        self.session = 0
        self.trial = 0
        self.t_trial = 0
        self.t_cum = 0
        self.success_trial = 0
        self.fail_trial = 0
        self.trial_result=[]
        self.pre_position_rec = []
        self.post_position_rec = []
        self.hit_rec = []
        self.action_rec = []
        self.trial_num_rec = []
        self.saccade_num_rec = []
        self.cum_reward_rec = []
        self.reward_place_rec = []
        self.trial_finish_rec = []
        self.trial_start_rec = []
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
        self.pre_position_rec.append(self.position.copy())
        self.position += action
#         print('post positon', self.position)
        self.check_position()
        self.post_position_rec.append(self.position.copy())
        self.action_rec.append(action)
#         print('checked positon', self.position)

        self.transform_obs()
#         obs = self.position_one_hot
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
#             obs = self.position_one_hot
        if flag_trial_finish:
            self.reset_trial()
#             obs = self.position_one_hot
            
#         obs = self.position_one_hot
        obs = self.position_one_hot_xy
        
        
        done = False
        if self.trial >= self.trial_max:
            done = True    
        
        self.hit_rec.append(reward)
        self.trial_num_rec.append(self.trial)
        self.saccade_num_rec.append(self.t_trial)
        self.cum_reward_rec.append(np.sum(self.hit_rec))
        self.reward_place_rec.append(self.reward_place)
        self.trial_finish_rec.append(flag_trial_finish)
        self.trial_start_rec.append(self.t_trial == 1)
        
        
        
        return np.append(obs,reward),reward,done,info
            
    
    def check_position(self):
        for k in range(2):
            if self.position[k] >= self.field_size:
                self.position[k] = self.field_size - 1
            if self.position[k] < 0:
                self.position[k] = 0
    def reward_check(self):
        dis = np.sqrt(np.sum((self.position - self.reward_place)**2))
        if dis <= self.reward_radius:
            reward = self.reward_size
        else:
            reward = 0
        return reward,dis
    def transform_obs(self):
        self.position_one_hot = np.zeros([self.field_size, self.field_size])
        self.position_one_hot[self.position[0], self.position[1]] = 1
        
        self.position_one_hot_x = np.zeros(self.field_size)
        self.position_one_hot_x[self.position[0]] = 1
        self.position_one_hot_y = np.zeros(self.field_size)
        self.position_one_hot_y[self.position[1]] = 1
        self.position_one_hot_xy = np.hstack([self.position_one_hot_x, self.position_one_hot_y])
        
        
    def sample_random_action(self):
        a = np.random.randint(self.n_act)
        return a
    
    
    
    def plot_set(self,ax1):
        ax1.set_xlim([0 - 0.5, env.field_size - 0.5])
        ax1.set_ylim([0 - 0.5, env.field_size - 0.5])
        ax1.set_aspect('equal', adjustable='box')
        ax1.vlines(range(env.field_size),-0.5,env.field_size-0.5,alpha = 0.3)
        ax1.hlines(range(env.field_size),-0.5,env.field_size-0.5,alpha = 0.3)
        return ax1

    def plot(self,k,ax1):

        obs1 = self.pre_position_rec[k]
        obs2 = self.post_position_rec[k]
        hit = self.hit_rec[k]
        trial_num = self.trial_num_rec[k]
        saccade_num = self.saccade_num_rec[k]
        cum_reward = self.cum_reward_rec[k]
        reward_place = self.reward_place_rec[k]
        reward_radius = self.reward_radius
        trial_finish = self.trial_finish_rec[k]
        trial_start  = self.trial_start_rec[k]

        if trial_start:
            img1 = ax1.scatter(obs1[0],obs1[1],color = 'g')
        else:
            img1 = ax1.scatter(obs1[0],obs1[1],color = 'k')

        if hit == 0:
            if trial_finish:
                img2 = ax1.scatter(obs2[0],obs2[1],color = 'c')
            else:
                img2 = ax1.scatter(obs2[0],obs2[1],color = 'k')
        elif hit > 0:
            img2 = ax1.scatter(obs2[0],obs2[1],color = 'r')

        img3 = ax1.quiver(obs1[0],obs1[1],obs2[0] - obs1[0],obs2[1] - obs1[1],angles = 'xy', scale_units='xy', scale=1)
    #     ax1.set_title('trial : {}, saccade : {}, cum_reward : {}'.format(trial_num,saccade_num,cum_reward))
        tit = 'trial : {}, saccade : {}, cum_reward : {}'.format(trial_num,saccade_num,cum_reward)
        img4 = plt.text(-0.15, env.field_size - 1 + 0.15, tit)

        im = self.plot_circle(reward_place, reward_radius, ax1)


        return [img1,img2,img3,img4] + im

    def plot_circle(self,xy,radius,ax1,color = 'r'):
        x = xy[0]
        y = xy[1]
        xx = np.linspace(-radius, radius, 1000)
        temp = np.sqrt(radius**2 - xx**2)
        img1 = ax1.plot(xx + x, temp + y, color)
        img2 = ax1.plot(xx + x, -temp + y, color)
        return img1 + img2
    
    
    def save_animation(self,fn,save_range):
        ims = []
        fig = plt.figure(figsize=[5,5])
        ax1 = fig.add_subplot(111)
        ax1 = self.plot_set(ax1)

        n = min(save_range,len(env.cum_reward_rec))
        for k in range(n):
            img = self.plot(env,k,ax1)
            ims.append(img)
        ani = animation.ArtistAnimation(fig, ims, interval=300, repeat_delay=1000)
        ani.save(fn, writer="imagemagick")
        plt.close()