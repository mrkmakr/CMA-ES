import numpy as np

def create_sec(start,reverse_prob,n):
    correct=[start]
    now=start
    for _ in range(n-1):
        if np.random.rand()<reverse_prob:
            now=1-now
        correct.append(now)
        
    return correct

class murao():
    def __init__(self):
        self.n_obs = 3
        self.n_act = 2
        #self.reverse_interval = 25
        self.tmax = 150
        self.p_reverse_cand =  [0.005,0.01,0.025,0.05]
        self.p_cand = [0.7,0.8,0.9]
        self.t = 0
        self.p_reverse = np.random.choice(self.p_reverse_cand)
        self.p = np.random.choice(self.p_cand)
        self.correct_seq = create_sec(np.random.randint(2),self.p_reverse,self.tmax)
        self.reward_size = 1.0/self.tmax
        #self.reward_size = 1.0
    def seed(self,seed):
        np.random.seed(seed)
    def reset(self,same = False):
        self.t  = 0
        obs = np.array([0,0,0])
        
        if same == False:
            self.p_reverse = np.random.choice(self.p_reverse_cand)
            self.p = np.random.choice(self.p_cand)
        self.correct_seq = create_sec(np.random.randint(2),self.p_reverse,self.tmax)
        return obs
    
    def reset25(self):
        self.t=0
        obs = np.array([0,0,0])
        self.correct_seq = [0] * 25 + [1] * 25
        self.correct_seq *= 3
        self.p = 0.8
        
        return obs
    def step(self,action):
        reward = self.return_reward(action)
        
        
        
        if action == 0:
            obs = np.array([1,0,reward])
        elif action == 1:
            obs = np.array([0,1,reward])
        reward = reward * self.reward_size


        
        self.t += 1
        if self.t == self.tmax:
            done = True
        else:
            done = False
        
        info = {}
        
        return obs,reward,done,info
        
    def return_reward(self,action):
        if self.correct_seq[self.t] == action:
            p = self.p
        else:
            p = 1 - self.p

        if np.random.rand() < p:
            return 1
        else:
            return 0