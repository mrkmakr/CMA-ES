from isa import isa
import time
import multiprocessing as mp
import numpy as np
from deap.benchmarks import rastrigin
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym

def randn(shape):
    r = np.random.randn(shape[0],shape[1])
    return r

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    x[x<0]=0
    return x

class gru_lin():
    def __init__(self,n_inp,n_state,n_out):
        self.n_inp = n_inp
        self.n_state = n_state
        self.n_out = n_out
        self.n_instate = n_inp + n_state + 1
        
        self.n_dims = [(self.n_state, self.n_instate),
                      (self.n_state, self.n_instate),
                      (self.n_state, self.n_instate),
                      (self.n_out, self.n_state)]

        self.n_dims_flatten = [a[0] * a[1] for a in self.n_dims]
        self.n_dims_cum = np.hstack([0,np.cumsum(self.n_dims_flatten)])
        self.n_dim = sum(self.n_dims_flatten)
        
        self.param = np.random.randn(self.n_dim)      
        self.set_param(self.param)
        
        self.state = np.zeros(self.n_state)
        self.state_rec = []
    
    def reset(self):
        self.state = np.zeros(self.n_state)
        self.state_rec = []
    
    def set_param(self,param):
        
        if self.n_dim != len(param):
            print('error : not match , n_dim = {}, n_param = {}'.format(self.n_dim,len(param)))
        else:
            self.param = param
            for k in range(4):
                k = 0
                self.W_z = param[self.n_dims_cum[k]:self.n_dims_cum[k+1]].reshape(self.n_dims[k])
                k = 1
                self.W_r = param[self.n_dims_cum[k]:self.n_dims_cum[k+1]].reshape(self.n_dims[k])
                k = 2
                self.W = param[self.n_dims_cum[k]:self.n_dims_cum[k+1]].reshape(self.n_dims[k])
                k = 3
                self.lin = param[self.n_dims_cum[k]:self.n_dims_cum[k+1]].reshape(self.n_dims[k])
        
    def forward(self,inp):
        instate = np.hstack([inp,self.state,1])
        z = sigmoid(self.W_z.dot(instate))
        r = sigmoid(self.W_r.dot(instate))
        temp = np.hstack([r * self.state,inp,1])
        state_update = self.W.dot(temp)
        self.state = (1-z) * self.state + z * state_update
        self.state = relu(self.state)
        self.state_rec.append(self.state)
        out = self.lin.dot(self.state)
        
        return out
        
        

env = isa(field_size=5,n_move=3)
n_obs = env.n_obs
n_act = env.n_act

# env = gym.make("CartPole-v0")
# n_obs = env.observation_space.high.shape[0]
# n_act = env.action_space.n

# env = gym.make("MountainCar-v0")
# env = gym.make("Pendulum-v0")

params_for_gru = (n_obs, 10, n_act)
model = gru_lin(*params_for_gru)
n_dim = model.n_dim
print('param_num :', n_dim)

#simple CMAES 
n_popu = 200
n_best = 10
mu = np.zeros(n_dim)
cov = np.eye(n_dim)

def sample(mu, cov = np.eye(n_dim)):   
    r = np.random.randn(n_dim)
    r = np.dot(np.linalg.cholesky(cov),r)
    r = r + mu
    return r

def eval_model(env, param, params_for_gru = params_for_gru, render = False):
    model = gru_lin(*params_for_gru)
    model.reset()
    model.set_param(param)
    
    observation = env.reset()
    reward_rec = []
    for t in range(200):
#         q = np.dot(observation, w)
        q = model.forward(observation)
        action = np.argmax(q)
#         action = q
        observation, reward, done, info = env.step(action)
        reward_rec.append(reward)
        if render:
            env.render()
        if done:
            break
    r = np.sum(reward_rec)
    return r

def eval_model_pool(param):
    env = isa(field_size=5,n_move=3)
    score = eval_model(env,param)
    return score

t = time.time()
best_score = -10000
imgs = []
score_best_rec = []
score_ave_rec = []
multi_cpu = False

# fig = plt.figure()
for k in range(100):
    params = np.array([sample(mu,cov) for _ in range(n_popu)])
    
    if multi_cpu:
        pool = mp.Pool(5)
        scores = pool.map(eval_model_pool,params)
    else:
        scores = [eval_model(env,param) for param in params]
    best_inds = np.argsort(scores)[::-1][:n_best]
    
    params_bests = params[best_inds]
#     cov = np.dot((params_bests-mu).T,params_bests-mu)/n_best
    cov = np.diag(np.std(params_bests,axis=0))
    mu = np.mean(params_bests,axis=0)

#     img = plot(params,best_inds,mu)
#     imgs.append(img)
    
    if best_score < np.max(scores):
        best_score = np.max(scores)
    score_best_rec.append(best_score)
    score_ave_rec.append(np.mean(scores))
    print('generation',k)
    print('best',best_score)
    print('ave',np.mean(scores))
    print('cov_amp',np.mean(cov**2))
    print('-----')
# ani = animation.ArtistAnimation(fig, imgs)
# ani.save('CMAES_search.gif', writer="imagemagick")
# plt.close()

print('best:',best_score)
print('time:',time.time() - t)

# eval_model(env,params_bests[0],render = True)
plt.plot(score_best_rec,label = 'best')
plt.plot(score_ave_rec,label = 'ave')
plt.xlabel('generation')
plt.ylabel('score')
plt.show()