from isa import isa
import time
import multiprocessing as mp
import numpy as np
from deap.benchmarks import rastrigin
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import gym

# env = gym.make("CartPole-v0")
env = isa(field_size=5,n_move=3)
n_obs = env.n_obs
n_act = env.n_act
# n_obs = env.observation_space.high.shape[0]
# n_act = env.action_space.n

# env = gym.make("MountainCar-v0")
# env = gym.make("Pendulum-v0")

#simple CMAES 
n_popu = 200
n_best = 100
n_dim = n_obs * n_act
mu = np.zeros(n_dim)
cov = np.eye(n_dim)

def sample(mu,cov = np.eye(n_dim)):   
    r = np.random.randn(n_dim)
    r = np.dot(np.linalg.cholesky(cov),r)
    r = r + mu
    return r

def eval_model(env, param, render = False):
    w = param.reshape(n_obs,n_act)
    observation = env.reset()
    reward_rec = []
    for t in range(200):
        q = np.dot(observation, w)
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





