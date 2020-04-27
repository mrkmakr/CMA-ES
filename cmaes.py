import numpy as np
class CMAES():
    def __init__(self, func, dim, n_population, n_best, mu_init = None, cov_init = None, seed = 42, tol = 1e-8, max_iter = 100):
        self.func = func
        self.dim = dim
        self.n_population = n_population ## size of population
        self.n_best = n_best ## size of selected best
        self.seed = seed
        np.random.seed(seed)
        self.mu = mu_init or np.random.randn(dim)
        self.cov = cov_init or np.eye(dim)
        
        self.best_score = 1e10
        self.eps = 1e-8
        self.tol = tol
        self.max_iter = max_iter
        
        self.score_best_rec = []
        self.score_ave_rec = []
        
    def sample(self):
        ## sampling new population
        r = np.random.randn(self.dim)
        r = np.dot(np.linalg.cholesky(self.cov + np.eye(self.dim) * self.eps),r)
        r = r + self.mu
        return r

    def evaluation(self, params):
        score = self.func(params)
        return score
    
    def optimize(self):
        
        for _ in range(self.max_iter):
            params = np.array([self.sample() for _ in range(self.n_population)])
            scores = [self.evaluation(param) for param in params]
            best_inds = np.argsort(scores)[:self.n_best]

            params_bests = params[best_inds]
            self.cov = np.dot((params_bests-self.mu).T,params_bests-self.mu)/self.n_best
            self.mu = np.mean(params_bests,axis=0)
            
            
            best_score_tmp = np.nanmin(scores)
            if self.best_score - best_score_tmp < self.tol:
                break
            
            if self.best_score > best_score_tmp:
                self.best_score = best_score_tmp
                self.best_param = params[best_inds[0]]

            self.score_best_rec.append(self.best_score)
            self.score_ave_rec.append(np.mean(scores))
            print(self.best_score)
            

def func(x):
    return np.sum(x ** 2)

if __name__ == "__main__":
    cmaes = CMAES(func, dim = 10, n_population = 200, n_best = 20, tol = 1e-10)
    cmaes.optimize()
