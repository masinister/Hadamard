import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

class HadamardEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, dim, order, render_mode = None):
        super(HadamardEnv, self).__init__()
        self.dim = dim
        self.order = order
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Dict(
            {
                "matrix": gym.spaces.MultiDiscrete([self.order] * (self.dim ** 2)),
                "target": gym.spaces.Discrete(self.dim ** 2),
            }
        )
        self.action_space = gym.spaces.Discrete(self.order)
        self._roots = np.roots([1] + [0]*(self.order-1) + [-1])
    
    def _get_obs(self):
        return {"matrix" : self._matrix, "target" : self._target}
    
    def _get_info(self):
        complex_matrix = np.array(list(map(lambda i : self._roots[i], self._matrix))).reshape((self.dim,self.dim))
        determinant = np.linalg.det(complex_matrix)
        return {"complex_matrix" : complex_matrix, "det" : determinant}

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        self._matrix = self.observation_space["matrix"].sample()
        self._target = 0
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        self._matrix[self._target] = action
        self._target += 1
        observation = self._get_obs()
        info = self._get_info()
        terminated = self._target >= self.dim ** 2    
        if self.render_mode == "human":
            self.render()
        return observation, np.abs(info["det"]) * (1 - terminated), terminated, False, info
    
    def render(self):
        img = np.reshape(self._matrix, (self.dim, self.dim))
        fig = plt.figure(0)
        plt.clf()
        plt.imshow(img)
        fig.canvas.draw()
        plt.pause(0.001)

    def close(self):
        plt.close(0)
        return
    
if __name__ == "__main__":
    env = HadamardEnv(15, 5)
    env.reset()
    print(env._get_obs())
    obs, reward, term, done, info = env.step(0)
    print(env._get_obs())
    print(info["det"])
    print(np.abs(info["det"]))
