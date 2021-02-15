
################################ENVIRONMENT##################################

import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions 
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt

# env = gym.make('forex-v0', frame_bound=(50, 100), window_size=10)
env = gym.make('stocks-v0', frame_bound=(50, 100), window_size=10)


#############################AGENT############################################

import numpy as np

from stable_baselines.deepq.policies import LnMlpPolicy
from stable_baselines import DQN

model = DQN(LnMlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000,  log_interval=500)

# model.save("MyRLModel")

# del model # remove to demonstrate saving and loading

# model = DQN.load("MyRLModel")

##############################OBSERVATION######################################

observation = env.reset()
while True:
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)
    # env.render()
    if done:
        print("Info:", info)
        break

plt.cla()
env.render_all()
plt.show()