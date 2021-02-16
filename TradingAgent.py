
################################ENVIRONMENT##################################

import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions 
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL

# env = gym.make('forex-v0', frame_bound=(50, 100), window_size=10)
env = gym.make('stocks-v0', frame_bound=(50, 100), window_size=10)


#############################AGENT############################################

import numpy as np

from stable_baselines.deepq.policies import LnMlpPolicy
from stable_baselines import DQN

# model = DQN.load("MyRLModel")

model = DQN(LnMlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000,  log_interval=500)

# model.save("MyRLModel")

# del model # remove the model

##############################OBSERVATION######################################

import matplotlib.pyplot as plt

observation = env.reset()
done = False
while not done:
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)

print("Info:", info)
plt.cla()
env.render_all()
plt.show()