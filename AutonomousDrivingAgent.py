
################################ENVIRONMENT##################################

import gym
import highway_env

env = gym.make("highway-v0")

#############################AGENT############################################

import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

model = PPO2.load("MyRLModel") # comment out this line for the initial run

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100,  log_interval=10)

model.save("MyRLModel")

# del model # remove to demonstrate saving and loading

##############################OBSERVATION######################################

observation = env.reset()
done = False
while not done:
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)
    env.render()

print("Info:", info)