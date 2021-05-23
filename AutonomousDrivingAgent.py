
################################ENVIRONMENT##################################

import gym
import highway_env

# env = gym.make("highway-v0")

from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("highway-v0", n_envs=1)

#############################AGENT############################################

from stable_baselines3 import DQN

# model = DQN.load("MyAutonomousDrivingAgent") # use an existing model, if available

model = DQN("MlpPolicy", env, verbose=2)
model.learn(total_timesteps=10, log_interval=1)


# from stable_baselines3 import PPO
# model = PPO(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=100,  log_interval=10)

# model.save("MyAutonomousDrivingAgent")
# del model # remove the model

##############################OBSERVATION######################################

observation = env.reset()
done = False
while not done:
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)
    env.render()

print("Info:", info)