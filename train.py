import gym_examples
import gymnasium
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO,A2C
import os

model = "A2C"
models_dir = f'models/{model}'
logdir = 'logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gymnasium.make('gym_examples/GridWorld-v0')
env.reset()

model = A2C('MlpPolicy',env,tensorboard_log=logdir)

TIMESTEPS = 10000

for i in range(1,1000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name='a2c',progress_bar=True)
    model.save(f"{models_dir}/{TIMESTEPS*i}")

