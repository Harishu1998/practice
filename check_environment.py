from stable_baselines3.common.env_checker import check_env
import gymnasium
import gym_examples

env = gymnasium.make('gym_examples/GridWorld-v0')
env.reset()
print(env.step(env.action_space.sample()))

