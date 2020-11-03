import gym 


env = gym.make('CarRacing-v0', verbose=0)
s = env.reset()

while True:
    state, r, done, _ = env.step(env.action_space.sample())