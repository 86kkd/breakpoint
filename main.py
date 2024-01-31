import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from net_work import *

# env = gym.make("CartPole-v1",render_mode='human')
# env = gym.make("ALE/Breakout-v5",render_mode="human")
env = gym.make("ALE/Breakout-v5")
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = state.shape[2]

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(4096)


episode_durations = []

steps_done = 0 

if torch.cuda.is_available():
    num_episodes = 10000
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state,EPS_DECAY,EPS_END,EPS_START,device,env,policy_net,steps_done)
        steps_done += 1
        # print(action)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        print(f"reward:{reward},    trucates:{truncated},   treminated:{terminated}\nt:{t}")
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model(memory,BATCH_SIZE,device,policy_net,target_net,optimizer,GAMMA)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        # env.render()
        if done:
            episode_durations.append(t + 1)
            print(f"episode_duration:{episode_durations[-1]}")
            plot_durations(episode_durations,is_ipython,show_result=False)
            break
torch.save(policy_net.state_dict(), 'policy_net.pt')
torch.save(target_net.state_dict(),'target_net.pt')
print('Complete')
plot_durations(episode_durations,is_ipython,show_result=False)
plt.ioff()
plt.show()
plt.savefig('foo.png')


# apt-get install -y  \
#     build-essential \
#     cmake \
#     git \
#     libglib2.0-0 \
#     ca-certificates \
#     wget \
#     curl \
#     libffi-dev \
#     libssl-dev \
#     zlib1g-dev \
#     libgl1-mesa-glx\