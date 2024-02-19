import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from net_work import *
from train import train 

# env = gym.make("CartPole-v1",render_mode='human')
env = gym.make("ALE/Breakout-v5",render_mode="human")
# env = gym.make("ALE/Breakout-v5")
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()

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

policy_net,target_net,episode_durations = train(
    env,device,memory,policy_net,
    EPS_DECAY,EPS_END,EPS_START,BATCH_SIZE,GAMMA,
    optimizer,is_ipython,target_net,TAU)

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
