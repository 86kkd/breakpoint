import torch
from itertools import count
from net_work import *

def train(env,device,memory,policy_net,EPS_DECAY,EPS_END,EPS_START,BATCH_SIZE,GAMMA,optimizer,is_ipython,target_net,TAU):
    episode_durations = []
    steps_done = 0 

    if torch.cuda.is_available():
        num_episodes = 10000
    else:
        num_episodes = 50
    for _ in range(num_episodes):
        # Initialize the environment and get its state
        state, _ = env.reset()
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
