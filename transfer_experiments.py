from cmath import exp
import random
import numpy as np
import torch
import pickle

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from transfer_utils import FourRoomsNoReward, NormalizedTransitionsDataset, get_eigen_options, \
    get_eigen_reward, solve_task_with_options, train_dsaa, train_dsaa_options, solve_dsaa_task, \
        train_contrastive_encoder, solve_contrastive_task, get_dsaa_indiv_options

from utils import ReplayBuffer, get_nbrs

def train(exp_type = "eigenoptions"):
    # ------------- 1. Explore Environment -------------
    env_config = {
        "max_steps": 100000 # no reset
    }
    env = FourRoomsNoReward(env_config)

    print("**Exploring Environment**")
    data = []
    replay_buffer = ReplayBuffer(env_config["max_steps"])
    state = env.reset()
    for _ in range(env_config["max_steps"]):
        action = env.action_space.sample()
        next_state , _, done, _ = env.step(action)
        # print(state)
        data.append([state, next_state])
        
        replay_buffer.add((state, next_state, action, 0, 0))
        
        if False:#done:
            state = env.reset()
        else:
            state = next_state
    print("num samples:", len(replay_buffer))
    # ------------- 2. Train abstraction -------------
    print("**Training Abstraction**")
    if exp_type == "dsaa":
        print("DSAA")
        phi = train_dsaa(replay_buffer, config={})

        if True:
            env_grid = (env.example_obs == 1)*1.0
            # print(env_grid)
            with torch.no_grad():
                all_phis = torch.zeros((19, 19))
                for i in range(env_grid.shape[1]):
                    for j in range(env_grid.shape[2]):
                        if env_grid[0,i,j] > 0:
                            all_phis[i,j] = -1
                            continue
                        
                        tmp_grid = torch.FloatTensor([i,j])
                        tmp_enc = phi(tmp_grid.unsqueeze(0))
                        all_phis[i,j] = torch.argmax(tmp_enc[0])
            
            plt.imshow(all_phis)
            plt.colorbar()
            plt.savefig("tmp_data/dsaa_abstraction.png")

    elif exp_type == "contrastive":
        print("CONTRASTIVE")
        batch_size = 128
        # NOTE: we are normalizing the data here
        dataset = NormalizedTransitionsDataset(transitions=data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print("**Training Contrastive Encoder**")
        contrastive_encoder = train_contrastive_encoder(dataloader, z_dim=2)
        if True:
            with torch.no_grad():
                xx,yy = np.meshgrid(np.linspace(0,18,19), np.linspace(0,18,19))
                all_states = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)], axis=1).astype(int)
                env_grid = env.example_obs[0]
                plot_points = [[], []]
                for i,j in np.random.permutation(all_states):
                    if env_grid[i,j] == 1:
                        continue
                    tmp_tensor = torch.FloatTensor([[i,j]]) / 18.0
                    tmp_enc = contrastive_encoder(tmp_tensor)
                    plot_points[0].append(tmp_enc[0][0])
                    plot_points[1].append(tmp_enc[0][1])

                plot_points = np.array(plot_points)
                plt.scatter(plot_points[0,:], plot_points[1, :])
                plt.savefig("tmp_data/contrastive_2d_encoding.png")

    elif exp_type == "eigenoptions":
        print("EIGENOPTIONS")
        env_grid = env.example_obs[0]
        eigen_reward = get_eigen_reward(env_grid)

    # ------------- 3. Train options -------------
    print("**Training Options**")
    if exp_type == "dsaa":
        plt.clf()
        _, abstract_adjacency = train_dsaa_options(phi, replay_buffer, config={})
        env_grid = env.example_obs[0]
        option_policies = get_dsaa_indiv_options(env_grid, phi, abstract_adjacency)
        # print(option_policies)
        # print(option_policies.keys())

    elif exp_type == "eigenoptions":
        num_options = 8
        option_policies, option_termination = get_eigen_options(env_grid, eigen_reward, num_options, display=True)

    # ------------- 4. Solve tasks using abstraction -------------
    print("**Training Transfer Policy**")
    random_seed = 27835849
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    num_tasks = 30
    batch = replay_buffer.sample(num_tasks)
    _, tasks, _, _, _ = zip(*batch)
    env = FourRoomsNoReward({"max_steps": 200})
    all_successes = []
    for t in tasks:
        if exp_type == "dsaa":
            trained_policy, avg_success, task_successes = solve_dsaa_task(env, t, phi, option_policies, abstract_adjacency)
        elif exp_type == "contrastive":
            trained_policy, avg_success, task_successes = solve_contrastive_task(env, t, contrastive_encoder)
        elif exp_type == "eigenoptions":
            trained_policy, avg_success, task_successes = solve_task_with_options(env, t, option_policies, option_termination)
        print(f"Task {t}, Final Avg Success {avg_success:1.3f}")
        all_successes.append([t, task_successes])
    pickle.dump(all_successes, open(f"tmp_data/episode_success_{exp_type}.pickle", "wb"))

    return


if __name__ == "__main__":
    train(exp_type = "dsaa")
    # train(exp_type = "contrastive")
    # train(exp_type = "eigenoptions")
    pass