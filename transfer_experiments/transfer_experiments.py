'''
**Figures 5,6,10 and Table 1 from paper**

Compare DSAA to Eigenoption, Contrastive, and a Baseline in FourRooms *transfer learning task*
'''

from cmath import exp
import random
import numpy as np
import torch
import pickle

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

# relative python imports... need to fix this properly
from transfer_experiments.transfer_utils import FourRoomsNoReward, NormalizedTransitionsDataset, get_eigen_options, \
    get_eigen_reward, random_explore, solve_task_with_options, train_dsaa, train_dsaa_options, solve_dsaa_task, \
        train_contrastive_encoder, solve_contrastive_task, get_dsaa_indiv_options, \
            get_successor_options, get_successor_options_reward, montezuma_test

from utils import ReplayBuffer, get_nbrs

from environments.env_wrappers import MontezumaNoReward

def transfer_exp(exp_type = "eigenoptions"):
    # ------------- 1. Explore Environment -------------
    env_config = {
        "max_steps": 200000 # no reset
    }
    # env = FourRoomsNoReward(env_config)
    env = MontezumaNoReward(env_config)
    img_data = []

    print("**Exploring Environment**")
    data = []
    replay_buffer = ReplayBuffer(env_config["max_steps"])
    state = env.reset()
    for _ in range(env_config["max_steps"]):
        action = env.action_space.sample()
        next_state , _, done, _ = env.step(action)
        # print(state)
        data.append([state, next_state])
        img_data.append(env.get_image())

        replay_buffer.add((state, next_state, action, 0, 0))
        
        if done:
            state = env.reset()
        else:
            state = next_state
    print("num samples:", len(replay_buffer))
    # ------------- 2. Train abstraction -------------

    montezuma_test(data, img_data)
    exit()

    print("**Training Abstraction**")
    if exp_type == "dsaa":
        print("DSAA")
        num_abstract_states = 16
        phi = train_dsaa(replay_buffer, 
                        config={"num_abstract_states": num_abstract_states, 
                                "obs_size": env.observation_size,
                                "num_abstraction_updates": 40000})
        torch.save(phi.state_dict(), "rebuttal_imgs/phi_5.torch")
        if False:
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
            # plt.colorbar()
            plt.savefig(f"rebuttal_imgs/dsaa_abstraction_{num_abstract_states}.png", bbox_inches='tight')
            plt.savefig(f"rebuttal_imgs/dsaa_abstraction_{num_abstract_states}.svg", bbox_inches='tight', format="svg")

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
                plt.savefig("tmp_data/contrastive_2d_encoding.png", bbox_inches='tight')
                plt.savefig("tmp_data/contrastive_2d_encoding.svg", bbox_inches='tight', format="svg")
    
    elif exp_type == "eigenoptions":
        print("EIGENOPTIONS")
        env_grid = env.example_obs[0]
        eigen_reward = get_eigen_reward(env_grid)

    elif exp_type == "SRoptions":
        print("SUCCESSOR OPTIONS")
        num_clusters = 16
        successor_options_reward = get_successor_options_reward(data, num_clusters)

    return
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

    elif exp_type == "SRoptions":
        env_grid = env.example_obs[0]
        option_policies, option_termination = get_successor_options(env_grid, successor_options_reward, num_clusters, display=True)

    # ------------- 4. Solve tasks using abstraction -------------
    print("**Training Transfer Policy**")
    # TODO: for some reason this isn't working properly and the runs are different
    #           - doesn't matter too much since the results are quite consistent
    random_seed = 27835849 # mashed keyboard
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    num_tasks = 30
    batch = replay_buffer.sample(num_tasks)
    _, tasks, _, _, _ = zip(*batch)
    env = FourRoomsNoReward({"max_steps": 200})
    all_successes = []
    for t in tasks:
        if exp_type == "random":
            all_successes.append([t, random_explore(env, t)])
            continue

        if exp_type == "dsaa":
            trained_policy, avg_success, task_successes = solve_dsaa_task(env, t, phi, option_policies, abstract_adjacency)
        elif exp_type == "contrastive":
            trained_policy, avg_success, task_successes = solve_contrastive_task(env, t, contrastive_encoder)
        elif exp_type == "eigenoptions" or exp_type == "SRoptions":
            trained_policy, avg_success, task_successes = solve_task_with_options(env, t, option_policies, option_termination)
        
        print(f"Task {t}, Final Avg Success {avg_success:1.3f}")
        all_successes.append([t, task_successes])
    pickle.dump(all_successes, open(f"rebuttal_imgs/episode_success_{exp_type}_paperreward_11_9.pickle", "wb"))

    return


if __name__ == "__main__":
    # transfer_exp(exp_type = "random")
    # transfer_exp(exp_type = "dsaa")
    # transfer_exp(exp_type = "contrastive")
    transfer_exp(exp_type = "eigenoptions")