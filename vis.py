import matplotlib.pyplot as plt
import pickle, random, torch
import numpy as np

from utils import get_nbrs
from torch_models import Abstraction

# Given an adjacency matrix draw the corresponding graph
def draw_abstract_mdp(adjacency, save_path="", file_format="png"):
    import networkx as nx
    plt.clf()
    G = nx.MultiDiGraph()
    for s in range(len(adjacency)):
        nbrs = get_nbrs(adjacency, s)
        for s_prime in nbrs:
            e = (s, s_prime)
            G.add_edge(e[0], e[1])
            
    pos = nx.spring_layout(G)
    nx.draw(G, pos,
            with_labels=True,
            connectionstyle='arc3, rad = 0.1',
            node_color='lightgreen')
    
    plt.savefig("{}/abstract_mdp.{}".format(save_path, file_format), bbox_inches="tight", format=file_format)

# Given an abstraction for Arm2D environment, draw the 3D configuration space of the first 3 joints as a video of 2D frames
# TODO: currently only works for 3 joints
def arm2D_cspace_video(save_path, phi, num_abstract_states, yes_obj = False):
    import matplotlib.animation as animation
    frame_skip = 3
    # This method gets called every frame - feels like there must be a more efficient implementation
    def plot_stuff(frame):
        print(f"\r{frame}", end="")
        plt.clf()
        # TODO: should remove this hardcoding
        xvalues = np.arange(180) - 90
        yvalues = np.arange(180) - 90
        dim2, dim3 = np.meshgrid(xvalues, yvalues)
        dim1 = np.ones(180*180)*(frame*frame_skip-90)
        pts = np.concatenate((dim1.reshape(-1,1), dim2.reshape(-1,1), dim3.reshape(-1,1)), axis=1)

        if yes_obj:
            # obj starts at (13,13)
            # TODO: remove this hardcoding
            obj_pose = np.zeros((180*180, 2)) + 13.0 
            pts = np.concatenate((pts, obj_pose), axis=1)

        with torch.no_grad():
            abstract_state = phi(torch.FloatTensor(pts)).argmax(dim=1).view(180,180)
        
        plt.imshow(abstract_state.numpy(), vmin=0, vmax=num_abstract_states-1.0)
        plt.colorbar() # TODO: ideally we have one colorbar for the entire video, this one changes...
        return [] # TODO: I don't recall what the animation return value does

    ani = animation.FuncAnimation(plt.figure(), plot_stuff, frames=180//frame_skip, interval=100, blit=True)
    ani.save("{}/cspace.mp4".format(save_path))

# This method makes the 2 Room SuccessorRepresentation visualization from the paper
def make_sr_vis(save_path, option_type="uniform"):
    from environments.env_wrappers import TwoRoomsViz
    def wall_hugging_option(env):
        dir_to_vec = {0: [-1, 0], 1: [0,1], 2: [1,0], 3: [0,-1]}
        dir_to_wall = []
        for action in dir_to_vec:
            tmp_pos = env.agent_pos[:2] + dir_to_vec[action]
            dir_to_wall.append(not env.is_free(tmp_pos))
        
        action = random.randrange(4)
        if dir_to_wall[0]:
            action = 3
        if dir_to_wall[3]:
            action = 2
        if dir_to_wall[2]:
            action = 1
        if dir_to_wall[1] and not dir_to_wall[0]:
            action = 0
        
        if dir_to_wall[0] and dir_to_wall[2]:
            action = 1
            if random.random() < 0.5:
                action = 3
        return action

    plt.clf()
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,4), gridspec_kw = {'hspace':0.1})
    # fig.tight_layout()
    # fig.subplots_adjust(hspace=0.01)
    diffs = np.zeros((2, 3,10,19))
    for row_num, option_type in enumerate(["uniform", "hugwall"]):
        env = TwoRoomsViz()
        state = env.reset()
        i = np.argmax(state)
        psi = np.zeros((len(state), len(state)))
        state_pairs = []
        for cur_idx in range(500000):
            if option_type == "hugwall":
                action = wall_hugging_option(env.env)
            else:
                action = random.randrange(env.action_size)
                
            next_state,_,_,_ = env.step(action)
            next_i = np.argmax(next_state)
            state_pairs.append([state, i, next_i])
            
            i = next_i
            state = next_state
            if cur_idx % 5000 == 0:
                print(cur_idx, end="\r")
        print("Done exploration")

        for _ in range(5):
            perm = np.random.permutation(len(state_pairs))
            for cur_idx, p_i in enumerate(perm):
                state, i, next_i = state_pairs[p_i]
                psi[i] = psi[i] + 0.1*((state + 0.99*psi[next_i]) - psi[i])
                if cur_idx % 5000 == 0:
                    print(cur_idx, end="\r")
        print("Done Update")
            
        # fig, ax = plt.subplots(nrows=1, ncols=3)
        # diffs = np.zeros((3,10,19))
        ref_squares = [[4,4], [4,9], [4, 14]]
        for ax_idx, ref_square in enumerate(ref_squares):
            idx = 19*ref_square[0] + ref_square[1]
            print(idx)
            diffs[row_num, ax_idx] += np.abs(psi - psi[idx]).sum(axis=1).reshape(19,19)[:10,:19]
            walls = psi.sum(axis=1).reshape(19,19)[:10,:19] < 0.01
            diffs[row_num, ax_idx][walls.nonzero()] = -1.0

    refs = []
    for s in ref_squares:
        refs.append(plt.Circle(s, 0.5, color='blue'))

    for row_num in range(2):
        for ax_idx in range(3):
            im = ax[row_num, ax_idx].imshow(diffs[row_num, ax_idx], vmin=-1.0, vmax=np.max(diffs), cmap="hot")
            if True:#row_num == 0:
                ax[row_num, ax_idx].set_axis_off()
            else:
                ax[row_num, ax_idx].set_xticks(np.arange(0, 19, 2.0))
                ax[row_num, ax_idx].set_yticks(np.arange(0, 10, 2.0))
            # ax[row_num, ax_idx].set_adjustable('datalim')
            ax[row_num, ax_idx].set_aspect('auto')
            c = plt.Circle((ref_squares[ax_idx][1], ref_squares[ax_idx][0]), 0.5, color='blue')
            ax[row_num, ax_idx].add_patch(c)

    cax = fig.add_axes([ax[1, 2].get_position().x1+0.01,
                        ax[1, 2].get_position().y0,
                        0.02,
                        ax[1, 2].get_position().height * 2 + 0.05])
    cbar=plt.colorbar(im, cax=cax)

    cbar.set_ticks([0,int(np.max(diffs))])
    cbar.set_ticklabels([0,1])
    
    # plt.colorbar()
    # fig.suptitle(f"Relative SR distance under {option_type} policy")
    # plt.savefig(f"{save_path}/SR_{option_type}.png")
    # fig.subplots_adjust(hspace=0.1)
    plt.savefig(f"{save_path}/SR_vis_blue_ref_TMP.svg", format="svg")
    plt.close()

# for the two algorithms (baseline and dsaa) and for the two tasks (easy and hard) we plot the returns
def plot_returns_from_paper():
    plt.clf()
    def make_np(ar, longest):
        new_ar = np.ones((len(ar), longest))
        for i,j in enumerate(ar):
            new_ar[i][0:len(j)] = j
        return new_ar

    dsaa_easy = pickle.load(open("saved_data/dsaa_easy.pickle", "rb"))
    baseline_easy = pickle.load(open("saved_data/baseline_easy.pickle", "rb"))
    baseline_hard = pickle.load(open("saved_data/baseline_hard.pickle", "rb"))
    dsaa_hard = pickle.load(open("saved_data/dsaa_hard.pickle", "rb"))

    # make all the returns the same length (different lengths because some succeed sooner)
    longest = max([len(a) for a in dsaa_easy + baseline_easy + baseline_hard + dsaa_hard])
    d_easy = make_np(dsaa_easy, longest)
    d_hard = make_np(dsaa_hard, longest)
    b_easy = make_np(baseline_easy, longest)
    b_hard = make_np(baseline_hard, longest)

    # smoothing for nicer visualization
    gamma = 0.9
    for col in range(1, longest):
        d_easy[:,col] = d_easy[:,col-1]*gamma + d_easy[:,col]*(1-gamma)
        d_hard[:,col] = d_hard[:,col-1]*gamma + d_hard[:,col]*(1-gamma)
        
        b_easy[:,col] = b_easy[:,col-1]*gamma + b_easy[:,col]*(1-gamma)
        b_hard[:,col] = b_hard[:,col-1]*gamma + b_hard[:,col]*(1-gamma)

    # mean and standard deviation
    means_d_easy = np.mean(d_easy, axis=0)
    stds_d_easy = np.std(d_easy, axis=0)
    means_d_hard = np.mean(d_hard, axis=0)
    stds_d_hard = np.std(d_hard, axis=0)

    means_b_easy = np.mean(b_easy, axis=0)
    stds_b_easy = np.std(b_easy, axis=0)
    means_b_hard = np.mean(b_hard, axis=0)
    stds_b_hard = np.std(b_hard, axis=0)

    x = np.arange(longest)
    plt.plot(x, means_d_easy, label="dsaa_easy")
    plt.plot(x, means_d_hard, label="dsaa_hard")
    plt.plot(x, means_b_easy, label="base_easy")
    plt.plot(x, means_b_hard, label="base_hard")
    plt.xlabel("Number of Episodes", fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=14)
    plt.ylabel("Average Return", fontsize=13)
    plt.fill_between(x, (means_d_easy-stds_d_easy).clip(0,1), (means_d_easy+stds_d_easy).clip(0,1), color="blue", alpha=0.2)
    # plt.fill_between(x, (means_d_hard-stds_d_hard).clip(0,1), (means_d_hard+stds_d_hard).clip(0,1), color="orange", alpha=0.2)
    plt.fill_between(x, (means_b_easy-stds_b_easy).clip(0,1), (means_b_easy+stds_b_easy).clip(0,1), color="green", alpha=0.2)
    # plt.fill_between(x, (means_b_hard-stds_b_hard).clip(0,1), (means_b_hard+stds_b_hard).clip(0,1), color="red", alpha=0.2)
    plt.savefig("saved_data/paper_avg_returns_9_28.png", bbox_inches="tight")
    plt.savefig("saved_data/paper_avg_returns_9_28.svg", bbox_inches="tight", format="svg")
    

if __name__=="__main__":
    from argparse import ArgumentParser
    import json

    parser = ArgumentParser()
    parser.add_argument("--exp_path", dest="exp_path", default="experiments/exp_test")
    parser.add_argument('--make_cspace_vid', dest='make_cspace_vid', action='store_true', default=False)
    parser.add_argument('--draw_abstract_mdp', dest='draw_abstract_mdp', action='store_true', default=False)
    parser.add_argument('--make_sr_vis', dest='make_sr_vis', action='store_true', default=False)
    parser.add_argument('--paper_returns', dest='paper_returns', action='store_true', default=False)
    
    args = parser.parse_args()
    save_path = args.exp_path
   
    if args.draw_abstract_mdp:
        adjacency_matrix = pickle.load(open(f"{save_path}/abstract_adjacency.pickle", "rb"))
        draw_abstract_mdp(adjacency_matrix, save_path)
    
    if args.make_cspace_vid:
        with open("{}/config.json".format(args.exp_path)) as f_in:
            config = json.load(f_in)
            phi = Abstraction(obs_size=5, num_abstract_states=config["num_abstract_states"])
            phi.load_state_dict(torch.load(f"{save_path}/phi.torch"))
            arm2D_cspace_video(save_path, phi, num_abstract_states=config["num_abstract_states"], yes_obj = True)
    
    if args.make_sr_vis:
        make_sr_vis("saved_data", option_type=None)
        # make_sr_vis("saved_data", option_type="uniform")
        # make_sr_vis("saved_data", option_type="hugwall")
    
    if args.paper_returns:
        plot_returns_from_paper()