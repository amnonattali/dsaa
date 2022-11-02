'''
**Figure 7 from paper**
'''

import random, pickle, torch
import numpy as np

from torch_models import Abstraction, SuccessorRepresentation, MonolithicPolicy
from update_models import train_option_policies, update_abstraction
from utils import ReplayBuffer, delete_edge, get_nbrs, get_new_skill, delete_edge
from environments.env_wrappers import BaseFourRooms, FourRoomsRandomNoise, Manipulator2D, Manipulator2DNoOBJ

# makes printing to stdout more readable
torch.set_printoptions(precision=3, linewidth=120, sci_mode=False)
np.set_printoptions(precision=3, suppress=True)

env_types = {
    "BaseFourRooms": BaseFourRooms,
    "FourRoomsRandomNoise": FourRoomsRandomNoise,
    "YesOBJManipulator": Manipulator2D,
    "NoOBJManipulator": Manipulator2DNoOBJ
}

def dsaa_experiments(config):
    # ------------- INIT -------------
    env = env_types[config["env_type"]](config)
    save_path = config["save_path"]

    obs_size = env.observation_size
    num_actions = env.action_size
    num_abstract_states = config["num_abstract_states"]
    num_skills = num_abstract_states # one skill per possible next abstract state
    learning_rate = config["learning_rate"]
    
    # initialize abstraction model
    phi = Abstraction(obs_size, num_abstract_states)
    phi_optimizer = torch.optim.Adam(phi.parameters(), lr=learning_rate)
    # initialize successor representation
    psi = SuccessorRepresentation(num_abstract_states)
    psi_optimizer = torch.optim.Adam(psi.parameters(), lr=learning_rate)

    # initialize option policies (online and target models)
    online_Q = MonolithicPolicy(num_abstract_states, num_skills, obs_size, num_actions, config)
    target_Q = MonolithicPolicy(num_abstract_states, num_skills, obs_size, num_actions, config)
    option_optimizer = online_Q.option_optimizer    
    target_Q.network.load_state_dict(online_Q.network.state_dict())
    
    # initialize replay buffer
    replay_buffer_size = config["option_replay_buffer_size"]
    replay_buffer = ReplayBuffer(replay_buffer_size)

    first_iteration = True # in the first iteration we get pure random exploration
    if config["load_saved_abstraction"]:
        phi.load_state_dict(torch.load(f"{save_path}/phi.torch"))
        psi.load_state_dict(torch.load(f"{save_path}/psi.torch"))
        
        online_Q.network.load_state_dict(torch.load(f"{save_path}/mono_Q.torch"))
        abstract_adjacency = pickle.load(open(f"{save_path}/abstract_adjacency.pickle", "rb"))
        
        print("Loaded saved abstraction and abstract graph:\n", abstract_adjacency)
        # if we load a saved model it is not the first iteration
        first_iteration = False
    else:
        # if we don't load a saved model we initialize the adjacency to the identity
        abstract_adjacency = torch.eye(num_abstract_states)
    
    # option training params
    num_exploration_iters = config["option_replay_buffer_size"] # we just fill up the buffer once
    # initialize env_done to True so we trigger a reset in the first iteration
    env_done = True
    option_done = True
    all_episode_success = [] # keep track of episode successes (environment task)
    for iteration in range(config["max_iter"]):
        # ------------- OPTION EXPLORATION PHASE -------------
        
        # visit counts represents the number of times each abstract state is transitioned into
        # this is useful for biasing exploration towards low visit states
        #   i.e., "intrinsic motivation" in the abstract state space
        visit_counts = torch.zeros((num_abstract_states))
        old_visit_counts = np.copy(visit_counts) # used to keep track of new visits each iteration

        # rewarding_options keeps track of which options, when executed, provide the most reward 
        # NOTE: this is a heuristic replacement for training a proper abstract policy
        rewarding_options = torch.zeros((num_abstract_states, num_abstract_states))
        
        # option_success is a running exponential average of the success of pi_(s,z)
        option_success = torch.zeros((num_abstract_states, num_skills))
        avg_success = 0 # avg success keeps track of recent success rate of options which were called
        
        # To avoid wasting time on option policies that keep fail we sometimes delete edges
        bad_edges = set([]) 
        num_times_stuck = np.zeros((num_abstract_states, num_skills))

        # some more statistics
        running_episode_reward = 0
        episode_reward = 0
        print(f"**Beginning Exploration Phase for {num_exploration_iters} Steps**")
        for option_iter in range(num_exploration_iters):
            # if the episode is over reset
            if env_done:
                running_episode_reward = running_episode_reward*0.95 + (1.0 - 0.95)*(episode_reward>0)
                all_episode_success.append(episode_reward > 0)
                # NOTE: in the unsupervised case the loop runs for max_iter iterations
                if len(all_episode_success) > 10 and np.mean(all_episode_success[-10:]) > config["success_threshold"]:
                    return all_episode_success
                        
                env_done = False
                option_done = False
                steps_in_current_state = 0  # this is how long we've been running the option for
                episode_reward = 0          # this is the episode return
                current_option_reward = 0   # this is how much reward was accumulated during the current option
                
                state = env.reset()
                # we compute the current abstract state
                with torch.no_grad():
                    # TODO: fix ugly code
                    abstract_state = phi(torch.FloatTensor(state).view(1,-1))
                    a_num = phi.to_num(abstract_state)[0].item()
                    abstract_state = abstract_state[0]
                # In principle this new skill is just a random neighbor in the abstract graph
                #   when we find an option that solves the environment task we plan a shortest path to it
                skill = get_new_skill(first_iteration, a_num, abstract_adjacency, 
                                        visit_counts, rewarding_options, episode_reward, config["use_env_reward"])

                # augment the primitive state with the abstract state
                state += abstract_state.tolist()
                # the starting abstract state counts as being transitioned into
                visit_counts[a_num] += 1

            # if the option has terminated (but the episode has not) we need a new skill
            elif option_done:
                option_done = False
                steps_in_current_state = 0
                current_option_reward = 0
                skill = get_new_skill(first_iteration, a_num, abstract_adjacency, 
                                        visit_counts, rewarding_options, episode_reward, config["use_env_reward"])

            steps_in_current_state += 1
            if first_iteration:
                action = random.randrange(num_actions)
            else:
                action = online_Q.choose_action(state, skill)
                
            # step in environment... env takes care of structuring state correctly
            next_state, env_reward, env_done, _ = env.step(action)
            
            # get the next abstract state
            with torch.no_grad():
                next_abstract_state = phi(torch.FloatTensor(next_state).view(1,-1))
                next_a_num = phi.to_num(next_abstract_state)[0].item()
                next_abstract_state = next_abstract_state[0]
            
            next_state += next_abstract_state.tolist() # again, we augment the state with the abstract state     
            
            # keep track of environment reward
            episode_reward += env_reward
            current_option_reward += env_reward
            rewarding_options[a_num, skill] += env_reward

            # option ends when an abstract transition occurs
            option_done = (a_num != next_a_num)
            # get reward if transition to the correct abstract state
            option_reward = 0.0
            if option_done:
                visit_counts[next_a_num] += 1
                # ---- reward ----
                # NOTE: in the softQlearning setting where we reward the agent with entropy, 
                #   we need to make sure that the agent gets more reward for transitioning than for high entropy
                option_reward += config["option_success_reward"] * (float(next_a_num == skill))
                
                # ---- adding edges ----
                # whenever a transition occurs we mark it
                # if the transition hadn't been seen before add edge to graph (meaning add skill)
                if not next_a_num in get_nbrs(abstract_adjacency, a_num) and not (a_num, next_a_num) in bad_edges:
                    abstract_adjacency[a_num, next_a_num] = 1.0 # TODO: eventually we may want to actually use this weight
                    print(f"Adding edge {a_num} --> {next_a_num}")
                    
                # ---- track option success statistics ----
                if a_num != skill:
                    # if the option received some reward we consider this option successful
                    cur_suc = ((option_reward > 0) or (config["use_env_reward"] and current_option_reward > 0))
                    option_success[a_num, skill] = option_success[a_num, skill]*0.95 + 0.05*cur_suc
                    avg_success = avg_success*0.99 + 0.01*cur_suc
                else: # self loop edge is not supposed to transition
                    option_success[a_num, a_num] = option_success[a_num, a_num] * 0.99
                
                
            # add experience to replay buffers
            # NOTE because during the update phi-based reward is recomputed, we only store the environment reward
            #   this is clearly inefficient to recompute the abstract states/reward, but prepares us for future work 
            #       in which the abstraction update and option updates are more interleaved 
            if config["use_env_reward"]: 
                option_env_reward = 2 * config["option_success_reward"] * env_reward
            else:
                option_env_reward = 0
            replay_buffer.add((state[:obs_size], next_state[:obs_size], action, option_env_reward, option_done))
            
            # we mark option as done if it hasn't transitioned for too long
            if not first_iteration and steps_in_current_state > config["max_steps_in_state"]:
                visit_counts[a_num] += 1 # it counts as having visited here again (since we were here a while)
                option_done = True
                # if we didn't transition it counts as a failure (assuming not self loop)
                if a_num != skill:
                    # in the supervised case it counts as success if we got environment reward
                    if not (config["use_env_reward"] and current_option_reward > 0):
                        num_times_stuck[a_num, skill] += 1
                else: # self loop edge should get here
                    option_success[a_num, a_num] = option_success[a_num, a_num]*0.99 + 0.01

            # prepare next iteration
            state = next_state
            abstract_state = next_abstract_state
            a_num = next_a_num

            # Now update the option policy for this abstract state
            if not first_iteration and \
                        option_iter % config["option_update_freq"] == 0 and \
                        len(replay_buffer) > config["option_batch_size"]:
                train_option_policies(online_Q, target_Q, option_optimizer, 
                    phi, replay_buffer, config, num_updates=1)
                
            # log results
            if option_iter % config["log_freq"] == config["log_freq"] - 1:
                new_visits = visit_counts - old_visit_counts
                old_visit_counts = np.copy(visit_counts)
                
                if config["delete_bad_edges"]:
                    delete_edge(option_success, abstract_adjacency, num_times_stuck, visit_counts, bad_edges)
                
                all_avg_suc = option_success[abstract_adjacency.nonzero()[:,0],abstract_adjacency.nonzero()[:,1]].mean().item()
                print(f"Stats (iter {iteration}, option_iter: {option_iter}):" + 
                    f"\n\tVisit counts          {visit_counts}" +
                    f"\n\tNew visits:           {new_visits.sum()}" +
                    f"\n\tAvg success (recent): {avg_success:1.3f}" +
                    f"\n\tAvg Success (all):    {all_avg_suc:1.3f}" +
                    f"\n\tSelf loop success:    {torch.diag(option_success).mean().item():1.3f}" +
                    f"\n\tRunning reward:       {running_episode_reward:4.3f}" +
                    f"\n\tBad edges:            {bad_edges}"
                )
        print(f"Exploration phase finished, total visits {visit_counts.sum()}")
        
        if not first_iteration:
            torch.save(online_Q.network.state_dict(), f"{save_path}/mono_Q.torch")
            torch.save(phi.state_dict(), f"{save_path}/prev_phi.torch")
            torch.save(psi.state_dict(), f"{save_path}/prev_psi.torch")
            pickle.dump(abstract_adjacency, open(f"{save_path}/abstract_adjacency.pickle", "wb"))
            pickle.dump(option_success, open(f"{save_path}/option_success.pickle", "wb"))

        # ------------- ABSTRACTION UPDATE PHASE -------------
        update_abstraction(phi, phi_optimizer, psi, psi_optimizer, replay_buffer, config)
        
        # optimizations ---------------------------------------------------------------------------------------------
        print("Finished abstraction update, now initializing option policies with new abstraction and old data.")
        # ------------- OPTION INITIALIZATION WITH NEW ABSTRACTION -------------
        train_option_policies(online_Q, target_Q, option_optimizer, 
                    phi, replay_buffer, config, num_updates=config["post_abstraction_option_updates"])
        torch.save(online_Q.network.state_dict(), f"{save_path}/mono_Q_post_abstraction.torch")

        # ------------- INITIALIZE ABSTRACT GRAPH WITH KNOWN TRANSITIONS -------------
        # TODO: this is inefficient, should do this during the previous step
        batch = replay_buffer.sample(len(replay_buffer)) # NOTE: needs to fit in memory...
        batch_state, batch_next_state, _, _, _ = zip(*batch)
        batch_state = torch.FloatTensor(batch_state)
        batch_next_state = torch.FloatTensor(batch_next_state)
        with torch.no_grad():
            abstract_state = phi(batch_state)
            next_abstract_state = phi(batch_next_state)
            
            abstract_state_nums = phi.to_num(abstract_state).flatten()
            next_abstract_state_nums = phi.to_num(next_abstract_state).flatten()
        abstract_adjacency = torch.eye(num_abstract_states)
        abstract_adjacency[abstract_state_nums, next_abstract_state_nums] = 1
        print("Initialized adjacency matrix:\n", abstract_adjacency)
        pickle.dump(abstract_adjacency, open(f"{save_path}/abstract_adjacency_post_abstraction.pickle", "wb"))

        first_iteration = False # first iteration is over
        env_done = True # we want to reset at the beginning of each exploration phase

def run_10_exps(config):
    all_episode_successes = []
    for i in range(0,10):
        torch.manual_seed(i)
        random.seed(i)
        np.random.seed(i)

        print("*************Current run (seed):", i)
        episode_success = dsaa_experiments(config)
        # pickle.dump(episode_success, open(f"{config['save_path']}/episode_success_seed_{i}.pickle", "wb"))
        all_episode_successes.append(episode_success)

    pickle.dump(all_episode_successes, open(f"{config['save_path']}/episode_successes.pickle", "wb"))

if __name__ == "__main__":
    from argparse import ArgumentParser
    import json

    parser = ArgumentParser()
    parser.add_argument("--exp_path", dest="exp_path", default="experiments/exp_test")
    parser.add_argument('--load_save', dest='load_save', action='store_true', default=False)
    
    args = parser.parse_args()

    with open("{}/config.json".format(args.exp_path)) as f_in:
        config = json.load(f_in)
        config["save_path"] = args.exp_path
        config["load_saved_abstraction"] = args.load_save
        
        all_episode_successes = []
        for i in range(0,10):
            torch.manual_seed(i)
            random.seed(i)
            np.random.seed(i)

            print("*************Current run (seed):", i)
            episode_success = dsaa_experiments(config)
            # pickle.dump(episode_success, open(f"{config['save_path']}/episode_success_seed_{i}.pickle", "wb"))
            all_episode_successes.append(episode_success)

        pickle.dump(all_episode_successes, open(f"{config['save_path']}/episode_successes.pickle", "wb"))