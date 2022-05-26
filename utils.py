import torch, heapq, random
from collections import deque
import numpy as np

# Simple replay buffer using a first in first out queue
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def reset(self):
        self.buffer.clear()

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Get the list of neighbors of a_num from an adjacency matrix
def get_nbrs(adj, a_num):
    return adj[a_num].nonzero().flatten().numpy()

# In principle we sample a skill as a random neighbor in the adjacency graph
# In the supervised case
#   once we've found an option that succeeds at the environment task, we simply plan a path to it
#   after which we again sample random 
def get_new_skill(first_iteration, a_num, abstract_adjacency, visit_counts, rewarding_options, episode_reward, use_env_reward):
    if first_iteration:
        skill = -1 # in the first iteration we will just take random actions (might not be necessary, but simpler)
    else:
        if use_env_reward and rewarding_options.sum().item() > 0 and episode_reward <= 0.01:
            max_reward_pair = (rewarding_options==torch.max(rewarding_options)).nonzero()[0]
            if a_num == max_reward_pair[0]:
                skill = max_reward_pair[1]
            else:
                max_reward_path = plan_abstract_path(a_num, max_reward_pair[0], abstract_adjacency)
                skill = max_reward_path[0][1]
        else:
            random_decision = random.random()
            if random_decision < 0.25:
                # We find that sometimes choosing the neighbor with fewest visits is helpful for diversifying the data
                candidates = get_nbrs(abstract_adjacency, a_num).tolist()
                skill = candidates[torch.argmin(visit_counts[candidates]).item()]
            elif random_decision < 0.5:
                skill = a_num # lazy random walk
            else: # random neighbor from the adjacency graph
                skill = np.random.choice(get_nbrs(abstract_adjacency, a_num))
    return skill

# Simple implementation of shortest path algorithm on the abstract graph
# We can use this to bias the path using weights on the edges
def plan_abstract_path(start, abstract_state_goal, adjacency):
    def backtrack(visited_states, current_state):
        path = []
        skill = None
        while current_state is not None:
            path.append((current_state, skill))
            skill = visited_states[current_state][2]
            current_state   = visited_states[current_state][0]
        path.reverse()
        if len(path) == 1:
            path[0] = (path[0][0], path[0][0]) # the skill is self
        return path

    visited_states = {start: (None, 0, None)}
    frontier = []
    heapq.heappush(frontier, (0, start))
    while frontier:
        dist, current_state = heapq.heappop(frontier)
        if current_state == abstract_state_goal:
            return backtrack(visited_states, current_state)
        
        for n in get_nbrs(adjacency, current_state):
            skill = int(n) # skill is just next abstract state
            edge_weight = adjacency[current_state, skill]
            
            if skill == current_state or edge_weight < 0.01:
                continue
            
            if skill not in visited_states or dist + edge_weight < visited_states[skill][1]:
                new_dist = (dist + edge_weight)
                heapq.heappush(frontier, (new_dist, skill))
                visited_states[skill] = (current_state, new_dist, skill)
    
    rand_act = start
    return [(start, rand_act), (rand_act, None)]

# If some particular option, say a-->b, often gets stuck
#   we delete it (delete the edge from the adjacency matrix)
#   assuming there is some other path from a to b that is not too long
def delete_edge(option_success, abstract_adjacency, num_times_stuck, visit_counts, bad_edges):
    # sort the options in order of success rate
    rel_suc = np.array([[option_success[tmp_s, tmp_z].item(), tmp_s, tmp_z] for tmp_s, tmp_z in abstract_adjacency.nonzero().numpy() if tmp_s != tmp_z])
    # we want to avoid deleting things that were just added... use num_stuck
    if len(rel_suc) == 0:
        rel_suc = np.zeros((1,3))
    sorted_suc = np.argsort(rel_suc, axis=0)[:,0]
    for min_suc in sorted_suc:
        edge_start = int(rel_suc[min_suc, 1])
        edge_end = int(rel_suc[min_suc, 2])
        # don't delete an edge if we haven't tried it enough, it is very successful, or the target is rarely visited
        if num_times_stuck[edge_start, edge_end] < 10 or rel_suc[min_suc, 0] > 0.5 or visit_counts[edge_end] < 50:
            continue # TODO: some magic numbers here...

        tmp_adj = torch.clone(abstract_adjacency)
        tmp_adj[edge_start, edge_end] = 0
        
        can_remove_path = plan_abstract_path(edge_start, edge_end, tmp_adj)
        can_remove = (can_remove_path[-1][0] == edge_end) and len(can_remove_path) < 5
        if can_remove:
            print(f"Removed edge {edge_start}-->{edge_end}, success was {rel_suc[min_suc, 0]:0.3f}, num_stuck is {num_times_stuck[edge_start, edge_end]}")
            print(f"Alternative path is {[tmpp[0] for tmpp in can_remove_path]}")
            # NOTE: both pytorch arrays and python sets are passed by reference so this will affect inputs
            abstract_adjacency[edge_start, edge_end] = 0
            bad_edges.add((edge_start, edge_end))
            # only delete one edge
            return (edge_start, edge_end)
        else:
            print(f"Cannot remove {edge_start}-->{edge_end}")
    return None