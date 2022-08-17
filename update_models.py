import torch
import torch.nn.functional as F

# Using a replay_buffer of transitions in the environment
# we train option policies based on the intrinsic reward computed from an abstraction phi
# we reward each option for transitioning into a specific abstract state
def train_option_policies(online_Q, target_Q, option_optimizer, 
                            phi, replay_buffer, config, num_updates=None):
    running_loss = 0
    for update_idx in range(num_updates):
        batch = replay_buffer.sample(config["option_batch_size"])
        # When we sample from the buffer we ignore the environment "done", and compute our own
        batch_state, batch_next_state, batch_action, batch_env_reward, _ = zip(*batch)
        batch_state = torch.FloatTensor(batch_state)
        batch_next_state = torch.FloatTensor(batch_next_state)
        batch_action = torch.FloatTensor(batch_action).unsqueeze(1)
        batch_env_reward = torch.FloatTensor(batch_env_reward).unsqueeze(1)
        
        with torch.no_grad():
            # NOTE: when we train option policies we use Softmax not Gumbel so the abstraction is consistent
            abstract_state = phi(batch_state)
            next_abstract_state = phi(batch_next_state)
            
            # We append the full abstract state encoding, not just the index
            #   this gives the model more info about abstract state boundaries
            batch_state = torch.cat((batch_state, abstract_state), dim=1)
            batch_next_state = torch.cat((batch_next_state, next_abstract_state), dim=1)
            
            # We need the abstract state indices in order to check where abstract state transitions occur
            abstract_state_nums = phi.to_num(abstract_state)
            next_abstract_state_nums = phi.to_num(next_abstract_state)
            
            # Our option policy terminates when an abstract state transition occurs
            batch_done = 1.0*(next_abstract_state_nums != abstract_state_nums).view(-1,1)
            
        # some stats
        loss = 0
        total_r = 0
        q_mean = 0
        # We loop over each possible next_abstract_state goal - this determines the intrinsic reward
        for skill in range(config["num_abstract_states"]):
            # In Double DQN the target model lags behind the online one for stability
            online_Q.learn_steps += 1
            if online_Q.learn_steps % config["ddqn_target_update_steps"] == 0:
                target_Q.network.load_state_dict(online_Q.network.state_dict())
            
            # We reward our option policy when there is a transition into the correct next abstract state
            #   note that we don't reward the agent if it is already in the goal
            reward = config["option_success_reward"] * \
                ((abstract_state_nums != skill) * (next_abstract_state_nums == skill)).view(-1,1)
            
            # We disensentivize self-loop edges from transitioning outside the current state
            # TODO: not clear this is necessary (since soft-Q learning implies avoid ending the episode in the absence of reward)
            reward = reward - 5*((abstract_state_nums == skill) * (next_abstract_state_nums != abstract_state_nums)).view(-1,1)
            # (1/(1-config["option_gamma"]))*
            # self_mask = (abstract_state_nums != skill).view(-1,1)

            # In the supervised setting we provide the policy with the environment reward for the transition
            if config["reward_self"]: # only reward the self loop option
                batch_env_reward = (batch_env_reward * (abstract_state_nums == skill)).view(-1,1)
            else: # reward all options
                batch_env_reward = batch_env_reward.view(-1,1)
            
            # Now that we have computed the reward we are finally ready for the update
            current_q = online_Q(batch_state, skill).gather(1, batch_action.long())
            with torch.no_grad():
                next_q = target_Q(batch_next_state, skill)
                if config["soft_Q_update"]:
                    # In soft Q learning we sample actions according to the softmax of the Q values (probs*next_q)
                    # We add some reward based on the entropy (-probs*log(probs))
                    probs = torch.softmax(next_q, dim=1) + 0.1**16
                    next_v = (probs * (next_q - config["option_entropy_coef"]*torch.clip(torch.log(probs), min=-5, max=0))
                            ).sum(dim=1, keepdim=True)
                else:
                    next_v = torch.max(next_q, dim=1, keepdim=True)[0]
                # The Temporal-Difference target is r + gamma * next_v * (1-done)
                q_target = (1.0*reward + batch_env_reward + config["option_gamma"] * next_v * (1 - batch_done))

            # loss += F.smooth_l1_loss(current_q.masked_select(self_mask), q_target.masked_select(self_mask))
            loss += F.smooth_l1_loss(current_q, q_target)
            total_r += reward.sum().item()
            q_mean += q_target.sum().item() / config["option_batch_size"]
        
        q_mean /= config["num_abstract_states"]    
        loss /= config["num_abstract_states"]
        option_optimizer.zero_grad()
        
        loss.backward()
        option_optimizer.step()
        running_loss = running_loss * 0.99 + 0.01 * loss.item()

        # print debug...
        # NOTE: (Mean Q, Total R, and unique states) are all from the current iteration. 
        # running_loss is an exponential average over all iterations
        if update_idx % 100 == 99:
            print(f"Loss {running_loss:2.4f}, Mean Q: {q_mean:4.3f}, " +
                f"Total R: {total_r}, unique states {len(set(abstract_state_nums.view(-1,).numpy()))}")

# We update phi and psi simultaneously using a combination of two losses:
#   - maximize the entropy of the abstract states across each batch:
#       L_H = 1/B [sum_x [- phi(x) * log(phi(x))]]
#   - minimize the SR temporal difference error: 
#       L_SR = 1/B [sum_{x,x'} [phi(x) + gamma * psi(phi(x')) - psi(phi(x))]^2] 
def update_abstraction(phi, phi_optimizer, psi, psi_optimizer, replay_buffer, config):
    total_sr_loss = 0
    total_entropy_loss = 0
    for abstraction_iter in range(config["num_abstraction_updates"]):
        if len(replay_buffer) > config["abstraction_batch_size"]:
            phi_optimizer.zero_grad()
            psi_optimizer.zero_grad()

            batch = replay_buffer.sample(config["abstraction_batch_size"])
            batch_state, batch_next_state, _, _, _ = zip(*batch)

            batch_state = torch.FloatTensor(batch_state)
            batch_next_state = torch.FloatTensor(batch_next_state)
            
            if config["use_gumbel"]:
                # compute abstract states
                abstract_state = phi.sample(batch_state, tau=config["gumbel_tau"], hard=config["hard"])
                with torch.no_grad():
                    next_abstract_state = phi.sample(batch_next_state, tau=config["gumbel_tau"], hard=config["hard"])
            else:
                abstract_state = phi(batch_state)
                next_abstract_state = phi(batch_next_state)
            
            # compute the entropy of the distribution of abstract states
            mean_abstract_state_probs = abstract_state.mean(dim=0)
            avg_entropy = (- mean_abstract_state_probs * torch.log(mean_abstract_state_probs)).sum()
            # indiv_entropy = (- abstract_state * torch.log(abstract_state)).sum(dim=-1).mean()
            # avg_entropy = avg_entropy - indiv_entropy

            # compute the successor representation
            successor_representation = psi(abstract_state)
            
            with torch.no_grad():
                next_successor_representation = psi(next_abstract_state)
            
            # state_diff = torch.sum((next_abstract_state - abstract_state.detach())**2, dim=-1, keepdim=True)
            sr_td_loss = ((successor_representation -
                (abstract_state.detach() + config["sr_gamma"] * next_successor_representation))**2).sum(dim=1).mean()
            
            # NOTE: we can use a one hot representation of the abstract state... doesn't seem to improve things
            # abstract_state_nums = torch.argmax(abstract_state, dim=-1, keepdim=True)
            # abstract_state_nums = torch.argmax(phi(batch_state), dim=-1, keepdim=True)
            # one_hot_abstract_state = torch.zeros((config["abstraction_batch_size"], config["num_abstract_states"])).scatter_(1, abstract_state_nums, 1.)
            # sr_td_loss = ((successor_representation -
            #     (one_hot_abstract_state + config["sr_gamma"] * next_successor_representation))**2).sum(dim=1).mean()
            
            # NOTE: we can apply the loss only to transitions... doesn't seem to improve things
            # abstract_state_nums = torch.argmax(abstract_state, dim=1).detach() #phi.to_num(abstract_state).detach()
            # next_abstract_state_nums = torch.argmax(next_abstract_state, dim=1).detach() #phi.to_num(next_abstract_state).detach()
            # sr_td_loss = torch.masked_select(sr_td_loss, abstract_state_nums != next_abstract_state_nums)#.mean()
            
            # NOTE: we can add some more losses... doesn't seem to improve things
            #   such as a contrastive one which encourages transitions to occur where SR changes maximally
            # contrastive_loss = (successor_representation*next_successor_representation).sum(dim=1)
            # contrastive_loss = torch.masked_select(contrastive_loss, abstract_state_nums != next_abstract_state_nums).mean()

            # sr_td_loss = sr_td_loss.sum() / len(sr_td_loss)
            # print(sr_td_loss.item())
            ent_loss = - config["abstraction_entropy_coef"] * avg_entropy
            abstraction_loss = sr_td_loss + ent_loss

            abstraction_loss.backward()
            phi_optimizer.step()
            psi_optimizer.step()

            total_sr_loss += sr_td_loss.item()
            total_entropy_loss += ent_loss.item()

            if abstraction_iter % 1000 == 990:
                with torch.no_grad():
                    print(torch.argmax(phi(batch_next_state), dim=-1))
                #     # print(torch.argmax(abstract_state, dim=-1))
                print("Mean abstract state:", mean_abstract_state_probs.detach())
                print(f"Abstraction iters {abstraction_iter}, "+
                    f"sr_loss {total_sr_loss / abstraction_iter:3.3f}, "+
                    f"entropy_loss {total_entropy_loss / abstraction_iter:3.3f}")
                # torch.save(phi.state_dict(), "{}/phi.torch".format(config["save_path"]))
                # torch.save(psi.state_dict(), "{}/psi.torch".format(config["save_path"]))