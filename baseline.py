import torch, random
import numpy as np
import torch.nn.functional as F

from environments.env_wrappers import Manipulator2D
from utils import ReplayBuffer
from torch_models import SoftQNetwork

def train_baseline_policy():
    # baseline configured for hard task
    config = {"max_steps":5000, "num_arm_joints": 3, "arm_joint_lengths": 10, "ball_goal_height": 11}

    env = Manipulator2D(config)
    input_size = env.observation_size
    num_actions = env.action_size
    
    # Params
    option_batch_size = 512
    learn_steps = 0
    target_update_steps = 20
    gamma = 0.95
    num_epochs = 1000
    learning_rate = 0.01
    softQ_entropy_coeff = 0.01
    
    online_policy = SoftQNetwork(inputs=input_size, 
                                            outputs=num_actions, 
                                            entropy_coef = softQ_entropy_coeff)

    target_policy = SoftQNetwork(inputs=input_size, 
                                            outputs=num_actions, 
                                            entropy_coef = softQ_entropy_coeff)
    target_policy.load_state_dict(online_policy.state_dict())
    option_optimizer = torch.optim.Adam(online_policy.parameters(), lr=learning_rate)

    replay_buffer = ReplayBuffer(1000000)

    # the first epoch is skipped...
    env_done = True
    running_reward = 0
    epoch_reward = 0
    episode_successes = []
    for epoch in range(num_epochs):
        
        while not env_done:        
            with torch.no_grad():
                # Get the primitive action from the option_policy for the current abstract_state and skill
                if random.random() < 0.7:
                    action = online_policy.choose_action(state)
                else:
                    action = random.randrange(num_actions)

            # Step in the environment
            next_state, env_reward, env_done, info = env.step(action)
            
            epoch_reward += env_reward
            
            ep_done = False # infinite horizon
            replay_buffer.add((state, next_state, action, env_reward*100.0, ep_done))

            state = next_state

        if epoch > 0:
            running_reward = running_reward*0.95 + (epoch_reward>0)*0.05
            episode_successes.append(epoch_reward>0)
            if len(episode_successes) > 10:
                if np.mean(episode_successes[-10:]) > 0.75:
                    return episode_successes
            print(f"Epoch {epoch}, Success: {epoch_reward>0}, Epoch reward {epoch_reward}, "+
                    f"Running reward {running_reward:4.3f}, final state {np.array(state)}")
            # arm_visit_entropy(replay_buffer)

        # Here env_done = True, reset everything
        env_done = False
        epoch_reward = 0
        state = env.reset()

        # Technically this code can come after every time we add to the buffer...
        if len(replay_buffer) > option_batch_size:
            learn_steps += 1
            if learn_steps % target_update_steps == 0:
                target_policy.load_state_dict(online_policy.state_dict())
            
            batch = replay_buffer.sample(option_batch_size)
            batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

            batch_state = torch.FloatTensor(batch_state)
            batch_next_state = torch.FloatTensor(batch_next_state)
            batch_action = torch.FloatTensor(batch_action).unsqueeze(1)
            batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1)
            batch_done = torch.FloatTensor(batch_done).unsqueeze(1)

            with torch.no_grad():
                next_q = target_policy(batch_next_state)
                next_v = target_policy.getV(next_q)
                y = batch_reward + (1 - batch_done) * gamma * next_v

            loss = F.mse_loss(online_policy(batch_state).gather(1, batch_action.long()), y)
            
            option_optimizer.zero_grad()
            loss.backward()
            option_optimizer.step()
    # The end
    return episode_successes

if __name__=="__main__":
    train_baseline_policy()