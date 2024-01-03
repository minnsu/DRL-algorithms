import numpy as np
import torch
from copy import deepcopy
from torch.distributions import Normal, Categorical

class TDActorCritic():
    def __init__(self, env, model, critic_loss_fn, optimizer, discount_factor, batch_size=32, action_space_type='discrete'):
        self.env = env

        self.model = model
        self.critic_loss_fn = critic_loss_fn
        self.optimizer = optimizer
        
        self.discount_factor = discount_factor

        self.batch_size = batch_size

        if action_space_type == 'discrete':
            self.action_type = 0
        elif action_space_type == 'continuous':
            self.action_type = 1

    def action(self, state):
        state = torch.FloatTensor(state)

        # with torch.no_grad():
        if self.action_type == 0:
            # output is softmax(probability of each actions)
            output, _ = self.model(state)
            distrib = Categorical(probs=output)
        elif self.action_type == 1:
            # output is means and stddevs
            (means, stddevs), _ = self.model(state)
            distrib = Normal(means, stddevs)

        action = distrib.sample()
        return action
    
    def train(self, n_episodes, max_steps):
        step_sums = 0
        reward_sums = 0
        prev_i = 0
        
        batches = []
        states = []
        actions = []
        rewards = []
        next_states = []

        for i in range(n_episodes):
            n_steps = 0
            state = self.env.reset()[0]
            while True:
                action = self.action(state)
                next_state, reward, done, info, _ = self.env.step(action.numpy())
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)

                reward_sums += reward

                state = next_state
                n_steps += 1

                if done or n_steps >= max_steps:
                    break
            
            step_sums += n_steps

            batches.append((deepcopy(states), deepcopy(actions), deepcopy(rewards), deepcopy(next_states)))

            states.clear()
            actions.clear()
            rewards.clear()
            next_states.clear()

            if (i and i % self.batch_size == 0) or i == n_episodes-1:
                print('Batch {} (Epi [{}/{}]): avg steps: {} avg reward: {}'.format(i // self.batch_size, i, n_episodes, step_sums / (i-prev_i), reward_sums / (i-prev_i)))
                step_sums, reward_sums = 0, 0
                prev_i = i

                self.update(batches)
                batches.clear()

    def update(self, batches):
        loss = 0

        for (states, actions, rewards, next_states) in batches:
            states = torch.FloatTensor(states)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            if self.action_type == 0:
                actions = torch.LongTensor(actions)
                actions.unsqueeze(dim=-1)

                batch_output, state_values = self.model(states)
                batch_dists = Categorical(batch_output)
            elif self.action_type == 1:
                actions = torch.FloatTensor(actions)
                
                (batch_means, batch_stddevs), state_values = self.model(states)
                batch_dists = Normal(batch_means, batch_stddevs)

            log_probs = batch_dists.log_prob(actions)
            _, next_state_values = self.model(next_states)
            
            TD = rewards + self.discount_factor * next_state_values - state_values

            tmp = -(TD.detach() * log_probs) + TD * TD
            loss += tmp.sum()
            
        loss /= len(batches)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, n_episodes, max_steps, eps=0):
        step_sums = 0
        reward_sums = 0
        prev_i = 0

        for i in range(n_episodes):
            n_steps = 0
            state = self.env.reset()[0]
            while True:
                action = self.action(state)
                next_state, reward, done, info, _ = self.env.step(action.numpy())
                state = next_state

                reward_sums += reward
                n_steps += 1
                if done or n_steps >= max_steps:
                    break
            
            step_sums += n_steps
            
            if (i % (n_episodes // 10) == 0) or i == n_episodes-1:
                print('(Epi [{}/{}]): avg steps: {} avg reward: {}'.format(i // self.batch_size, i, n_episodes, step_sums / (i-prev_i), reward_sums / (i-prev_i)))
                step_sums, reward_sums = 0, 0
                prev_i = i
