import numpy as np
import torch
from copy import deepcopy
from torch.distributions import Normal, Categorical

class PolicyGradient():
    def __init__(self, env, model, optimizer, batch_size=32, action_space_type='discrete'):
        self.env = env

        self.model = model
        self.optimizer = optimizer

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
            output = self.model(state)
            distrib = Categorical(probs=output)
        elif self.action_type == 1:
            # output is means and stddevs
            means, stddevs = self.model(state)
            distrib = Normal(means, stddevs)

        action = distrib.sample()
        return action
    
    def train(self, n_episodes, max_steps):
        step_sums = 0
        reward_sums = 0

        prev_i = 0
        states = []
        actions = []
        rewards = []
        for i in range(n_episodes):
            n_steps = 0
            state = self.env.reset()[0]
            
            epi_rewards = 0

            while True:
                action = self.action(state)
                next_state, reward, done, info, _ = self.env.step(action.numpy())
                
                states.append(state)
                actions.append(action)
                epi_rewards += reward

                state = next_state
                n_steps += 1

                if done or n_steps >= max_steps:
                    rewards += [epi_rewards] * n_steps
                    break
            
            step_sums += n_steps
            reward_sums += epi_rewards
            
            if (i and i % self.batch_size == 0) or i == n_episodes-1:
                print('Batch {} (Epi [{}/{}]): avg steps: {} avg reward: {}'.format(i // self.batch_size, i, n_episodes, step_sums / (i-prev_i), reward_sums / (i-prev_i)))
                step_sums, reward_sums = 0, 0
                prev_i = i

                self.update(states, actions, rewards)
                states.clear()
                actions.clear()
                rewards.clear()

    def update(self, states, actions, rewards):
        
        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards)
        if self.action_type == 0:
            actions = torch.LongTensor(actions)
            actions.unsqueeze(dim=-1)
        elif self.action_type == 1:
            actions = torch.FloatTensor(actions)

        self.optimizer.zero_grad()

        if self.action_type == 0:
            batch_output = self.model(states)
            batch_dists = Categorical(batch_output)
        elif self.action_type == 1:
            batch_means, batch_stddevs = self.model(states)
            batch_dists = Normal(batch_means, batch_stddevs)
        
        log_probs = batch_dists.log_prob(actions)
        loss = -(log_probs * rewards).sum()
        loss.backward()

        self.optimizer.step()

    def test(self, n_episodes, max_steps):
        step_sums = 0
        rewards_sums = 0
        prev_i = 0

        for i in range(n_episodes):
            epi_rewards = 0
            n_steps = 0
            state = self.env.reset()[0]
            while True:
                action = self.action(state)
                next_state, reward, done, info, _ = self.env.step(action.numpy())
                
                epi_rewards += reward
                
                state = next_state
                n_steps += 1
                if done or n_steps >= max_steps:
                    break
            
            step_sums += n_steps
            rewards_sums += epi_rewards

            if (i % (n_episodes // 10) == 0) or i == n_episodes-1:
                print('Batch {} (Epi [{}/{}]): avg steps: {} avg reward: {}'.format(i // self.batch_size, i, n_episodes, step_sums / (i-prev_i), reward_sums / (i-prev_i)))
                step_sums, reward_sums = 0, 0
                prev_i = i


class Reinforce():
    def __init__(self, env, model, optimizer, discount_factor, batch_size=32, action_space_type='discrete'):
        self.env = env

        self.model = model
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
            output = self.model(state)
            distrib = Categorical(probs=output)
        elif self.action_type == 1:
            # output is means and stddevs
            means, stddevs = self.model(state)
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
        for i in range(n_episodes):
            n_steps = 0
            state = self.env.reset()[0]
            while True:
                action = self.action(state)
                next_state, reward, done, info, _ = self.env.step(action.numpy())
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state
                n_steps += 1

                if done or n_steps >= max_steps:
                    break
            
            step_sums += n_steps
            reward_sums += sum(rewards)

            for j in range(n_steps-2, -1, -1):
                rewards[j] += self.discount_factor * rewards[j+1]

            batches.append((deepcopy(states), deepcopy(actions), deepcopy(rewards)))

            states.clear()
            actions.clear()
            rewards.clear()

            if (i and i % self.batch_size == 0) or i == n_episodes-1:
                print('Batch {} (Epi [{}/{}]): avg steps: {} avg reward: {}'.format(i // self.batch_size, i, n_episodes, step_sums / (i-prev_i), reward_sums / (i-prev_i)))
                step_sums, reward_sums = 0, 0
                prev_i = i

                self.update(batches)
                batches.clear()

    def update(self, batches):
        
        loss = 0
        self.optimizer.zero_grad()

        for (states, actions, rewards) in batches:
            states = torch.FloatTensor(states)
            rewards = torch.FloatTensor(rewards)
            if self.action_type == 0:
                actions = torch.LongTensor(actions)
                actions.unsqueeze(dim=-1)
            elif self.action_type == 1:
                actions = torch.FloatTensor(actions)


            if self.action_type == 0:
                batch_output = self.model(states)
                batch_dists = Categorical(batch_output)
            elif self.action_type == 1:
                batch_means, batch_stddevs = self.model(states)
                batch_dists = Normal(batch_means, batch_stddevs)
            
            log_probs = batch_dists.log_prob(actions)
            loss += -(log_probs * rewards).sum()
        
        loss /= len(batches)
        loss.backward()
        self.optimizer.step()

    def test(self, n_episodes, max_steps, eps=0):
        step_sums = 0
        reward_sums = 0
        prev_i = 0

        for i in range(n_episodes):
            n_steps = 0
            state = self.env.reset()[0]
            rewards = 0
            while True:
                action = self.action(state)
                next_state, reward, done, info, _ = self.env.step(action.numpy())
                state = next_state

                rewards += reward
                n_steps += 1
                if done or n_steps >= max_steps:
                    break
            
            step_sums += n_steps
            reward_sums += rewards
            
            if (i % (n_episodes // 10) == 0) or i == n_episodes-1:
                print('(Epi [{}/{}]): avg steps: {} avg reward: {}'.format(i // self.batch_size, i, n_episodes, step_sums / (i-prev_i), reward_sums / (i-prev_i)))
                step_sums, reward_sums = 0, 0
                prev_i = i
