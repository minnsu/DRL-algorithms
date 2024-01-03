import random
from collections import deque
import numpy as np
import torch
from copy import deepcopy

class DeepQNetwork():
    def __init__(self, env, model, loss_fn, optimizer, actions, update_epi=10, discount_factor=0.99, eps=1e-1, batch_size=32, epsilon_decay=1):
        self.env = env

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.actions = actions

        self.replay_buffer = deque(maxlen=10000)
        
        self.update_epi = update_epi
        self.discount_factor = discount_factor
        self.eps = eps
        self.batch_size = batch_size

        self.epsilon_decay = epsilon_decay

        self.__best_model__ = None

    def action(self, state):
        p = np.random.rand(1)
        if p < self.eps:
            return self.actions[np.random.randint(len(self.actions))]
        else:
            state = torch.tensor(state, dtype=torch.float32)
            idx = torch.argmax(self.model(state))
            return self.actions[idx]

    def train(self, n_episodes, max_steps):
        step_sums = 0
        reward_sums = 0
        prev_i = 0

        for i in range(n_episodes):
            n_steps = 0
            state = self.env.reset()[0]
            rewards = 0
            while True:
                action = self.action(state)
                
                next_state, reward, done, info, _ = self.env.step(action)
                self.replay_buffer.append((state, action, reward, next_state))

                rewards += reward
                state = next_state
                n_steps += 1
                if done or n_steps >= max_steps:
                    break

            step_sums += n_steps
            reward_sums += rewards
            if (i and i % self.update_epi == 0) or i == n_episodes-1):

                print('Episode [{}] => average steps: {} avg reward: {}'.format(i // self.batch_size, i, n_episodes, step_sums / (i-prev_i), reward_sums / (i-prev_i)))
                
                print('Updating policy.. ', end='')
                self.update()
                print('Done!')

                step_sums, reward_sums = 0, 0
                max_steps = 0
                self.eps *= self.epsilon_decay
                prev_i = i
            

    def update(self):
        batches = random.choices(self.replay_buffer, k=self.batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        for (state, action, reward, next_state) in batches:
            states.append(state)
            actions.append([action])
            rewards.append([reward])
            next_states.append(next_state)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))

        current_Q = self.model(states)
        next_Q = self.model(next_states)

        target = rewards + self.discount_factor * next_Q.max(axis=1)[0].reshape(-1, 1)
        loss = self.loss_fn(current_Q.gather(1, actions), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, n_episodes, max_steps, eps=0):
        tmp = self.eps
        self.eps = eps

        step_sums = 0
        prev_i = 0
        for i in range(n_episodes):
            n_steps = 0
            state = self.env.reset()[0]
            while True:
                action = self.action(state)
                next_state, reward, done, info, _ = self.env.step(action)
                state = next_state
                n_steps += 1
                if done or n_steps >= max_steps:
                    break

            step_sums += n_steps
            if (i % (n_episodes // 10) == 0) or i == n_episodes-1:
                print('Episode [{}] => average steps: {}'.format(i, step_sums // (i-prev_i+1)))
                step_sums = 0
                prev_i = i
        
        self.eps = tmp
