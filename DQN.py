import random
from collections import deque
import numpy as np
import torch

class DQN():
    def __init__(self, env, model, loss_fn, optimizer, actions, update_epi=10, discount_factor=0.99, eps=1e-1, batch_size=32, early_stopping_torelance=0, epsilon_decay=1):
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

        self.early_stopping_torelance = early_stopping_torelance
        self.epsilon_decay = epsilon_decay

    def train(self, n_episodes):
        max_avg = 0
        torelance = 0

        sums = 0
        for i in range(n_episodes):
            n_steps = 0
            obs1 = self.env.reset()[0]
            done = False
            while not done:
                action = self.action(obs1)
                
                obs2, reward, done, info, _ = self.env.step(action)
                self.replay_buffer.append((obs1, action, reward, obs2))

                if done:
                    break
                obs1 = obs2
                n_steps += 1

            sums += n_steps
            if i % self.update_epi == 0 or i == n_episodes-1:
                avg = sums // self.update_epi
                print('Episode {} => average steps: {}'.format(i, avg))
                sums = 0
                self.eps *= self.epsilon_decay
                if self.early_stopping_torelance and max_avg < avg:
                    max_avg = avg
                else:
                    torelance += 1
                
                print('Updating policy.. ', end='')
                self.update()
                self.replay_buffer.clear()
                print('Done!')
            
            if self.early_stopping_torelance and self.early_stopping_torelance <= torelance:
                print('Early Stopping!')
                break

    def update(self):
        batches = random.choices(self.replay_buffer, k=self.batch_size)

        current_states = []
        actions = []
        rewards = []
        next_states = []
        for (current_state, action, reward, next_state) in batches:
            current_states.append(current_state)
            actions.append([action])
            rewards.append([reward])
            next_states.append(next_state)

        current_states = torch.FloatTensor(np.array(current_states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))

        current_Q = self.model(current_states)
        next_Q = self.model(next_states)

        target = rewards + self.discount_factor * next_Q.max(axis=1)[0].reshape(-1, 1)
        loss = self.loss_fn(current_Q.gather(1, actions), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, n_episodes, eps=0):
        self.eps = eps
        sums = 0
        for i in range(n_episodes):
            n_steps = 0
            obs1 = self.env.reset()[0]
            done = False
            while not done:
                action = self.action(obs1)
                obs2, reward, done, info, _ = self.env.step(action)
                if done:
                    break
                obs1 = obs2
                n_steps += 1

            sums += n_steps
            if i % self.update_epi == 0 or i == n_episodes-1:
                print('Episode {} => average steps: {}'.format(i, sums // self.update_epi))
                sums = 0

    def action(self, state):
        p = np.random.rand(1)
        if p < self.eps:
            return self.actions[np.random.randint(len(self.actions))]
        else:
            state = torch.tensor(state, dtype=torch.float32)
            idx = torch.argmax(self.model(state))
            return self.actions[idx]
