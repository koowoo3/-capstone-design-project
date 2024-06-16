import gym
import pybullet_envs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

ENV = gym.make("InvertedPendulumSwingupBulletEnv-v0")
OBS_DIM = ENV.observation_space.shape[0]
ACT_DIM = ENV.action_space.shape[0]
ACT_LIMIT = ENV.action_space.high[0]
ENV.close()

ENABLE_GRAD_CLIPPING = True
GRAD_CLIP_MAX_NORM = 0.1


#########################################################################################################################
############ 이 template에서는 DO NOT CHANGE 부분을 제외하고 마음대로 수정, 구현 하시면 됩니다                    ############
#########################################################################################################################

## 주의 : "InvertedPendulumSwingupBulletEnv-v0"은 continuious action space 입니다.
## Asynchronous Advantage Actor-Critic(A3C)를 참고하면 도움이 될 것 입니다.


class NstepBuffer:
    '''
    Save n-step trainsitions to buffer
    '''
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state, action, reward, next_state, done):
        '''
        add sample to the buffer
        '''
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def sample(self):
        '''
        sample transitions from buffer
        '''
        return self.states, self.actions, self.rewards, self.next_states, self.dones
    
    def reset(self):
        '''
        reset buffer
        '''
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []


class ActorCritic(nn.Module):
    '''
    Pytorch module for Actor-Critic network
    '''
    def __init__(self):
        '''
        Define your architecture here
        '''
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(OBS_DIM, 256)
        self.fc2 = nn.Linear(256, 128)
        
        self.fc5 = nn.Linear(OBS_DIM,256)
        self.fc6 = nn.Linear(256, 128)
        
        self.fc3 = nn.Linear(OBS_DIM,256)
        self.fc4 = nn.Linear(256, 128)
        
        
        
        self.mean = nn.Linear(128, ACT_DIM)
        self.std = nn.Linear(128, ACT_DIM)

        self.value = nn.Linear(128, 1)


    def actor(self, states):
        '''
        Get action distribution (mean, std) for given states
        '''
        #assert NotImplementedError
        x = torch.tanh(self.fc1(states))
        x = torch.tanh(self.fc2(x))
        y = torch.tanh(self.fc5(states))
        y = torch.tanh(self.fc6(y))
        
        #x = torch.tanh(self.fc12(x))
        mean = self.mean(x)
        std = self.std(y)
        std = F.softplus(std) + 1e-5
        return mean, std

    def critic(self, states):
        '''
        Get values for given states
        '''

        x = torch.tanh(self.fc3(states))
        x = torch.tanh(self.fc4(x))
        x = self.value(x)
        return x
        
        

class Worker(object):
    def __init__(self, global_actor, global_epi, sync, finish, n_step, seed):
        self.env = gym.make('InvertedPendulumSwingupBulletEnv-v0')
        self.env.seed(seed)
        self.lr = 0.001
        self.gamma = 0.95
        self.entropy_coef = 0.01

        ############################################## DO NOT CHANGE ##############################################
        self.global_actor = global_actor
        self.global_epi = global_epi
        self.sync = sync
        self.finish = finish
        self.optimizer = optim.Adam(self.global_actor.parameters(), lr=self.lr)
        ###########################################################################################################  
        
        self.n_step = n_step
        self.local_actor = ActorCritic()
        self.nstep_buffer = NstepBuffer()

    def select_action(self, state):
        '''
        selects action given state

        return:
            continuous action value
        '''
        #assert NotImplementedError
        state = torch.FloatTensor(state).unsqueeze(0)
        #print(state)
        mean, std = self.local_actor.actor(state)
        
        #print("front")
        
        dist = Normal(mean, std)
        #print("back")
        action = dist.sample()
        return action.clamp(-ACT_LIMIT, ACT_LIMIT)
    

    def train_network(self, states, actions, rewards, next_states, dones):
        '''
        Advantage Actor-Critic training algorithm
        '''
        #print("12")
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        total_loss = None #TODO
        discounted_rewards = []
        R = 0
        cnt = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            cnt += 1
            if done:
                R = 0
            if not done and cnt == 1:
                R = self.local_actor.critic(next_states[-1]).item()
            R = reward + self.gamma * R
            
                
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        values = self.local_actor.critic(states)
        #next_values = self.local_actor.critic(next_states)
        advantages = discounted_rewards.unsqueeze(1) - values

        value_loss = F.mse_loss(values, discounted_rewards.unsqueeze(1))
        mean, std = self.local_actor.actor(states)
        
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(axis=-1, keepdim=True)
        #print(log_probs)
        actor_loss = -(log_probs * advantages).mean() - self.entropy_coef * dist.entropy().mean()

       

        # Total loss
        total_loss = actor_loss + value_loss
    
        ############################################## DO NOT CHANGE ##############################################
        # Global optimizer update 준비
        
        # Global Network와 Local Network의 모든 파라미터의 gradients를 0으로 초기화
        self.optimizer.zero_grad(set_to_none=False)
        
        total_loss.backward()

        # Gradient Clipping 관련 전역 변수가 정의되어 있는지 확인
        if 'ENABLE_GRAD_CLIPPING' in globals() and 'GRAD_CLIP_MAX_NORM' in globals():
            # 활성화 여부에 따라 Gradient Clipping 적용
            if ENABLE_GRAD_CLIPPING:
                torch.nn.utils.clip_grad_norm_(parameters=self.local_actor.parameters(), max_norm=GRAD_CLIP_MAX_NORM)

        # Local parameter를 global parameter로 전달
        for global_param, local_param in zip(self.global_actor.parameters(), self.local_actor.parameters()):
                global_param._grad = local_param.grad

        # Global optimizer update
        self.optimizer.step()

        # Global parameter를 local parameter로 전달
        self.local_actor.load_state_dict(self.global_actor.state_dict())
        ###########################################################################################################  

    def train(self):
        step = 1

        while True:
            state = self.env.reset()
            done = False

            while not done:
                #print('stop here1')
                action = self.select_action(state)
                #print('stop here2')
                next_state, reward, done, _ = self.env.step(action)
                #print('stop here3')
                self.nstep_buffer.add(state, action.item(), reward, next_state, done)
                #print('stop here4')

                # n step마다 한 번씩 train_network 함수 실행
                if step % self.n_step == 0 or done:
                    self.train_network(*self.nstep_buffer.sample())
                    self.nstep_buffer.reset()                    
                
                state = next_state
                step += 1

            ############################################## DO NOT CHANGE ##############################################
            # 에피소드 카운트 1 증가                
            with self.global_epi.get_lock():
                self.global_epi.value += 1
            
            # evaluation 종료 조건 달성 시 local process 종료
            if self.finish.value == 1:
                break

            # 매 에피소드마다 global actor의 evaluation이 끝날 때까지 대기 (evaluation 도중 파라미터 변화 방지)
            with self.sync:
                self.sync.wait()
            ###########################################################################################################

        self.env.close()