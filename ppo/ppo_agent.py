import torch
import torch.nn.functional as F
import numpy as np
from utils import compute_advantage,orthogonal_init,convert_transition
from ppo.ppo_net import Net

class PPOAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, is_main=True, lr=1e-4, lambd=0.97, epochs=5, eps=0.2, entropy_coef=0.01, gamma=0.995,pre_model=None, device=None):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net=Net(state_dim,hidden_dim,action_dim).to(device)

        self.gamma=gamma
        self.lambd=lambd
        self.epochs=epochs#一条序列的数据用来训练轮数
        self.eps=eps#PPO中截断范围中的参数
        self.entropy_coef=entropy_coef
        self.device=device

        self.is_last_model=True
        self.is_training=True

        self.is_main=is_main

        if self.is_main:
            self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, eps=5e-6)

        if pre_model is not None:
            self.load_model(pre_model,is_last_model=True)
        else:
            orthogonal_init(self.net)

        self.random_mode=False

        self.id=1


    def take_action(self,state,legal_action):
        state = np.array(state)

        if self.random_mode:#随机模式，避免对局过于相似
            legal_action = np.array(legal_action)
            legal_action = (legal_action == 0)

            action = self.id * 10 + np.argmax((state[self.id * 10:(self.id + 1) * 10] == 0).astype(np.float))
            if not legal_action[action]:
                p_action = legal_action.astype(np.float)
                action = np.random.choice(range(legal_action.size), 1, p=p_action / sum(p_action))

        else:
            state=torch.tensor(state,dtype=torch.float).unsqueeze(0).to(self.device)
            if not torch.is_tensor(legal_action):
                legal_action=torch.tensor(legal_action,dtype=torch.int).unsqueeze(0).to(self.device)
            legal_action = legal_action == 0
            probs,_=self.net(state)

            action=self.select_legal_action(probs,legal_action,use_max=not self.is_training)
        return action

    def _legal_soft_max(self, input_hidden, legal_action):
        _const_w, _const_e = 1e18, 1e-5

        tmp = input_hidden - _const_w * (1.0 - legal_action)
        max_tmp = torch.max(tmp,dim=-1, keepdim=True)[0]
        tmp = torch.clip(tmp - max_tmp, -_const_w, 1)
        tmp=torch.exp(tmp)+_const_e

        probs = tmp / torch.sum(tmp,dim=-1, keepdim=True)
        return probs

    def legal_probs(self,probs,legal_actions,mask_prob=0.0,use_random=False):
        if use_random:
            p_size=probs.size(-1)
            probs=0.8*probs+0.2*torch.tensor(np.random.dirichlet(0.12*np.ones(p_size))).reshape(-1,p_size).to(probs)

        probs[~legal_actions]= mask_prob
        probs/=probs.sum()
        return probs

    def select_legal_action(self,probabilities, legal_actions,use_max=False):

        masked_probabilities = probabilities.clone()

        masked_probabilities = self.legal_probs(masked_probabilities, legal_actions,use_random=not use_max)
        if use_max:
            action=torch.argmax(masked_probabilities,dim=-1)
        else:
            try:
                action = torch.multinomial(masked_probabilities, 1)  # 根据概率进行采样，选择一个动作
            except Exception as e:
                print(e)
                p=(legal_actions.float()/(legal_actions.float().sum())).flatten().detach().cpu().numpy()
                action=np.random.choice(range(legal_actions.size(-1)),legal_actions.size(0),p=p)

        return action.item()

    def load_model(self,model,is_last_model=False):

        self.net.load_state_dict(torch.load(model))
        self.is_last_model=is_last_model

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self,state_dict):
        self.net.load_state_dict(state_dict)


    def update(self, transitions):
        if not self.is_main or not isinstance(transitions, list):
            transitions=[transitions]
        if len(transitions)==0:
            return
        states_list, actions_list, rewards_list, next_states_list, dones_list, legal_actions_list,advantage_list=[],[],[],[],[],[],[]

        #将收集的训练样本转换为训练数据
        for transition in transitions:
            states, actions, rewards, next_states, dones, legal_actions,flags = convert_transition(transition, self.device)
            length=states.size(0)

            td_target = rewards + self.gamma * self.net.values(next_states) * (1 - dones)
            td_delta = td_target - self.net.values(states)
            advantage = compute_advantage(self.gamma, self.lambd, td_delta.cpu()).to(self.device)#优势函数

            states_list.extend(torch.chunk(states,length))
            next_states_list.extend(torch.chunk(next_states, length))
            actions_list.extend(torch.chunk(actions,length))
            rewards_list.extend(torch.chunk(rewards,length))
            dones_list.extend(torch.chunk(dones,length))
            legal_actions_list.extend(torch.chunk(legal_actions,length))
            advantage_list.extend(torch.chunk(advantage,length))
        length=len(states_list)
        states=torch.stack(states_list,dim=0).view(length,-1)
        actions=torch.stack(actions_list,dim=0).view(length,-1)
        rewards=torch.stack(rewards_list,dim=0).view(length,-1)
        next_states=torch.stack(next_states_list,dim=0).view(length,-1)
        dones=torch.stack(dones_list,dim=0).view(length,-1)
        legal_actions=torch.stack(legal_actions_list,dim=0).view(length,-1)
        advantage=torch.stack(advantage_list,dim=0).view(length,-1)#advantage用于指导优化策略网络
        td_target = rewards + self.gamma * self.net.values(next_states) * (1 - dones) #td_target,用于优化价值网络
        old_probs = self.net.probs(states)

        old_log_probs = torch.log(old_probs.gather(1, actions)).detach()

        for _ in range(self.epochs):
            probs, values = self.net(states)

            e_probs = self._legal_soft_max(probs, legal_actions)


            log_probs = torch.log(probs.gather(1, actions))

            entropy =-torch.sum(e_probs * torch.log(e_probs))

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断

            actor_loss = torch.mean(-torch.min(surr1, surr2) - entropy * self.entropy_coef)  # 策略网络损失函数
            critic_loss = torch.mean(
                F.mse_loss(values, td_target.detach())
            )#价值网络损失函数

            loss = actor_loss + critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)#梯度裁剪
            self.optimizer.step()

