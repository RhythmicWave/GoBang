import torch
import numpy as np

def orthogonal_init(layer, gain=np.sqrt(1)):#正交初始化
    for name, param in layer.named_parameters():
        if torch.is_tensor(param) and len(param.size())>1:
            if "weight" in name or "bias" in name:
                torch.nn.init.orthogonal_(param, gain=gain)

    return layer

def compute_advantage(gamma, lambd, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lambd * advantage + delta
        advantage_list.append(advantage)

    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def convert_transition(dict, device, discrete_action=True, trans_tensor=True):
    '''
    转换参数
    :param dict: 参数字典
    :param device: cpu或gpu
    :param discrete_action: 是离散动作还是连续动作
    :return:
    '''
    states=np.array(dict["states"],dtype=np.float32)
    rewards=np.array(dict["rewards"],dtype=np.float32)
    next_states=np.array(dict["next_states"],dtype=np.float32)
    dones=np.array(dict["dones"],dtype=np.float32)
    legal_action=np.array(dict["legal_action"],dtype=np.int64)

    actions=np.array(dict["actions"],dtype=np.int64)
    flags=np.array(dict['flags'],dtype=np.float32)

    if trans_tensor:
        states = torch.tensor(states,
                              dtype=torch.float).to(device)
        rewards = torch.tensor(rewards,
                               dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor(next_states,
                                   dtype=torch.float).to(device)
        dones = torch.tensor(dones,
                             dtype=torch.float).view(-1, 1).to(device)
        flags = torch.tensor(flags,
                             dtype=torch.float).view(-1, 1).to(device)
        legal_action=torch.tensor(legal_action,
                                  dtype=torch.int).to(device)
        if(discrete_action):
            actions = torch.tensor(actions).view(-1, 1).to(
                device)  # 动作不再是float类型
        else:
            actions = torch.tensor(actions, dtype=torch.float).view(-1, 1).to(device)

    return states,actions,rewards,next_states,dones,legal_action,flags

