import numpy as np
from utils import convert_transition

def expand_equi_data(width, height, transition):  #等价棋谱

    extend_data = []
    size = width * height
    states,actions,rewards,next_states,dones,legal_action,flags=convert_transition(transition, device=None, trans_tensor=False)
    states=states.reshape((-1,width,height))
    next_states=next_states.reshape((-1,width,height))
    legal_action=legal_action.reshape((-1,width,height))

    action_mat=np.zeros_like(states)

    action_mat[range(actions.size),actions//width,actions%height]=1.0

    for i in [1, 2, 3, 4]:  # 将棋盘进行翻转后,不会影响局势,故可以用来扩充.
        equi_states = np.array([np.rot90(s, i) for s in states])
        equi_next_states=np.array([np.rot90(s, i) for s in next_states])
        equi_action_mat=np.array([np.rot90(s, i) for s in action_mat])
        equi_action=np.argmax(equi_action_mat.reshape((-1,size)),axis=-1)
        equi_legal_action=np.array([np.rot90(s, i) for s in legal_action])

        trans={'states': equi_states.reshape((-1,size)), 'actions': equi_action.flatten(), 'next_states': equi_next_states.reshape((-1,size)), 'rewards': np.copy(rewards), 'dones': np.copy(dones), "legal_action":equi_legal_action.reshape((-1,size)),"flags":np.copy(flags)}
        extend_data.append(trans)


        equi_states = np.array([np.fliplr(s) for s in equi_states])  # 上下翻转
        equi_next_states = np.array([np.fliplr(s) for s in equi_next_states])
        equi_legal_action = np.array([np.fliplr(s) for s in equi_legal_action])
        equi_action_mat=np.array([np.fliplr(s) for s in equi_action_mat])
        equi_action = np.argmax(equi_action_mat.reshape((-1,size)),axis=-1)
        trans={'states': equi_states.reshape((-1,size)), 'actions': equi_action.flatten(), 'next_states': equi_next_states.reshape((-1,size)), 'rewards': np.copy(rewards), 'dones': np.copy(dones), "legal_action":equi_legal_action.reshape((-1,size)),"flags":np.copy(flags)}
        extend_data.append(trans)
    return extend_data