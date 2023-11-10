from tqdm import tqdm
import numpy as np
import random
import os
from model_manager import ModelManager
from env.go_bang_qeui_data import expand_equi_data

def train_on_policy_agent(env, agents, num_episodes,save_path="checkpoints"):
    return_list = []
    state=env.reset()
    last_model_prob=0.75
    flag = 1
    state_dim=state.size
    train_step=0
    model_manager=ModelManager(save_path)

    transition_buffer=[]
    main_agent_index=0
    mask_action = None
    for j,agent in enumerate(agents):
        if agent.is_main:
            main_agent_index=j
            break

    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                train_step+=1
                episode_return = 0
                transition_dict = {'states': [[],[]], 'actions': [[],[]], 'rewards': [[],[]], 'dones': [[],[]],'flags':[[],[]]}

                done = False
                steps=0
                index=1 if random.random()>0.5 else 0#index表示谁是黑棋

                transition_dict['states'][index].append(state)

                while not done:
                    if state[0].size>1:
                        state=state[0]

                    transition_dict["flags"][index].append(flag)

                    legal_action=np.copy(state)
                    if mask_action is not None:
                        legal_action[mask_action]=1
                        mask_action=None
                    action = agents[index].take_action(state*flag,legal_action=legal_action)#*flag
                    next_state, reward, done, flag,_= env.step(action)

                    transition_dict['actions'][index].append(action)
                    if not done:
                        transition_dict['states'][1 - index].append(next_state*flag)#*flag
                        if len(transition_dict['actions'][1-index])>0:
                            transition_dict['rewards'][1-index].append(reward[1-(flag*-1+1)//2])
                            transition_dict['dones'][1-index].append(done)

                    else:
                        transition_dict['states'][1 - index].append(next_state*flag)#*flag
                        transition_dict['states'][index].append(next_state*flag*-1)#*flag*-1
                        transition_dict['rewards'][index].append(reward[(flag*-1+1)//2])
                        transition_dict['rewards'][1-index].append(reward[1-(flag*-1+1)//2])
                        transition_dict['dones'][0].append(done)
                        transition_dict['dones'][1].append(done)
                        episode_return += transition_dict['rewards'][1][-1]

                    state = next_state

                    index=1-index
                    steps += 1

                transitions = [
                    {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'flags': [],"legal_action":[]},
                    {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'flags': [],"legal_action":[]}
                    ]
                for k,v in transition_dict.items():
                    if isinstance(v[0],list):
                        transitions[0][k] = v[0]
                        transitions[1][k] = v[1]
                    else:
                        transitions[0][k]=v
                        transitions[1][k]=v
                    if k=="states":
                        transitions[0][k]=transitions[0][k][:-1]
                        transitions[0]["next_states"]=v[0][1:]
                        transitions[0]["legal_action"]=transitions[0][k]
                        transitions[1][k] = transitions[1][k][:-1]
                        transitions[1]["next_states"]=v[1][1:]
                        transitions[1]["legal_action"] = transitions[1][k]

                return_list.append(episode_return)

                if len(transitions[main_agent_index]["states"])>0:
                    extend_data=expand_equi_data(env.width,env.height,transitions[main_agent_index])
                    transition_buffer.extend(extend_data)

                if agents[1-main_agent_index].is_last_model and len(transitions[1-main_agent_index]["states"])>0 and not agents[1-main_agent_index].random_mode:
                    extend_data = expand_equi_data(env.width, env.height, transitions[1-main_agent_index])

                    transition_buffer.extend(extend_data)



                if (i_episode+1) % 10 == 0:
                    print(f"used count:{state_dim-env.count}")
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

                if train_step %8 ==0:

                    agents[main_agent_index].update(transition_buffer)
                    transition_buffer.clear()
                    if agents[1-main_agent_index].is_last_model:
                        agents[1-main_agent_index].load_state_dict(agents[main_agent_index].state_dict())


                if train_step%43==0:
                    agents[1 - main_agent_index].random_mode = False
                    model_manager.save_model(agents[main_agent_index].state_dict())
                    model_name, is_last_model = model_manager.get_random_model(last_model_prob)

                    if model_name is not None:
                        agents[1-main_agent_index].load_model(os.path.join(save_path, model_name), is_last_model)

                flag = 1
                if (random.random()>0.8 or transitions[1]["rewards"][-1]<0) and len(transitions[1]["states"])>0 and not agents[1-main_agent_index].random_mode:
                    try:
                        length=len(transitions[1]['states'])
                        state_index = random.randint(length-min(7,length),length-min(3,length))

                        flag=transitions[1]['flags'][state_index]
                        mask_action=transitions[1]['actions'][state_index]

                        init_state = transitions[1]['states'][state_index]
                        state = env.reset(flag,np.copy(init_state))


                    except Exception as e:
                        print(e)
                        state=env.reset()
                else:
                    state = env.reset()



    return return_list


