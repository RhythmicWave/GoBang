from env.go_bang_ui import GoBangUI
import random
from ppo.ppo_agent import PPOAgent
import numpy as np

class Game:
    PVP = 0  # 玩家对战
    PVA = 1  # 玩家与AI对战
    AVA = 2  # AI与AI对战
    def __init__(self,width=10,height=10,mode=0,ai_model=None):
        self.width=width
        self.height=height
        self.ui_game=GoBangUI(width, height)

        if type(ai_model) is not list and type(ai_model) is not tuple:
            ai_model=[ai_model]
        if len(ai_model)==1:
            self.ai_model=[ai_model[0],ai_model[0]]
        else:
            self.ai_model=ai_model

        self.players=[None,None]
        self.set_mode(mode)

    def set_mode(self,mode):
        state_dim=self.width*self.height
        if mode==self.PVA:
            if self.ai_model is None:
                raise Exception("No AI Model!")
            else:
                index=0
                agent=PPOAgent(state_dim, 256, state_dim)
                agent.is_training=False
                agent.load_model(self.ai_model[0])
                self.players[index]=agent
                self.players[1-index]=self.ui_game
        elif mode==self.AVA:
            if self.ai_model is None:
                raise Exception("No AI Model!")
            else:
                self.players=[PPOAgent(state_dim, 256, state_dim), PPOAgent(state_dim, 256, state_dim)]
                self.players[0].load_model(self.ai_model[0])
                self.players[0].is_training=False
                self.players[1].load_model(self.ai_model[1])
                self.players[1].is_training = False
        elif mode==self.PVP:
            self.players=[self.ui_game,self.ui_game]

    def run(self):
        current_player=1 if random.random()>0.5 else 0
        flag=1
        done=False
        state=self.ui_game.reset()
        while not done:
            legal_action = np.copy(state)
            action = self.players[current_player].take_action(state * flag, legal_action)
            state, _, done, flag, _ = self.ui_game.step(action)
            self.ui_game.draw_board()
            current_player=1-current_player

            if done:
                if flag*-1==1:
                    print(f"player {current_player}, 黑棋获胜!")
                else:
                    print(f"player {current_player}, 白棋获胜!")
        self.ui_game.take_action(state, state)#避免结束游戏时UI直接关闭


if __name__ == '__main__':
    game=Game(width=10,height=10,mode=Game.PVA,ai_model=f"checkpoints\\model_2093.pth")
    game.run()


