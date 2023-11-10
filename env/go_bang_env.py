import numpy as np

class GoBangEnv:
    def __init__(self,width,height):
        self.width=width
        self.height=height
        self.done=False
        self.board=np.zeros(width * height)
        self.count=self.width*self.height#表示棋盘剩余空位
        self.flag=1 #1表示黑棋,-1表示白棋


    def reset(self,flag=1,board=None):
        self.done=False
        self.flag = flag#黑棋先手
        if board is not None:
            self.board=np.copy(board)
            self.count=board.size-np.count_nonzero(board)
        else:
            self.count=self.width*self.height
            self.board=np.zeros(self.width*self.height)

        return np.copy(self.board)

    def step(self, action, flag=None):
        if flag is None:
            flag = self.flag

        reward1 = -0.003
        reward2 = -0.003


        if self.board[action] == 0:#判断是否为合法动作
            self.flag *= -1#黑白棋交替落子

            self.board[action] = flag
            self.done=self.check_win(self.board.reshape(self.width,self.height),flag, n=5, check_value=5)
            self.count -= 1
            if self.done:
                base_rwd =3*( 0.5 + 0.56 * self.count / self.board.size)

                reward1 -= flag * base_rwd
                reward2 += flag * base_rwd

            if self.count <= 5:# 看作和棋
                self.done = True
                reward1 -= 1
                reward2 -= 1
        else:
            print("不合法动作!")
        return np.copy(self.board), [reward1,reward2], self.done,self.flag,None

    def check_win(self,board, flag, n, check_value):
        width, height = board.shape
        # 检查水平方向
        for row in range(width):
            for col in range(height - n + 1):
                piece = board[row, col:col + n]
                if flag in piece:
                    value = np.sum(piece)
                    value = int(value)
                    if np.sign(value) == np.sign(flag) and abs(value) >= check_value:
                        return True

        # 检查垂直方向
        for row in range(width - n + 1):
            for col in range(height):
                piece = board[row:row + n, col]
                if flag in piece:
                    value = np.sum(piece)
                    value = int(value)
                    if np.sign(value) == np.sign(flag) and abs(value) >= check_value:
                        return True

        # 检查对角线方向（左上到右下）
        for row in range(width - n + 1):
            for col in range(height - n + 1):
                piece = np.diagonal(board[row:row + n, col:col + n])
                if flag in piece:
                    value = np.sum(piece)
                    value = int(value)
                    if np.sign(value) == np.sign(flag) and abs(value) >= check_value:
                        return True

        # 检查对角线方向（右上到左下）
        for row in range(width - n + 1):
            for col in range(n - 1, height):
                piece = np.diag(np.fliplr(board[row:row + n, col - n + 1:col + 1]))
                if flag in piece:
                    value = np.sum(piece)
                    value = int(value)
                    if np.sign(value) == np.sign(flag) and abs(value) >= check_value:
                        return True

        return False
