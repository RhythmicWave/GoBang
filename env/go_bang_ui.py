import pygame
from env.go_bang_env import GoBangEnv

class GoBangUI(GoBangEnv):
    # 设置颜色
    WHITE = (255, 255, 255)
    BOARD_COLOR = (240, 170, 120)
    BLACK = (0, 0, 0)
    def __init__(self,width,height):
        super(GoBangUI,self).__init__(width,height)
        # 设置棋盘格子大小
        self.GRID_SIZE = 60

        width_win_size=self.width*self.GRID_SIZE
        height_win_size=self.height*self.GRID_SIZE
        # 初始化pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width_win_size, height_win_size))
        self.draw_board()

    def draw_board(self,board=None):
        if board is None:
            board=self.board
        board=board.reshape((self.width,self.height))
        self.screen.fill(self.BOARD_COLOR)

        for row in range(self.width):
            for col in range(self.height):
                if board[row, col] == 1:
                    pygame.draw.circle(self.screen, self.BLACK, (col * self.GRID_SIZE + self.GRID_SIZE // 2, row * self.GRID_SIZE + self.GRID_SIZE // 2),
                                       self.GRID_SIZE // 2 - 2)
                elif board[row, col] == -1:
                    pygame.draw.circle(self.screen, self.WHITE, (col * self.GRID_SIZE + self.GRID_SIZE // 2, row * self.GRID_SIZE + self.GRID_SIZE // 2),
                                       self.GRID_SIZE // 2 - 2)

                pygame.draw.rect(self.screen, self.BLACK, (col * self.GRID_SIZE, row * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE), 1)

        pygame.display.update()

    def take_action(self,state,legal_action):#对齐Agent的take_action
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 0,0
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    row=event.pos[1] // self.GRID_SIZE
                    col=event.pos[0] // self.GRID_SIZE
                    index=row*self.width+col
                    if self.board[index]==0:
                        return row*self.width+col

    def __del__(self):
        pygame.quit()