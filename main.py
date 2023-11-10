from train import train
from env.go_bang_env import GoBangEnv
from ppo.ppo_agent import PPOAgent
import time


def main():
    start_time = time.time()
    try:
        width = 10
        height = 10
        state_dim = width * height
        env = GoBangEnv(width, height)
        pre_model=None
        agents = [PPOAgent(state_dim, 256, state_dim, is_main=False,pre_model=pre_model), PPOAgent(state_dim, 256, state_dim,lr=7e-5, is_main=True,pre_model=pre_model)]
        train(env, agents, 90000)

    except KeyboardInterrupt:   # 当用户按下Ctrl+C中断程序时，会抛出KeyboardInterrupt异常
        print('\n程序被中断！')
    except Exception as e:   # 当程序出现其他异常时，会抛出相应的异常对象
        print('程序出现异常：', e)
    finally:
        end_time = time.time()

        duration_seconds = end_time - start_time  # 计算时间差（单位：秒）
        duration_hours = int(duration_seconds // 3600)  # 计算小时数
        duration_minutes = int((duration_seconds % 3600) // 60)  # 计算分钟数

        print("总运行时间为：{}小时 {}分钟".format(duration_hours, duration_minutes))

if __name__ == '__main__':
    main()