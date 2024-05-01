from tools import *
import time
import csv
from Mec import MecEnv
import random

def rand_action(state):
    user_data_size = state[0]
    require_cpu = state[1]
    tolerant_delay = state[2]
    user_coordinate = (state[3], state[4])
    remain_capacity = state[5:(5 + Constants.edge_num)]

    server_index = []  # 获取可连接的服务器
    u2s_distance = []
    action_index = [0]
    for i in range(Constants.edge_num):
        u2s_dis = calcu_dis(user_coordinate[0], user_coordinate[1], Constants.server_coordinate[i][0], Constants.server_coordinate[i][1])
        if u2s_dis <= Constants.coverage_radius[i]:  # 满足距离需求
            server_index.append(i+1)
            u2s_distance.append(u2s_dis)
    trans_time = [float("inf") for _ in range(Constants.edge_num)]
    for i in range(len(u2s_distance)):
        trans_rate = calcu_trans_rate(u2s_distance[i])
        trans_time[server_index[i]-1] = user_data_size * 1024 / trans_rate
        min_required_resource = require_cpu / (tolerant_delay - trans_time[server_index[i]-1])
        if tolerant_delay and remain_capacity[server_index[i]-1] >= min_required_resource:
            action_index.append(server_index[i])

    index = random.choice(action_index)  # 随机选择一个服务器
    param = 1

    # if index > 0:
    #     param = np.round(np.random.uniform(0, 1), 3)  # 生成资源分配的参数 [0,1),保留三位小数
    # else:
    #     param = 1
        # param = np.round(np.random.uniform(0.95, 1), 3)
    all_params = [np.zeros((1,)) for _ in range(Constants.edge_num + 1)]  #
    all_params[index][:] = param
    return (index, all_params)

if __name__ == '__main__':
    env = MecEnv(seed=2)  # seed=1
    max_steps = 5000
    total_reward = 0.
    returns = []
    start_time = time.time()
    best = -float("inf")

    f = open('./log/N/100N_RO.csv', 'w', encoding='utf-8', newline="")  # 这个文件用于存放实验过程中的输出情况
    csv_write = csv.writer(f)
    csv_write.writerow(['Episode', 'total_r_avg', 'episode_reward', 'episode_cost', 'episode_user_num', 'episode_t'])

    for i in range(Constants.episodes):
        print("epoch:", i)
        state = env.reset()
        action = rand_action(state)
        episode_reward = 0.
        episode_user_num = 0
        episode_cost = 0
        episode_t_exe = 0
        for j in range(max_steps):
            next_state, reward, cost, user_num, t_exe, terminal, _ = env.step(action)
            next_action = rand_action(next_state)
            action = next_action
            state = next_state

            episode_reward += reward  # calculate the episode reward
            episode_user_num += user_num  # 每次回合中能服务的用户数量
            episode_cost += cost  # cost表示任务成功执行的能耗，0表示任务被丢弃，无法成功完成
            episode_t_exe += t_exe

            if terminal:
                break
        returns.append(episode_reward)
        total_reward += episode_reward

        csv_write.writerow([i, total_reward / (i + 1), episode_reward, episode_cost, episode_user_num, episode_t_exe])

        if episode_reward > best:
            best = episode_reward

    f.close()
    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    print(best * 500.0)
    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

    # python RO.py >./log/N/100N_RO.log 2>&1&