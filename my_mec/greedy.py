from tools import *
import time
import csv
from Mec import MecEnv

def greedy_action(state):
    user_data_size = state[0]
    require_cpu = state[1]
    tolerant_delay = state[2]
    user_coordinate = (state[3], state[4])
    remain_capacity = state[5:(5 + Constants.edge_num)]

    server_index = []  # 获取可行的动作集
    u2s_distance = []
    for i in range(Constants.edge_num):
        u2s_dis = calcu_dis(user_coordinate[0], user_coordinate[1], Constants.server_coordinate[i][0], Constants.server_coordinate[i][1])
        if u2s_dis <= Constants.coverage_radius[i]:  # 满足距离需求，查找哪些服务器剩余的计算资源能满足用户需求 这里需要把传输时延减掉
            server_index.append(i + 1)
            u2s_distance.append(u2s_dis)

    if server_index and tolerant_delay:
        trans_time = [float("inf") for _ in range(Constants.edge_num)]
        available_server = []  # 记录既满足距离要求，也满足计算资源需求
        ser_dis = [float("inf") for _ in range(Constants.edge_num)]  # 用于记录用户到边缘服务器的距离
        resorce_list = [float("-inf") for _ in range(Constants.edge_num)]  # 用于记录用户到边缘服务器的距离
        for i in range(len(u2s_distance)):
            trans_rate = calcu_trans_rate(u2s_distance[i])
            trans_time[server_index[i]-1] = user_data_size * 1024 / trans_rate
            min_required_resource = require_cpu / (tolerant_delay - trans_time[server_index[i]-1])  #
            # print("----min required resource----", min_required_resource)
            if trans_time[server_index[i]-1] < float("inf") and remain_capacity[server_index[i] - 1] >= min_required_resource:
                available_server.append(server_index[i])  # 这里可以选取剩余最大资源的服务器，或者选择距离最近的服务器
                ser_dis[server_index[i]-1] = u2s_distance[i]
                resorce_list[server_index[i]-1] = remain_capacity[server_index[i]-1]
        # print("trans time:", trans_time)
        if available_server:
            index = np.argmax(resorce_list)+1    #
            param = np.round(require_cpu /(tolerant_delay- trans_time[index-1]) / remain_capacity[index-1], 16)  # - trans_time[index-1]# 生成资源分配的参数 [0,1)# np.random.uniform(0.5, 1)
        else:
            index = 0
            param = np.round((require_cpu / (tolerant_delay)) / Constants.user_capacity, 16)
            # if param > 1:
            # param = 1
    else:
        index = 0  # 4
        if tolerant_delay:
            # param = 1
            #
            param = np.round((require_cpu / (tolerant_delay)) / Constants.user_capacity, 16)
        else:
            param = 0
        if param > 1:
            param = 1
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

    f = open('./log/B/100N_grd_30M.csv', 'w', encoding='utf-8', newline="")  # 这个文件用于存放实验过程中的输出情况
    csv_write = csv.writer(f)
    csv_write.writerow(['Episode', 'total_r_avg', 'episode_reward', 'episode_cost', 'episode_user_num', 'episode_t'])

    for i in range(Constants.episodes):
        print("epoch:", i)
        state = env.reset()
        action = greedy_action(state)
        episode_reward = 0.
        episode_user_num = 0
        episode_cost = 0
        episode_t_exe = 0
        for j in range(max_steps):
            next_state, reward, cost, user_num, t_exe, terminal, _ = env.step(action)
            next_action = greedy_action(next_state)
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

    # nohup python greedy.py >./log/B/100N_grd_30M.log 2>&1&