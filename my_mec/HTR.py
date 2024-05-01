import time
import csv
from Mec import MecEnv
from tools import *

def HTR_choose_action(state):     # 选择资源最充足的服务器，就可以服务最多的用户
    user_data_size = state[0]
    require_cpu = state[1]
    tolerant_delay = state[2]
    user_coordinate = (state[3], state[4])
    remain_capacity = state[5:(5 + Constants.edge_num)]

    server_index = []  # 获取可连接的服务器
    u2s_distance = []   # 可连接服务器的距离

    action_index = []   # 既满足距离要求又满足计算资源需求的服务器
    resource_index = []     # 满足条件的服务器的剩余资源列表

    for i in range(len(Constants.server_coordinate)):
        u2s_dis = calcu_dis(user_coordinate[0], user_coordinate[1], Constants.server_coordinate[i][0], Constants.server_coordinate[i][1])
        if u2s_dis <= Constants.coverage_radius[i]:  # 满足距离需求
            server_index.append(i + 1)
            u2s_distance.append(u2s_dis)
        else:
            print("not in the coverage are", Constants.server_coordinate[i])

    trans_time = [float("inf") for _ in range(Constants.edge_num)]
    for i in range(len(u2s_distance)):
        trans_rate = calcu_trans_rate(u2s_distance[i])
        trans_time[server_index[i] - 1] = user_data_size * 1024 / trans_rate
        # print("trans_time:",trans_time)
        min_required_resource = require_cpu / (tolerant_delay - trans_time[server_index[i] - 1])
        if tolerant_delay and remain_capacity[server_index[i] - 1] >= min_required_resource:
            action_index.append(server_index[i])
            resource_index.append(remain_capacity[server_index[i] - 1])
    index = 0
    max_gain = 0
    # 计算本地卸载时的能耗
    _, local_energy = local_compute(require_cpu, Constants.user_capacity)
    if resource_index:
        # 分别计算任务卸载到各个服务器的能耗
        server_energy_list = [float("-inf") for _ in range(len(action_index))]
        for m in range(len(action_index)):
            e_energy = edge_energy(trans_time[action_index[m]-1], require_cpu, remain_capacity[action_index[m]-1])
            server_energy_list[m] = e_energy
        # print(f"local energy: {local_energy}, edge energy: {server_energy_list}")

        gains = []
        for i in range(len(server_energy_list)):
            gains.append(local_energy - server_energy_list[i])
        #print("""gains""", gains)
        if len(gains):
            max_gain = gains[0]
            index = action_index[0]
            for k in range(1, len(gains)):
                if gains[k] > 0 and gains[k] > max_gain:
                    index = action_index[k]
                    max_gain = gains[k]
        else:
            index = 0
    # gain_new = 0
    if index:
        # param1 = np.round((require_cpu / (tolerant_delay - trans_time[index - 1])) / remain_capacity[index - 1], 16)
        # alloc_resource = np.ceil(remain_capacity[index - 1] * param1)
        # # _, local_energy_new =local_compute(require_cpu, Constants.user_capacity)
        # edge_energy_new = edge_energy(trans_time[index-1], require_cpu, alloc_resource)
        # gain_new = local_energy - edge_energy_new
        # print("gain new---------------", gain_new)
        # if gain_new > max_gain:
        #     param = param1
        # else:
        param = np.round(remain_capacity[index-1] / remain_capacity[index-1], 1)

        # param = np.round(remain_capacity[index - 1] / remain_capacity[index - 1], 1)
    else:
        # index = 0  # 4
        if tolerant_delay:
            param = 1
            # param = np.round((require_cpu / (tolerant_delay)) / Constants.user_capacity, 16)
        else:
            param = 0
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

    f = open('./log/B/100N_htr_test.csv', 'w', encoding='utf-8', newline="")  # 这个文件用于存放实验过程中的输出情况
    csv_write = csv.writer(f)
    csv_write.writerow(['Episode', 'total_r_avg', 'episode_reward', 'episode_cost', 'episode_user_num', 'episode_t'])

    for i in range(Constants.episodes):
        print("epoch:", i)
        state = env.reset()
        action = HTR_choose_action(state)
        episode_reward = 0.
        episode_user_num = 0
        episode_cost = 0
        episode_t_exe = 0
        for j in range(max_steps):
            next_state, reward, cost, user_num, t_exe, terminal, _ = env.step(action)
            next_action = HTR_choose_action(next_state)
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

    # nohup python HTR.py >./log/B/100N_htr_30M.log 2>&1&
