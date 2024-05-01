import time
from Mec import MecEnv
import csv
from tools import *

def VSVBP_choose_action(state, last_action):     # 选择资源最充足的服务器，就可以服务最多的用户
    user_data_size = state[0]
    require_cpu = state[1]
    tolerant_delay = state[2]
    user_coordinate = (state[3], state[4])
    remain_capacity = state[5:(5 + Constants.edge_num)]

    server_index = []  # 获取可连接的服务器
    u2s_distance = []

    action_index = []
    resource_index = []
    for i in range(Constants.edge_num):
        u2s_dis = calcu_dis(user_coordinate[0], user_coordinate[1], Constants.server_coordinate[i][0], Constants.server_coordinate[i][1])
        if u2s_dis <= Constants.coverage_radius[i]:  # 满足距离需求
            server_index.append(i + 1)
            if u2s_dis < 0.00000001:
                u2s_dis = 0.00000001
            u2s_distance.append(u2s_dis)
    trans_time = [float("inf") for _ in range(Constants.edge_num)]
    for i in range(len(u2s_distance)):
        trans_rate = calcu_trans_rate(u2s_distance[i])
        trans_time[server_index[i] - 1] = user_data_size * 1024 / trans_rate
        min_required_resource = require_cpu / (tolerant_delay - trans_time[server_index[i] - 1])
        if tolerant_delay and remain_capacity[server_index[i] - 1] >= min_required_resource:
            action_index.append(server_index[i])
            resource_index.append(remain_capacity[server_index[i] - 1])

    if resource_index:
        if last_action in action_index:
            index = last_action
        else:
            index = action_index[np.argmax(resource_index)]
        # print("tran_time", trans_time[index-1])
        param = np.round((require_cpu / (tolerant_delay - trans_time[index - 1])) / remain_capacity[index - 1], 16)
    else:
        index = 0
        if tolerant_delay:
            param = 1
            # param = np.round((require_cpu / (tolerant_delay)) / Constants.user_capacity, 16)
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
    converg_cost = []
    converg_users = []
    start_time = time.time()

    f = open('log/VSVBP3.csv', 'w', encoding='utf-8', newline="")
    csv_write = csv.writer(f)
    csv_write.writerow(['Episode', 'total_r_avg', 'episode_reward','episode_cost', 'episode_user_num'])  # 表标题

    #这个文件用于存储实验结果的性能指标
    f2 = open('log/VSVBP_indicator3.csv', 'w', encoding='utf-8', newline="")
    csv_write2 = csv.writer(f2)
    csv_write2.writerow(['Episode', 'VSVBP_cost', 'VSVBP_users'])

    for i in range(Constants.episodes):
        print("epoch:", i)
        state = env.reset()
        action = VSVBP_choose_action(state, 0)
        last_action = action[0]
        print("action:", action[0])
        episode_reward = 0.
        episode_user_num = 0
        episode_cost = 0
        for j in range(max_steps):
            next_state, reward, cost, user_num, done, _ = env.step(action)
            episode_reward += reward  # calculate the episode reward
            episode_user_num += user_num
            episode_cost += cost
            next_action = VSVBP_choose_action(next_state, last_action)
            action = next_action
            state = next_state
            if done:
                break
        returns.append(episode_reward)
        total_reward += episode_reward

        if i >= 3000:
            converg_cost.append(episode_cost)  # 取2500次以后收敛的结果，
            converg_users.append(episode_user_num)
            if converg_users:
                csv_write2.writerow([i, sum(converg_cost) / sum(converg_users), sum(converg_users)])
        csv_write.writerow([i, total_reward / (i + 1), episode_reward, episode_cost, episode_user_num])

    f.close()
    f2.close()
    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)
    # print("Ave. cost = ", sum(converg_cost) / sum(converg_users))
    # print("success user number = ", sum(converg_users))