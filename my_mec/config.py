import numpy as np

class Constants:
    server_coordinate = np.array([[50, 50], [150, 50], [100, 100]], dtype=int)
    edge_num = len(server_coordinate)
    all_edge_capacity = np.array([4000 for _ in range(edge_num)]) # 4GHz
    total_task_num = 100  # 100
    penalty = 5

    # bandwidth = 5  # 10Mhz,Conversion of Units,1Khz=1000hz,1Mhz=1000KHz,
    bandwidth = 15 # 15
    user_capacity = 2000
    transmit_power = 0.5  # 500mV
    max_channel_power_gain = 1.02e-13
    user_switched_cap = 1e-27
    edge_switched_cap = 1e-29
    noise = 1e-13  # W
    # noise = -90
    coverage_radius = np.array([100 for _ in range(edge_num)])  # 100
    path_loss_const = 1e-4
    episodes = 1

    show_step_bool = True
    para_min = np.array([0 for i in range(edge_num+1)])     # 因此动作的参数是边缘服务器的数量加上本地计算，所以是edge_num +1
    para_max = np.array([1 for j in range(edge_num+1)])