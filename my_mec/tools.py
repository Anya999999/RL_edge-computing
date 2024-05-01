import math
from config import *

def calcu_dis(u_x, u_y, s_x, s_y):
    dis = math.sqrt((u_x - s_x) ** 2 + (u_y - s_y) ** 2)
    return dis

def offload_time_energy(distance_u2s, data_size, require_cpu, alloc_resource):
    trans_rate = calcu_trans_rate(distance_u2s)
    trans_time = data_size * 1024 / trans_rate
    print("trans_time:",trans_time)
    # print("trans rate-------", trans_rate)
    if alloc_resource:
        compute_time = require_cpu / alloc_resource
    else:
        compute_time = float("inf")
    total_time = trans_time + compute_time
    edge_energy = Constants.edge_switched_cap * require_cpu * pow(10, 6) * ((alloc_resource * pow(10, 6)) ** 2) + \
                  Constants.transmit_power * trans_time
    # _, full_local_energy = local_compute(require_cpu, Constants.user_capacity)
    # save_energy = (full_local_energy - edge_energy) / full_local_energy
    return np.round(total_time, 3), np.round(edge_energy, 3)  #, np.round(save_energy, 3)

def local_compute(require_cpu, alloca_resource):
    if alloca_resource > 0 and alloca_resource <= Constants.user_capacity:
        local_time = require_cpu / alloca_resource
    elif alloca_resource > Constants.user_capacity:
        local_time = require_cpu / Constants.user_capacity
    else:
        local_time = float("inf")
    local_energy = Constants.user_switched_cap * require_cpu * pow(10, 6) * ((alloca_resource * pow(10, 6)) ** 2)
    return local_time, local_energy

def calcu_trans_rate(u2s_dis):
    # trans_rate = 1000 * 1000 * Constants.bandwidth * np.log2(
    #     1 + Constants.transmit_power * Constants.path_loss_const * pow(u2s_dis, -4) / Constants.noise)
    trans_rate = 1000 * 1000 * Constants.bandwidth * np.log2(
        1 + Constants.transmit_power * Constants.max_channel_power_gain / Constants.noise)
    return trans_rate

def edge_energy(trans_time, require_cpu, alloc_resource):
    edge_energy = Constants.transmit_power * trans_time + Constants.edge_switched_cap * require_cpu * pow(10, 6) * \
                  ((alloc_resource * pow(10, 6)) ** 2)
    return np.round(edge_energy, 3)