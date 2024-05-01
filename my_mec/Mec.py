import gym
from gym import spaces
import copy
from tools import *
import random
import numpy as np
from gym.envs.registration import register
'''
    STATE:
    0. user data size
    1. required capacity
    2. delay tolerant
    3. user x
    4. user y
    5. remain capacity in edge
    6. release resource
   
'''
# episode = 0

class MecEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed=2):    
        self.seed = seed
        np.random.seed(self.seed)
        self.state = []
        self.local_time = 0 
        # self.local_use_record = []
        self.edge_capacity = Constants.all_edge_capacity
        self.remain_capacity = self.edge_capacity
        self.resource_use_record = [[] for _ in range(Constants.edge_num)] 
        self.released_resource = np.array([0 for _ in range(Constants.edge_num)])
        self.each_server_has_allocated = np.array([0 for _ in range(Constants.edge_num)])
        self.task_counter = 0
        num_actions = Constants.edge_num + 1  
        self.done = False

        self.action_space = spaces.Tuple((
            spaces.Discrete(num_actions),
            spaces.Tuple(
                tuple(spaces.Box(low=np.array([Constants.para_min[i]]), high=np.array([Constants.para_max[i]]), dtype=np.float32)
                      for i in range(num_actions))
            )
        ))
        print("动作的个数:", self.action_space[0].n)

        self.observation_space = spaces.Tuple((
            spaces.Box(low=0., high=1., shape=(len(self.get_state()),), dtype=np.float32),   
            spaces.Discrete(200),
        ))
        print("状态的维度:", self.observation_space[0].shape[0])

    def reset(self):
        self.task_counter = 0
        self.local_time = 0
        self.users_record = [[] for _ in range(Constants.edge_num)]
        self.remain_capacity = Constants.all_edge_capacity
        self.done = False 
        self.resource_use_record = [[] for _ in range(Constants.edge_num)]
        self.released_resource = np.array([0 for _ in range(Constants.edge_num)]) 
        self.each_server_has_allocated = np.array([0 for _ in range(Constants.edge_num)])   

        state = self.get_state().copy()
        return copy.deepcopy(state)

    def get_state(self):
        if self.task_counter == Constants.total_task_num:
            self.done = True
            user_info = np.array([0, 0, 0])
            user_location = np.array([0, 0], dtype=int)
            self.state = np.concatenate([user_info, user_location, self.remain_capacity, self.released_resource])
        else:
            user_info = self.task_generator()
            # rnd = random.choice([1, 2])
            user_x, user_y = np.random.uniform(50, 150), np.random.uniform(50, 150)
            user_location = np.array([user_x, user_y], dtype=int)
            self.state = np.concatenate([user_info, user_location, self.remain_capacity, self.released_resource])
        state = self.state.copy()
        return state

    def task_generator(self):
        user_data_size = np.random.uniform(6000, 8000)
        #user_data_size = np.random.uniform(8000, 10000)  # kbits, * 1e3 (300, 500)  (8000, 10000)
        required_cpu = np.random.uniform(7000, 10000)  # M, * 1e6 (900, 1100)    (6500, 8000)
        tolerant_delay = int(np.random.randint(3,5))  # np.random.uniform(1, 1.1) 
        return np.array([user_data_size, required_cpu, tolerant_delay])

    def step(self, action): 
        """
            action = (act_index, [param1, param2])
        """
  
        state = copy.deepcopy(self.state)

        data_size = state[0]
        require_cpu = state[1]
        tolerant_delay = state[2]
        user_coordinate = (state[3], state[4])
        remain_capacity = state[5:(5+Constants.edge_num)]

        flag = type(action)

        if flag is np.int64 or int:
            act_index = action
            act_param = 1


        each_server_has_allocated = np.array([0 for _ in range(Constants.edge_num)])

        success_user = 0
        cost = 0
        t_exe = 0
        if act_index != 0:
            alloc_resource = np.ceil(remain_capacity[act_index - 1] * act_param)
            u2s_dis = calcu_dis(user_coordinate[0], user_coordinate[1], Constants.server_coordinate[act_index-1][0],
                                Constants.server_coordinate[act_index-1][1])
            if u2s_dis > Constants.coverage_radius[act_index-1]:
                reward = -Constants.penalty ** 3
                print("not in the coverage")
            else:
                total_time, offload_energy = offload_time_energy(u2s_dis, data_size, require_cpu, alloc_resource)
                print("total_time:", total_time)

                if total_time <= tolerant_delay:
                    t_exe = total_time
                    reward = (5-offload_energy)*5+20  
                    success_user = 1
                    cost = offload_energy
                    print("off_Energy", offload_energy)
                    time_e = int(np.ceil(total_time))
                    self.resource_use_record[act_index-1].append([alloc_resource, time_e])
                    self.users_record[act_index - 1].append(user_coordinate + (act_index,))
                    # print("users records:", self.users_record)
                    each_server_has_allocated[act_index - 1] = alloc_resource
                    self.each_server_has_allocated += each_server_has_allocated
                    remain_capacity[act_index-1] -= alloc_resource
                else:
                    reward = -Constants.penalty ** 3 
        else:
            u2s_dis = 0
            alloc_resource = (Constants.user_capacity * act_param)
            local_time, local_energy = local_compute(require_cpu, alloc_resource)
            print("local_time:", local_time)
            if local_time <= tolerant_delay:
                self.local_time = int(np.ceil(local_time))
                # self.local_use_record.append([alloc_resource, int(np.ceil(local_time))])
                print("local energy", local_energy)
                t_exe = local_time
                cost = local_energy
                success_user = 1
                reward = -local_energy+20  # + 55
            else:
                reward = -Constants.penalty ** 3

        if act_index:
            print(f"dis: {u2s_dis}, TYPE: {act_index},  PARAM: {act_param},  allocate_resource: {act_param, alloc_resource}, reward: {reward}, sever_xy:{Constants.server_coordinate[act_index-1]} >>>>")
        else:
            print(
                f"dis: {u2s_dis}, TYPE: {act_index},  PARAM: {act_param},  allocate_resource: {act_param, alloc_resource}, reward: {reward}, sever_xy:{None} >>>>")
        # print("\n")
        if Constants.show_step_bool:
            print(f"STEP: {self.task_counter}, Req: {data_size, require_cpu, tolerant_delay,user_coordinate}")
            print(f"CAP: {self.remain_capacity}, Released: {self.released_resource}, has_allocated: {self.each_server_has_allocated}, resource_use_record: {self.resource_use_record}")
            print("\n")
     
        self.released_resource = self.update_release_record()
        remain_capacity += self.released_resource
        self.remain_capacity = remain_capacity
        self.task_counter += 1
        next_state = self.get_state().copy()

        info = {}
        return next_state, reward, cost, success_user, t_exe, self.done, info

    def update_release_record(self):
        released_resource = [0 for _ in range(Constants.edge_num)]
        records = copy.deepcopy(self.resource_use_record)
        each_sever_has_allocated = np.array([0 for _ in range(Constants.edge_num)])     # , dtype=int

        for server in range(len(records)-1,-1,-1):
            for item in range(len(records[server])-1,-1,-1):
                if records[server]:
                    records[server][item][1] -= 1
                    if records[server][item][1] == 0:
                        released_resource[server] += records[server][item][0]
                        each_sever_has_allocated[server] -= records[server][item][0]
                        del self.users_record[server][0]   
                        del (records[server][item])     

        self.each_server_has_allocated += each_sever_has_allocated
        self.resource_use_record = copy.deepcopy(records)
        self.local_time -= 1
        return released_resource
    #
