import os
import click
import time
import gym
from common import ClickPythonLiteralOption
from common.platform_domain import PlatformFlattenedActionWrapper
import csv
import numpy as np

from common.wrappers import ScaledParameterisedActionWrapper


# def evaluate(env, agent, episodes=1000):
#     returns = []
#     timesteps = []
#     for _ in range(episodes):
#         state, _ = env.reset()
#         terminal = False
#         t = 0
#         total_reward = 0.
#         while not terminal:
#             t += 1
#             state = np.array(state, dtype=np.float32, copy=False)
#             act, act_param, all_action_parameters = agent.act(state)
#             action = pad_action(act, act_param)
#             (state, _), reward, terminal, _ = env.step(action)
#             total_reward += reward
#         timesteps.append(t)
#         returns.append(total_reward)
#     return np.array(returns)
from config import Constants


def filename_generator(path, type, seed, algorithm):
    return f"{path}{type}-{algorithm}-seed{seed}.png"


@click.command()
@click.option('--seed', default=2, help='Random seed.', type=int)
@click.option('--evaluation-episodes', default=0, help='Episodes over which to evaluate after training.', type=int)
@click.option('--episodes', default=5000, help='Number of epsiodes.', type=int)
@click.option('--batch-size', default=256, help='Minibatch size.', type=int)  # 32
@click.option('--gamma', default=0.95, help='Discount factor.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=2000, help='Number of transitions required to start learning.',
              type=int)  # may have been running with 500????????
@click.option('--use-ornstein-noise', default=True,
              help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
@click.option('--replay-memory-size', default=40000, help='Replay memory size in transitions.', type=int)
@click.option('--epsilon-steps', default=2000, help='Number of episodes over which to linearly anneal epsilon.',
              type=int)
@click.option('--epsilon-final', default=0.05, help='Final epsilon value.', type=float)  # 0.1
@click.option('--tau-actor', default=0.001, help='Soft target network update averaging factor.', type=float)  # 0.001
@click.option('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.',
              type=float)  # 0.001

@click.option('--learning-rate-actor', default=0.001, help="Actor network learning rate.", type=float)  # 0.001
@click.option('--learning-rate-actor-param', default=0.0001, help="Critic network learning rate.", type=float)  # 0.0001

@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
@click.option('--clip-grad', default=1., help="Parameter gradient clipping limit.", type=float)  #
@click.option('--split', default=False, help='Separate action-parameter inputs.', type=bool)
@click.option('--multipass', default=True, help='Separate action-parameter inputs using multiple Q-network passes.',
              type=bool)
@click.option('--indexed', default=False, help='Indexed loss function.', type=bool)
@click.option('--weighted', default=False, help='Naive weighted loss function.', type=bool)
@click.option('--average', default=False, help='Average weighted loss function.', type=bool)
@click.option('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
@click.option('--zero-index-gradients', default=False,
              help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.",
              type=bool)
@click.option('--action-input-layer', default=0, help='Which layer to input action parameters.', type=int)
@click.option('--layers', default='[256,128]', help='Duplicate action-parameter inputs.',
              cls=ClickPythonLiteralOption)  # 128,64
@click.option('--save-freq', default=0, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="results/exp", help='Output directory.', type=str)
@click.option('--render-freq', default=100, help='How often to render / save frames of an episode.', type=int)
@click.option('--save-frames', default=False,
              help="Save render frames from the environment. Incompatible with visualise.", type=bool)
@click.option('--visualise', default=False, help="Render game states. Incompatible with save-frames.", type=bool)
@click.option('--title', default="PDQN", help="Prefix of output files", type=str)
@click.option('--window', default=10, help='Window of reward')

def run(seed, episodes, evaluation_episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final, zero_index_gradients, initialise_params, scale_actions,
        clip_grad, split, indexed, layers, multipass, weighted, average, random_weighted, render_freq,
        save_freq, save_dir, save_frames, visualise, action_input_layer, title, window):
    env1 = gym.make('Mec-v0')
    print('F:all_edge_capacity',Constants.all_edge_capacity,'N:total_task_num',Constants.total_task_num,'B:bandwidth',Constants.bandwidth)
    print('2')
    def pad_action(act, act_param):
        params = [np.zeros((1,), dtype=np.float32) for _ in
                  range(env.num_actions)]  # params = [[0],[0],[0],[0],[0],[0]]
        params[act][:] = act_param
        return (act, params)

    print('3')
    initial_params_ = [0.5 for k in range(env1.action_space[0].n)]

    if scale_actions:
        for a in range(env1.action_space.spaces[0].n):
            initial_params_[a] = 2. * (initial_params_[a] - env1.action_space.spaces[1].spaces[a].low) / (
                    env1.action_space.spaces[1].spaces[a].high - env1.action_space.spaces[1].spaces[a].low) - 1.

    env = PlatformFlattenedActionWrapper(env1) 
    if scale_actions: 
        env = ScaledParameterisedActionWrapper(env)
    np.random.seed(seed)

    from agents.pdqn import PDQNAgent
    from agents.pdqn_split import SplitPDQNAgent
    from agents.pdqn_multipass import MultiPassPDQNAgent
    assert not (split and multipass)
    agent_class = PDQNAgent
    if split:
        agent_class = SplitPDQNAgent
    elif multipass:
        agent_class = MultiPassPDQNAgent

        # dqn
    agent = agent_class(
        env.observation_space.spaces[0], env.action_space,
        batch_size=batch_size,
        learning_rate_actor=learning_rate_actor,
        learning_rate_actor_param=learning_rate_actor_param,
        epsilon_steps=epsilon_steps,
        gamma=gamma,
        tau_actor=tau_actor,
        tau_actor_param=tau_actor_param,
        clip_grad=clip_grad,
        indexed=indexed,
        weighted=weighted,
        average=average,
        random_weighted=random_weighted,
        initial_memory_threshold=initial_memory_threshold,
        use_ornstein_noise=use_ornstein_noise,
        replay_memory_size=replay_memory_size,
        epsilon_final=epsilon_final,
        inverting_gradients=inverting_gradients,
        actor_kwargs={'hidden_layers': layers,
                      'action_input_layer': action_input_layer, },
        actor_param_kwargs={'hidden_layers': layers,
                            'squashing_function': False,
                            'output_layer_init_std': 0.0001, },
        zero_index_gradients=zero_index_gradients,
        seed=seed, )

    # from agents.sarsa_lambda import SarsaLambdaAgent
    # assert not (split and multipass)
    # agent_class = SarsaLambdaAgent

    # DDPG
    # from agents.paddpg import PADDPGAgent
    # from agents.pdqn_split import SplitPDQNAgent
    # from agents.pdqn_multipass import MultiPassPDQNAgent
    # assert not (split and multipass)
    # agent_class = PADDPGAgent
    # ddpg
    """
    agent = agent_class(
        env.observation_space.spaces[0], env.action_space,
        batch_size=batch_size,
        learning_rate_actor=learning_rate_actor,
        learning_rate_actor_param=learning_rate_actor_param,
        epsilon_steps=epsilon_steps,
        gamma=gamma,
        tau_actor=tau_actor,
        tau_actor_param=tau_actor_param,
        clip_grad=clip_grad,

        initial_memory_threshold=initial_memory_threshold,
        use_ornstein_noise=use_ornstein_noise,
        replay_memory_size=replay_memory_size,
        epsilon_final=epsilon_final,
        inverting_gradients=inverting_gradients,
        actor_kwargs={'hidden_layers': layers,
                      'action_input_layer': action_input_layer, },
        actor_param_kwargs={'hidden_layers': layers,
                            'squashing_function': False,
                            'output_layer_init_std': 0.0001, },
        # indexed=indexed,
        # weighted=weighted,
        # average=average,
        # random_weighted=random_weighted,
        # zero_index_gradients=zero_index_gradients,
        seed=seed,
    )"""

    if initialise_params:
        initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.spaces[0].shape[0]))
        initial_bias = np.zeros(env.action_space.spaces[0].n)
        # print("权重和偏差：", initial_weights, initial_bias)
        for a in range(env.action_space.spaces[0].n):
            initial_bias[a] = initial_params_[a]
        agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)
    # print("agent:", agent)
    max_steps = 5000
    total_reward = 0.
    returns = []
    start_time = time.time()
    best = -float("inf")

    f = open('./pdqn-toler-time-4-6.csv', 'w', encoding='utf-8', newline="")
    csv_write = csv.writer(f)
    csv_write.writerow(['Episode', 'total_r_avg', 'episode_reward', 'episode_cost', 'episode_user_num', 'episode_t'])

    for i in range(episodes):
        print("----------episodes i----------", i)
        state = env.reset()
        print('run_state',state)
        state = np.array(state, dtype=np.float32, copy=False)  # 此时的状态是float类型的
        act, act_param, all_action_parameters = agent.act(state)
        action = pad_action(act, act_param)
        print('action:',action)
        episode_reward = 0.
        episode_user_num = 0
        episode_cost = 0
        episode_t_exe = 0
        agent.start_episode()
        for j in range(max_steps):
            #ddpg运行这一段
            ##ret = env.step(action)  # execute action in environment, and observe next state
            #print("----------episodes i----------", i)
            #print('----------开始运行 j------------------',j)
            #obs, reward, terminal, info, info2 = env.step(action)  # obtain result
            #cost, user_num, t_exe = info['cost'], info['user_num'], info['t_exe']
            ## cost, user_num, t_exe = info[0], info[1], info[2]
            #next_state = obs[0]
            ## print('next_state, reward, terminal, info, cost, user_num, t_exe:',next_state, reward, terminal, info, cost, user_num, t_exe)
            #next_state = np.array(next_state, dtype=np.float32, copy=False)  # convert to nparray
            ## todo
            #next_act, next_act_param, next_all_action_parameters = agent.act(next_state)  # choose action according to next state , server_available
            #next_action = pad_action(next_act, next_act_param)  # package action and param

            #agent.step(state, (act, all_action_parameters), reward, next_state,  # add sample and learn
                       #(next_act, next_all_action_parameters), terminal, time_steps=1)
            #act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters  # transfer state and action
            #action = next_action
            #state = next_state

            # dqn运行这一段
            obs, reward, terminal, info, info2 = env.step(action)  # obtain result
            cost, user_num, t_exe = info['cost'], info['user_num'], info['t_exe']
            ##cost, user_num, t_exe = info[0], info[1], info[2]
            next_state = obs[0]
            ## print('next_state, reward, terminal, info, cost, user_num, t_exe:',next_state, reward, terminal, info, cost, user_num, t_exe)
            next_state = np.array(next_state, dtype=np.float32, copy=False)  # convert to nparray
            # todo
            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)  # choose action according to next state , server_available
            next_action = pad_action(next_act, next_act_param)  # package action and param
            agent.step(state, (act, all_action_parameters), reward, next_state,(next_act, next_all_action_parameters), terminal, time_steps = 1)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters  # transfer state and action
            action = next_action
            state = next_state

            episode_reward += reward  # calculate the episode reward
            episode_user_num += user_num 
            episode_cost += cost 
            episode_t_exe += t_exe

            if terminal:
                break
        agent.end_episode()

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
    print("+++++++++下一个循环++++++++++++")


if __name__ == '__main__':
    print('11111')
    run()
