import json
import os
import re

import numpy as np
import pickle
from env.environment import DynamicSignalEnv
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from MARL.runner import Runner
from MARL.common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, \
    get_reinforce_args, \
    get_commnet_args, get_g2anet_args
import multiprocessing

import matplotlib.pyplot as plt

def marl_agent_wrapper():
    args = get_common_args()

    if args.alg.find('coma') > -1: 
        args = get_coma_args(args)
    elif args.alg.find('central_v') > -1:
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    else:
        args = get_mixer_args(args)
    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)

    env = DynamicSignalEnv(args)
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"] #4
    args.n_agents = env_info["n_agents"] #19
    args.state_shape = env_info["state_shape"] #143
    args.obs_shape = env_info["obs_shape"] #25
    args.episode_limit = env_info["episode_limit"]
    print("\n是否加载模型（测试必须）：", args.load_model, "是否打印中间变量：", args.havelook, "是否训练：",args.learn)

    save_path = args.result_dir + '/' + args.alg 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path_dir = get_next_folder(save_path, args.map + "_")
    train_save_path = save_path + "/" + save_path_dir
    train_save_path_model = train_save_path + '/model'
    train_save_path_actions = train_save_path + '/historydata/actions_result'
    train_save_path_data = train_save_path + '/historydata/train_process'
    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
    if not os.path.exists(train_save_path_model):
        os.makedirs(train_save_path_model)
    if not os.path.exists(train_save_path_data):
        os.makedirs(train_save_path_data)
    if not os.path.exists(train_save_path_actions):
        os.makedirs(train_save_path_actions)
    args.save_path = train_save_path
    parameters = vars(args)
    filename = train_save_path +'/training_parameters.json'
    with open(filename, 'w') as file:
        json.dump(parameters, file, indent=4)

    runner = Runner(env, args)

    if args.learn:
        runner.run(0) 
        with open(train_save_path+"/historydata/loss.txt", 'r') as file:
            lines = file.readlines()
        loss_values = [float(loss.split('(')[1].split(',')[0]) for loss in lines]
        plt.figure(figsize=(10, 5))
        plt.plot(loss_values, marker='o', linestyle='-', color='b', markerfacecolor='black', markersize=1)
        plt.title('Loss per Time Step')
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(train_save_path+"/historydata/loss.png", dpi=300)
    else:
        _, reward = runner.evaluate(10101)
        print('The ave_reward of {} is  {}'.format(args.alg, reward))

def get_next_folder(base_path, folder_prefix):
    dirs = os.listdir(base_path)
    pattern = re.compile(f"^{folder_prefix}(\\d+)$")
    max_num = 0
    for dir in dirs:
        match = pattern.match(dir)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return f"{folder_prefix}{max_num + 1}"


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    marl_agent_wrapper()
