"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""

import math
import random

import gym
import utils.CTM.ctm_start as cs
from utils.CTM.utils import calculate_fire_risk, load_all_firebytime


class DynamicSignalEnv(gym.Env):
    environment_name = "dynamic signal"
    def __init__(self, args):
        self.args = args
        self.baseGraph = None
        self.reward_threshold = -1000
        self.actions_record_for_agant = []

        self.fire_current = []
        self.state4marl = None
        self.obs4marl = None
        self.static_field = None
        self.fire_levels = None
        self.congestion_levels = None
        self.step_cell_people_count = []
        self.reset(load_all_firebytime())

    def reset(self, fire_info=None):
        random.seed(None)
        self.rannum = random.random()
        self.step_count = 0
        self.done = False

        self.all_firebytime = fire_info
        self.baseGraph = cs.init(self.all_firebytime)
        self.episode_time_slice = []
        self.action_space = [0, 1, 2, 3]

        self.state4marl, step0_count = self._get_state_list()
        self.step_cell_people_count.append(step0_count)
        self.obs4marl = self._get_obs_list()

    def _get_state_list(self):
        all_nodesinfo = self.baseGraph.nodesinfo
        all_people_number = 0
        cell_people_count = []
        static_field = []
        fire_levels = []
        congestion_levels = []
        for i in all_nodesinfo.values():
            N = math.ceil(i.current_number)
            all_people_number = all_people_number + N
            cell_people_count.append(N)
            static_field.append(N * (round(i.energy_domine / 22, 3)))

            _, fire_level = calculate_fire_risk(
                i.current_fireinfo[0],
                i.current_fireinfo[1],
                i.current_fireinfo[2],
                i.current_fireinfo[3],
            )
            fire_levels.append(fire_level * N)

            congestion_level = 0
            if i.current_density > 2.5:
                congestion_level = round((i.current_density - 2.5) / (6 - 2.5), 3)
            congestion_levels.append(N * congestion_level)
        self.current_people_number = all_people_number
        self.static_field = static_field
        self.fire_levels = fire_levels
        self.congestion_levels = congestion_levels
        cc = [
            x + y + z for x, y, z in zip(static_field, congestion_levels, fire_levels)
        ]
        return [all_people_number] + cc[0:242], cell_people_count  #

    def _get_obs_list(self):

        all_nodesinfo = self.baseGraph.nodesinfo

        fire_levels = []
        congestion_levels = []
        static_fields = []

        for i in range(len(all_nodesinfo)):
            node = all_nodesinfo[i + 1]
            _, fire_level = calculate_fire_risk(
                node.current_fireinfo[0],
                node.current_fireinfo[1],
                node.current_fireinfo[2],
                node.current_fireinfo[3],
            )
            fire_levels.append(fire_level)
            congestion_levels.append(round(node.current_number))
            static_fields.append(node.energy_domine)
        return [
            static_fields[0:242] + congestion_levels[0:242] + fire_levels[0:242]
            for _ in self.baseGraph.agent_cell_ids
        ]  #

    def step(self, actions):

        self.step_count += 1
        self.baseGraph.from_actions_get_groupId_submatrix(actions)
        self.baseGraph = cs.start_Sub_CTM(self.baseGraph, self.step_count)

        if self.current_people_number == 0 and self.step_count > 20:
            self.done = True

        self.state4marl, stepn_count = self._get_state_list()
        self.step_cell_people_count.append(stepn_count)
        self.obs4marl = self._get_obs_list()

        reward = (
            -sum(self.static_field)
            - sum(self.fire_levels)
            - sum(self.congestion_levels)
        )

        return (
            reward,
            self.done,
            [
                sum(self.static_field),
                sum(self.fire_levels),
                sum(self.congestion_levels),
            ],
        )

    def save_env_info(self, actions):
        self.actions_record_for_agant.append(actions)

    def get_state(self):
        assert self.state4marl is not None
        return self.state4marl

    def get_obs(self):
        return [
            self.get_obs_agent(i) for i in range(len(self.baseGraph.agent_cell_ids))
        ]

    def get_obs_agent(self, agent_id):
        return self.obs4marl[agent_id]

    def get_env_info(self):
        return {
            "n_actions": len(self.action_space),
            "n_agents": len(self.baseGraph.agent_cell_ids),
            "state_shape": len(self.get_state()),
            "obs_shape": len(self.get_obs()[0]),
            "episode_limit": 150,
        }

    def get_avail_agent_actions(self, agent_id):
        available_action = [0 for x in range(len(self.action_space))]
        for i in self.baseGraph.signal_available_direction[agent_id]:
            available_action[i - 1] = 1
        return available_action
