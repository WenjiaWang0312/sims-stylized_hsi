import os
import torch
import numpy as np
import json
import datetime
import time
import yaml
import copy
from gym import spaces

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd

import sims.learning.common_player as common_player
import sims.learning.amp_models as amp_models
import sims.learning.amp_network_builder as amp_network_builder
from rl_games.common.player import BasePlayer


class TaskControllerPlayer(common_player.CommonPlayer):
    def __init__(self, config, obs_shape):
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._disc_reward_scale = config['disc_reward_scale']
        
        # super().__init__(config)
        BasePlayer.__init__(self, config)
        self.network = config['network']
        
        self._setup_action_space()
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        
        net_config = self._build_net_config()
        net_config['input_shape'] = (obs_shape,)
        self.obs_shape = (obs_shape, )
        self._build_net(net_config)   


        return

    def restore(self, fn):
        if (fn != 'Base'):
            super().restore(fn)
            if self._normalize_amp_input:
                checkpoint = torch_ext.load_checkpoint(fn)
                self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
        return
    
    def _build_net(self, config):
        super()._build_net(config)
        
        # if self._normalize_amp_input:
        #     self._amp_input_mean_std = RunningMeanStd((self._amp_observation_space[0] // self.env.task._num_amp_obs_steps,)).to(self.device)
        #     self._amp_input_mean_std.eval()  
        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()  
        
        return
    # def _build_net(self, config):
    #     super()._build_net(config)
        
    #     if self._normalize_amp_input:
    #         self._amp_input_mean_std = RunningMeanStd((self._amp_observation_space[0] // self._amp_obs_steps,)).to(self.device)
    #         self._amp_input_mean_std.eval()  
        
    #     return

    def _post_step(self, info):
        super()._post_step(info)
        if (self.env.task.viewer):
            self._amp_debug(info)
        return
    
    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['amp_input_shape'] = self.env.amp_observation_space.shape
        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']
        return config
    # def _build_net_config(self):
    #     config = super()._build_net_config()
    #     if hasattr(self, 'env'):
    #         config['amp_input_shape'] = self.env.amp_observation_space.shape
    #         config['amp_obs_steps'] = self.env.task._num_amp_obs_steps
    #     else:
    #         config['amp_input_shape'] = self.env_info['amp_observation_space']
    #         config['amp_obs_steps'] = self.env_info['num_amp_obs_steps']

    #     self._amp_observation_space = config['amp_input_shape']
    #     self._amp_obs_steps = config['amp_obs_steps']
    #     return config

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward)

        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits)) 
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale
        return disc_r


class AMPMultiPlayerContinuous(common_player.CommonPlayer):
    def __init__(self, config):
        with open(config['llc_config'], 'r') as f:
            llc_config = yaml.load(f, Loader=yaml.SafeLoader)
            llc_config_params = llc_config['params']

        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._disc_reward_scale = config['disc_reward_scale']
        
        super().__init__(config)
        checkpoints = config['checkpoints']
        self._tasks = list(checkpoints.keys())
        self._agent_dict = dict()
        for task in self._tasks:
            llc_checkpoint = checkpoints[task]
            assert(llc_checkpoint != "")
            
            obs_shape = self.env.task.get_obs_size() + self.env.task.get_task_obs_size(task)
            self._agent_dict[task] = self._build_llc(llc_config_params, llc_checkpoint, obs_shape, task)
        return
    
    def _build_llc(self, config_params, checkpoint_file, obs_shape, task):
        config_params = copy.deepcopy(config_params)
        network_params = config_params['network']
        network_builder = amp_network_builder.AMPBuilder()
        network_builder.load(network_params)

        network = amp_models.ModelAMPContinuous(network_builder)
        llc_agent_config = self._build_llc_agent_config(config_params, network, task)

        _llc_agent = TaskControllerPlayer(llc_agent_config, obs_shape)
        _llc_agent.restore(checkpoint_file)
        print("Loaded LLC checkpoint from {:s}".format(checkpoint_file))
        return _llc_agent
    
    def _build_llc_agent_config(self, config_params, network, task):

        llc_env_info = copy.deepcopy(self.env_info)
        obs_space = llc_env_info['observation_space']
        obs_size = obs_space.shape[0]
        llc_env_info['observation_space'] = spaces.Box(obs_space.low[:obs_size], obs_space.high[:obs_size])
        # llc_env_info['amp_observation_space'] = self.env.amp_observation_space.shape
        llc_env_info['amp_observation_space'] = (self.env.task.get_amp_obs_size(task) * self.env.task._num_amp_obs_steps,)
        llc_env_info['num_amp_obs_steps'] = self.env.task._num_amp_obs_steps
        llc_env_info['num_envs'] = self.env.task.num_envs

        config = config_params['config']
        config['network'] = network
        config['env_info'] = llc_env_info

        return config

    def restore(self, fn):
        if fn != 'Base':
            super().restore(fn)
            if self._normalize_amp_input:
                checkpoint = torch_ext.load_checkpoint(fn)
                self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
            self.checkpoint_fn = fn
        return
    
    def _build_net(self, config):
        super()._build_net(config)
        
        # if self._normalize_amp_input:
        #     self._amp_input_mean_std = RunningMeanStd((self._amp_observation_space[0] // self.env.task._num_amp_obs_steps,)).to(self.device)
        #     self._amp_input_mean_std.eval()  
        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()  
        
        return

    def _post_step(self, info):
        super()._post_step(info)
        if self.env.task.viewer:
            self._amp_debug(info)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['amp_input_shape'] = self.env.amp_observation_space.shape
        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']
        return config
    
    # def _build_net_config(self):
    #     config = super()._build_net_config()
    #     if hasattr(self, 'env'):
    #         config['amp_input_shape'] = self.env.amp_observation_space.shape
    #         config['amp_obs_steps'] = self.env.task._num_amp_obs_steps
    #     else:
    #         config['amp_input_shape'] = self.env_info['amp_observation_space']
    #         config['amp_obs_steps'] = self.env_info['num_amp_obs_steps']

    #     self._amp_observation_space = config['amp_input_shape']
    #     return config

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            num_env_print = min(2, amp_obs.shape[0])
            amp_obs = amp_obs[0:num_env_print]
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs)
            disc_reward = amp_rewards['disc_rewards']
            for i in range(num_env_print):
                disc_pred_curr = disc_pred.detach().cpu().numpy()[i, 0]
                disc_reward_curr = disc_reward.cpu().numpy()[i, 0]
                print("env: {} disc_pred: {} disc_reward: {}".format(i, disc_pred_curr, disc_reward_curr))

        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            shape = amp_obs.shape
            amp_obs = amp_obs.view(-1, self.env.amp_observation_space.shape[0] // self.env.task._num_amp_obs_steps)
            amp_obs = self._amp_input_mean_std(amp_obs)
            amp_obs = amp_obs.view(shape)
        return amp_obs

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits)) 
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale
        return disc_r
    
    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        n_games = n_games * n_game_life
        games_played = 0

        num_trials = self.env.num_envs
        res = torch.zeros((num_trials, self.env.task.max_episode_length, self.env.task.num_bodies, 7), dtype=torch.float32, device=self.device)
        res[..., -1] = 1 # Set quaternion to 0 0 0 1
        box_states = torch.zeros((num_trials, self.env.task.max_episode_length, 13), dtype=torch.float32, device=self.device)
        has_collected = torch.zeros(num_trials, dtype=torch.long, device=self.device)
        task_ids = torch.zeros(num_trials, self.env.task.max_episode_length, dtype=torch.long, device=self.device)

        for _ in range(n_games):
            if torch.sum(has_collected) == num_trials:
                break
            
            if games_played >= n_games:
                break

            obs_dict = self.env_reset()
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)
            for task in self._tasks:
                batch_size = self._agent_dict[task].get_batch_size(obs_dict['obs'], batch_size) # llc_agent batch_size

            mask = (has_collected == 0)
            res[mask, self.env.task.progress_buf] = self.env.task._fetch_humanoid_full_states()[mask] # First frame
            task_ids[mask, self.env.task.progress_buf] = self.env.task._fetch_every_env_task()[mask]
            box_states[mask, self.env.task.progress_buf] = self.env.task._box_states[mask]

            done_indices = []

            for n in range(self.max_steps):

                obs_dict = self.env_reset(done_indices)

                action = torch.zeros((self.env.task.num_envs, self.env.task.num_actions), dtype=torch.float32, device=self.device)

                every_env_task = self.env.task._fetch_every_env_task()
                # sit_task_mask = (every_env_task == self.env.task._SIT_TASK_FLAG)
                # getup_task_mask = (every_env_task == self.env.task._GETUP_TASK_FLAG)
                # sit_task_indices = sit_task_mask.nonzero(as_tuple=False)[:, 0]
                # getup_task_indices = getup_task_mask.nonzero(as_tuple=False)[:, 0]

                # if len(sit_task_indices) > 0:
                #     sit_obs = {k: v[sit_task_indices] for k, v in obs_dict.items()}
                #     sit_action = self._llc_agent.get_action(sit_obs, is_determenistic)
                #     action[sit_task_indices] = sit_action
                # if len(getup_task_indices) > 0:
                #     getup_obs = {k: v[getup_task_indices] for k, v in obs_dict.items()}
                #     getup_action = self.get_action(getup_obs, is_determenistic)
                #     action[getup_task_indices] = getup_action
                for task in self._tasks: # walk, sit, lie, touch, getup, standup
                    task_mask = (every_env_task == self.env.task.skill_flags[task])
                    curr_task_indices = task_mask.nonzero(as_tuple=False)[:, 0]
                    if len(curr_task_indices) > 0:
                        task_obs = {k: v[curr_task_indices] for k, v in obs_dict.items()}
                        # from IPython import embed; embed()
                        task_action = self._agent_dict[task].get_action(task_obs, is_determenistic)
                        action[curr_task_indices] = task_action
                        
                obs_dict, r, done, info =  self.env_step(self.env, action) 
                # step
                    # pre_physics_step
                        # self._update_action()
                        # self._update_action_mask()
                        # self._update_observation()
                        # self._update_reward()
                    #
                    #
                # Start 4096 envs and collect results from these 4096 envs
                mask = (has_collected == 0)
                res[mask, self.env.task.progress_buf[mask]] = self.env.task._fetch_humanoid_full_states()[mask]
                task_ids[mask, self.env.task.progress_buf[mask]] = every_env_task[mask]

                box_states[mask, self.env.task.progress_buf[mask]] = self.env.task._box_states[mask]

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    has_collected[done_indices] = 1
                    if torch.sum(has_collected) == num_trials:
                        break
                        

                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break
                
                done_indices = done_indices[:, 0]

        # # save data
        import numpy as np
        if 'record_name' in self.config:
            record_name = self.config['record_name']
            os.makedirs('render', exist_ok=True)
            np.savez(f"render/{record_name}.npz", global_translation_of_bodies=res[..., 0:3].cpu().numpy(), global_rotation_of_bodies=res[..., 3:7].cpu().numpy(), box_scale=self.env.task._box_size.cpu().numpy().tolist(), box_states=box_states.cpu().numpy(), task_ids=task_ids.cpu().numpy())
            print(f"record {record_name} saved")
            
