# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch 
from rl_games.algos_torch import torch_ext
import datetime
import os
import json
from rl_games.algos_torch.running_mean_std import RunningMeanStd
import sims.learning.common_player as common_player


class AMPPlayerContinuous(common_player.CommonPlayer):
    def __init__(self, config):
        self._eval = config["eval"]
        self.record = config["record"] # record simulated states
        
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._disc_reward_scale = config['disc_reward_scale']
        record_name = config['record_name']
        self.record = record_name != ''
        
        super().__init__(config)
        return

    def restore(self, fn):
        if (fn != 'Base'):
            self._checkpoint_fn = fn
            super().restore(fn)
            if self._normalize_amp_input:
                checkpoint = torch_ext.load_checkpoint(fn)
                self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
        return
    
    def _build_net(self, config):
        super()._build_net(config)
        
        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()  
        
        return

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

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            # print("disc_pred: ", disc_pred, disc_reward)

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


    def run_test(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()
        
        if hasattr(self.env.task, "_collecting_sit_states"):
            if self.env.task._collecting_sit_states:
                self.init_poses()

                # self.env.task.cfg["env"]["enableTrackInitState"] = True
        
        num_collect = 20
        if self.record:
            num_trials = self.env.num_envs
            human_states = torch.zeros((num_trials, self.env.task.max_episode_length*num_collect, 15, 7), dtype=torch.float32, device=self.device)
            trial_ids = torch.zeros((self.env.task.max_episode_length*num_collect), dtype=torch.long, device=self.device) -1
            # human_dof_pos = torch.zeros((num_trials, self.env.task.max_episode_length, 28), dtype=torch.float32, device=self.device)
            # object_states = torch.zeros((num_trials, self.env.task.max_episode_length, 7), dtype=torch.float32, device=self.device)
            # human_states[..., -1] = 1.0 # Set quaternion to 0 0 0 1
            # object_states[..., -1] = 1.0
            has_collected = torch.zeros(num_trials, dtype=torch.long, device=self.device)
            box_states = torch.zeros((num_trials, self.env.task.max_episode_length*num_collect, 13), dtype=torch.float32, device=self.device)
        need_init_rnn = self.is_rnn
        captions = []
        for _ in range(n_games):

            # if self.record:
                # if torch.sum(has_collected) == num_trials:
                #     print("162")
                #     break

            obs_dict = self.env_reset()
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

            if self.record:
                mask = (has_collected == 0)
                # human_states[mask, self.env.task.progress_buf] = self.env.task._fetch_humanoid_full_states()[mask] # First frame
                
                # human_states[:, self.env.task.global_progress] = self.env.task._fetch_humanoid_full_states()
                # human_dof_pos[mask, self.env.task.progress_buf] = self.env.task._fetch_humanoid_dof_pos()[mask]
                # object_states[mask, self.env.task.progress_buf] = self.env.task._fetch_object_full_states()[mask]

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            if self.env.task.global_progress >= self.env.task.max_episode_length * num_collect-1:
                break
            for n in range(self.max_steps):
                if self.env.task.global_progress >= self.env.task.max_episode_length * num_collect-1:
                    break
                obs_dict = self.env_reset(done_indices)

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_determenistic)
                else:
                    action = self.get_action(obs_dict, is_determenistic)
                obs_dict, r, done, info =  self.env_step(self.env, action)

                
                if self.record:
                    # Start 4096 envs and collect results from these 4096 envs
                    mask = (has_collected == 0)
                    # human_states[mask, self.env.task.progress_buf[mask]] = self.env.task._fetch_humanoid_full_states()[mask]
                    human_states[:, self.env.task.global_progress] = self.env.task._fetch_humanoid_full_states()
                    trial_ids[self.env.task.global_progress] = self.env.task.num_resets - 1
                    if hasattr(self.env.task, "captions") and len(self.env.task.captions) > 0:
                        captions.append(self.env.task.captions[0])
                    # human_dof_pos[mask, self.env.task.progress_buf[mask]] = self.env.task._fetch_humanoid_dof_pos()[mask]
                    # object_states[mask, self.env.task.progress_buf[mask]] = self.env.task._fetch_object_full_states()[mask]
                    if hasattr(self.env.task, "_box_states") and self.env.task._box_states is not None:
                        box_states[:, self.env.task.global_progress] = self.env.task._box_states

                cr += r
                steps += 1
  
                self._post_step(info)

                if render:
                    self.env.render(mode = 'human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:

                    if self.record:
                        has_collected[done_indices] = 1
                        # if torch.sum(has_collected) == num_trials:
                            # print("224")
                            # break

                    if self.is_rnn:
                        for s in self.states:
                            s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)
                    if self.print_stats:
                        if print_game_res:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count)

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        print('255')
                        break
                    
                done_indices = done_indices[:, 0]

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life)


        import numpy as np
        if 'record_name' in self.config and self.record:
            record_name = self.config['record_name']
            dict_tosave = dict(global_translation_of_bodies=human_states[..., 0:3].cpu().numpy(), 
                global_rotation_of_bodies=human_states[..., 3:7].cpu().numpy(),
                captions=captions,
                trial_ids=trial_ids.cpu().numpy(),
                # global_trans_of_object=object_states[..., 0:3].cpu().numpy(),
                # global_rot_of_object=object_states[..., 3:7].cpu().numpy(),
                # dof_pos=human_dof_pos.cpu().numpy(),)
            )
            if hasattr(self.env.task, "mesh_vertices_offset"):
                dict_tosave["mesh_vertices"] = self.env.task.mesh_vertices_offset.cpu().numpy()
                dict_tosave["mesh_triangles"] = self.env.task.mesh_triangles.cpu().numpy()
                dict_tosave['verts_obj_idx'] = self.env.task.verts_obj_idx.cpu().numpy()
                dict_tosave['faces_obj_idx'] = self.env.task.faces_obj_idx.cpu().numpy()
                num_objs = dict_tosave['verts_obj_idx'].max() + 1
                face_offset = 0
                for i in range(num_objs):
                    dict_tosave[f'verts_obj_{i}'] = self.env.task.mesh_vertices_offset[self.env.task.verts_obj_idx == i].cpu().numpy()
                    if i > 0:
                        face_offset += dict_tosave[f'verts_obj_{i-1}'].shape[0]

                    dict_tosave[f'faces_obj_{i}'] = self.env.task.mesh_triangles[self.env.task.faces_obj_idx == i].cpu().numpy() - face_offset
            if hasattr(self.env.task, "_box_states"):
                dict_tosave["box_states"] = box_states.cpu().numpy()
                dict_tosave["box_scale"] = self.env.task._box_size.cpu().numpy()
            np.savez(
                os.path.join('render', f"{record_name}.npz"), 
                **dict_tosave

            )
            print("saved at ===>", os.path.join('render', f"{record_name}.npz"))

            # import json
            # import datetime

            # from env.tasks.humanoid_sit import ObjectCategoryId

            # time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # save_dir = "sim_results/{}".format(time)
            # os.makedirs(save_dir)

            # # get object info
            # obj_split = "train" if self.env.task._is_train else "test"
            # all_objects_urdf = self.env.task._object_lib._object_urdfs # list
            # all_objects_category = [i.name for i in ObjectCategoryId]
            # all_objects_cateIds = self.env.task._object_lib._object_cateIds.cpu().numpy()
            # every_env_object_unique_id = self.env.task._every_env_object_ids.cpu().numpy()
            # obj_info = {}
            # for i in range(self.env.task.num_envs):
            #     obj_info[i] = {
            #         "category": all_objects_category[all_objects_cateIds[every_env_object_unique_id[i]]],
            #         "urdf": all_objects_urdf[every_env_object_unique_id[i]],
            #         "split": obj_split,
            #     }

            # json_data = json.dumps(obj_info)
    
            # with open(os.path.join(save_dir, "object.json"), 'w') as file:
            #     file.write(json_data)


            # print("saved at ===>", save_dir)

        return

    def run_eval(self):
        is_determenistic = self.is_determenistic
        num_envs = self.env.num_envs
        num_trials = num_envs
        assert num_envs == num_trials
        num_repeat = 4

        print("evaluating policy: {} trials".format(num_envs))

        eval_res = {}

        for i in range(num_repeat):
            obs_dict = self.env_reset()
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)
            
            done_indices = []
            normal_done_indices = []

            games_played = 0
            games_success = 0
            sum_success_executionTime = 0
            sum_success_precision = 0

            games_fail_because_terminate = 0
            games_fail = 0

            has_collected = torch.zeros(num_envs, device=self.device)

            while games_played < num_trials:
                obs_dict = self.env_reset(normal_done_indices)
                action = self.get_action(obs_dict, is_determenistic)
                obs_dict, r, done, info =  self.env_step(self.env, action)

                normal_all_done_indices = done.nonzero(as_tuple=False)
                normal_done_indices = normal_all_done_indices[::self.num_agents, 0]

                done = torch.where(has_collected == 0, done, torch.zeros_like(done))
                has_collected[done == 1] += 1

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents, 0]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    success_done = info['success'][done_indices]
                    executionTime_done = info['executionTime'][done_indices]
                    percision_done = info['precision'][done_indices]
                    terminate_done = info['terminate'][done_indices]

                    # compute number of success
                    success_indices = success_done.nonzero(as_tuple=False)[:, 0]
                    success_count = len(success_indices)
                    games_success += success_count

                    terminate_count = torch.logical_and(terminate_done, success_done == 0).sum().cpu().item()
                    fail_count = done_count - success_count - terminate_count

                    games_fail_because_terminate += terminate_count
                    games_fail += fail_count

                    if success_count > 0:
                        sum_success_executionTime += executionTime_done[success_indices].sum().cpu().numpy()
                        sum_success_precision += percision_done[success_indices].sum().cpu().numpy()
  
                # self._post_step(info)
            
            success_rate = games_success / games_played
            if games_success > 0:
                mean_success_executionTime = sum_success_executionTime / games_success
                mean_success_precision = sum_success_precision / games_success
            else:
                mean_success_executionTime = 0
                mean_success_precision = 0
            
            curr_exp_name = "ObjectSet_{}_{}".format("test", i)
            eval_res[curr_exp_name] = {
                "num_trials": games_played,
                "success_trials": games_success,
                "success_rate": success_rate,
                "success_executionTime": mean_success_executionTime,
                "success_precision": mean_success_precision,
                "fail_trials": games_fail,
                "fail_trials_because_terminate": games_fail_because_terminate,
                "fail_trials_total": games_fail + games_fail_because_terminate,
            }

            print(curr_exp_name)
            print(eval_res[curr_exp_name])

        final_res_average = {}
        for exp_name, res in eval_res.items():
            for k in ['success_rate', 'success_executionTime', 'success_precision']:
                v = res[k]
                if k not in final_res_average:
                    final_res_average[k] = 0
                final_res_average[k] += v
        
        for k in final_res_average:
            final_res_average[k] /= num_repeat
        print("final_res_average:")
        print(final_res_average)

        # save metrics
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = os.path.dirname(self._checkpoint_fn)
        ckp_name = os.path.basename(self._checkpoint_fn)
        save_dir = os.path.join(folder_name, "../metrics", ckp_name[:-4])
        os.makedirs(save_dir, exist_ok=True)
        json.dump(eval_res, open(os.path.join(save_dir, "metrics_{}.json".format(time)), 'w'))
        print("save at {}".format(os.path.join(save_dir, "metrics_{}.json".format(time))))

        # save generated sit pose
        # save_path = os.path.join(save_dir, "generated_seated_poses_{}.pth".format(time))
        # torch.save(self._pose_buffer, save_path)
        # print("saved generated seated poses at ===> {}".format(save_path))

        # save_path = os.path.join(save_dir, "object_urdfs_{}.json".format(time))
        # with open(save_path, "w") as f:
        #     json.dump({"object_urdfs": self.env.task._object_lib._object_urdfs}, f)
        # print("saved object_urdfs at ===> {}".format(save_path))

        return

    def run(self):

        if self._eval:
            self.run_eval()
        else:
            self.run_test()

        return
