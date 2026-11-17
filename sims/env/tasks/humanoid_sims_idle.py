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
import numpy as np
from sims.utils.torch_utils import quat_from_angle_axis
from sims.utils import torch_utils
from sims.utils.torch_utils import quat_rotate, quat_mul
from sims.env.tasks.humanoid_sims_loco import HumanoidSIMSLoco, compute_loco_observations, points_in_polygons


class HumanoidSIMSIdle(HumanoidSIMSLoco):
    action_type = 'loco'
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self.stand_target = torch.zeros(self.num_envs, 3, device=self.device)
        self.stand_target[:, 2] = 0.86

    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        root_pos = self._tar_pos_dist.sample((len(env_ids),))[:, :2]
        target_overlap = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        ps_global = self._every_env_obj_bps[env_ids, :, :4] # num_envs, num_obj, 4, 3
        for i in range(self.num_obj_per_scene):
            target_overlap_obj_i = points_in_polygons(root_pos[..., :2], ps_global[:, i, :, :2])
            target_overlap_obj_i = torch.Tensor(target_overlap_obj_i).bool().to(self.device)
            target_overlap = target_overlap | target_overlap_obj_i
        while(torch.sum(target_overlap) > 0):
            target_overlap_ids = torch.where(target_overlap)[0]
            root_pos[target_overlap_ids] = self._tar_pos_dist.sample((len(target_overlap_ids),))
            target_overlap_left = torch.zeros(len(target_overlap_ids), dtype=torch.bool, device=self.device)
            for i in range(self.num_obj_per_scene):
                target_overlap_obj_i = points_in_polygons(root_pos[target_overlap_ids], ps_global[target_overlap_ids, i, :, :2])
                target_overlap_obj_i = torch.Tensor(target_overlap_obj_i).bool().to(self.device)
                target_overlap[target_overlap_ids] = target_overlap_left | target_overlap_obj_i
        self._humanoid_root_states[env_ids, :2] = root_pos
        self.stand_target[env_ids, :2] = self._humanoid_root_states[env_ids, :2].clone()

        rand_rot_theta = 2 * np.pi * torch.rand([len(env_ids)], device=self.device)
        axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        root_rot = quat_from_angle_axis(rand_rot_theta, axis)
        self._humanoid_root_states[env_ids, 3:7] = root_rot 

        if self._enable_conditional_disc:
            if len(env_ids):
                self._every_env_motion_label[env_ids] = self._sample_motion_label(env_ids)


    def _reset_ref_state_init(self, env_ids):
        sk_ids = torch.multinomial(self._skill_init_prob, num_samples=env_ids.shape[0], replacement=True)

        for uid, sk_name in enumerate(self._skills):
            curr_motion_lib = self._motion_lib_dict[sk_name]
            curr_env_ids = env_ids[(sk_ids == uid).nonzero().squeeze(-1)] # be careful!!!

            if len(curr_env_ids) > 0:

                num_envs = curr_env_ids.shape[0]
                motion_ids = curr_motion_lib.sample_motions(num_envs)

                if (self._state_init == HumanoidSIMSIdle.StateInit.Random
                    or self._state_init == HumanoidSIMSIdle.StateInit.Hybrid):
                    motion_times = curr_motion_lib.sample_time(motion_ids) # avoid times with serious self-penetration
                elif (self._state_init == HumanoidSIMSIdle.StateInit.Start):
                    motion_times = torch.zeros(num_envs, device=self.device)
                else:
                    assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

                root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                    = curr_motion_lib.get_motion_state(motion_ids, motion_times)

                self._set_env_state(env_ids=curr_env_ids, 
                                    root_pos=root_pos, 
                                    root_rot=root_rot, 
                                    dof_pos=dof_pos, 
                                    root_vel=root_vel, 
                                    root_ang_vel=root_ang_vel, 
                                    dof_vel=dof_vel)

                self._reset_ref_env_ids[sk_name] = curr_env_ids
                self._reset_ref_motion_ids[sk_name] = motion_ids
                self._reset_ref_motion_times[sk_name] = motion_times


                body_pos, body_rot, body_vel, body_ang_vel \
                    = curr_motion_lib.get_motion_state_max(motion_ids, motion_times)

                self._kinematic_humanoid_rigid_body_states[curr_env_ids] = torch.cat((body_pos, body_rot, body_vel, body_ang_vel), dim=-1)
        
        self.stand_target[env_ids, :2] = self._humanoid_root_states[env_ids, :2].clone()

        if self._enable_conditional_disc:
            if len(env_ids) > 0:
                self._every_env_motion_label[env_ids] = self._sample_motion_label(env_ids)
        return


    def _compute_task_obs(self, env_ids=None):
        self.new_rigid_body_pos = self._rigid_body_pos.clone()

        head = quat_rotate(self._rigid_body_rot[:, 1], self.torso2head)
        head += self._rigid_body_pos[:, 1]
        self.new_rigid_body_pos[:, 2] = head

        torso = quat_rotate(self._rigid_body_rot[:, 0], self.pelvis2torso)
        torso += self._rigid_body_pos[:, 0]
        self.new_rigid_body_pos[:, 1] = torso
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_pos =root_states[:, :3].clone()

            height_map = self.envs_heightmap_obj.clone().view(-1, self.obj_htmp_scale*self.obj_htmp_scale*self.num_obj_per_scene, 3)

            origin_root_pos = root_states[:, :3].clone()
            origin_root_pos[:, 0] = origin_root_pos[:, 0] - self.x_offset
            origin_root_pos[:, 1] = origin_root_pos[:, 1] - self.y_offset
            tar_dir = self.tar_dir
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_pos = root_states[:, :3].clone()
            height_map = self.envs_heightmap_obj.clone().view(-1, self.obj_htmp_scale*self.obj_htmp_scale*self.num_obj_per_scene, 3)[env_ids]

            origin_root_pos = root_states[:, :3].clone()
            origin_root_pos[:, 0] = origin_root_pos[:, 0] - self.x_offset[env_ids]
            origin_root_pos[:, 1] = origin_root_pos[:, 1] - self.y_offset[env_ids]
            tar_dir = self.tar_dir[env_ids]

        obs = compute_loco_observations(root_states, tar_pos, self.humanoid_in_mesh, origin_root_pos, self.local_scale, height_map, self.mesh_pos, tar_dir, self.with_heightmap)

        char_root_state = self._humanoid_root_states
        target = self.stand_target
        self.rew_buf[:], self.location_diff_buf[:], self.tar_dir[:, :2] = compute_loco_reward(target, char_root_state,  self._is_train)
        # if self.reward_frame % 30 == 0 and self.num_envs == 1:
        #     print('reward:', round(self.rew_buf[0].item(), 4), 'location_diff:', round(self.location_diff_buf[0].item()))
        return obs
    

def compute_loco_reward(target, root_state, is_train):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor

    dist_threshold = 3
    pos_err_scale = 0.5
    pos_err_scale_near = 5
    vel_err_scale = 2.0
    
    root_pos = root_state[..., 0:3]
    root_rot = root_state[..., 3:7]
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    

    pos_diff = target - root_pos
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward_far = torch.exp(-pos_err_scale * pos_err)
    pos_reward_near = torch.exp(-pos_err_scale_near * pos_err)

    dist_mask = pos_err < dist_threshold ** 2

    tar_dir = torch.nn.functional.normalize(pos_diff[..., 0:2], dim=-1)

    pos_reward = 0.5 * pos_reward_far + 0.5 * pos_reward_near

    pos_reward_w = 1
    

    far_reward = pos_reward_w * pos_reward 
    reward = far_reward
    reward[dist_mask] = 1.0
    if is_train:
        print('pos_reward:',round(pos_reward.mean().item(), 4),)
    pos_dist = torch.sqrt(pos_err)
    return reward, pos_dist, tar_dir

