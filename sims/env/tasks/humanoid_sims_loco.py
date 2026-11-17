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
from sims.env.tasks.humanoid_sims_task import HumanoidSIMSTask, compute_loco_reward, compute_loco_observations

from shapely.geometry import Point, Polygon
from shapely.prepared import prep


def points_in_polygons(points_2d, polygons_2d):
    if isinstance(points_2d, torch.Tensor):
        points_2d = points_2d.detach().cpu().numpy()
    if isinstance(polygons_2d, torch.Tensor):
        polygons_2d = polygons_2d.detach().cpu().numpy()
    if len(points_2d) != len(polygons_2d):
        raise ValueError("Each point must have one corresponding polygon.")
    return np.asarray([
        prep(Polygon(polygon)).contains(Point(point))
        for point, polygon in zip(points_2d, polygons_2d)
    ], dtype=bool)


class HumanoidSIMSLoco(HumanoidSIMSTask):
    action_type = 'loco'
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        spacing = self.spacing
        if spacing <= 0.5:
            self._tar_pos_dist = torch.distributions.uniform.Uniform(
                torch.tensor([-4.5, -4.5], device=self.device),
                torch.tensor([4.5, 4.5], device=self.device))
        else:
            self._tar_pos_dist = torch.distributions.uniform.Uniform(
                torch.tensor([-(spacing - 0.5), -(spacing - 0.5)], device=self.device),
                torch.tensor([(spacing - 0.5), (spacing - 0.5)], device=self.device))
        
    def _reset_actors(self, env_ids):

        if self._enable_task_update:
            if self.num_obj_per_scene > 1:
                offsets = torch.randint(1, self.num_obj_per_scene, (len(env_ids),), device=self.device)
            else:
                offsets = 0
            self._every_env_target_obj_idx[env_ids] = (self._every_env_target_obj_idx[env_ids] + offsets) % self.num_obj_per_scene
            # self._every_env_target_obj_idx[env_ids] = torch.randint(0, self.num_obj_per_scene, (len(env_ids),), device=self.device)
            self.stand_point_choice[env_ids] = torch.zeros((len(env_ids),), device=self.device).long()
        self.stand_point_chosen[env_ids] = self.scene_stand_point[env_ids, self._every_env_target_obj_idx[env_ids], self.stand_point_choice[env_ids]]
        super(HumanoidSIMSTask, self)._reset_actors(env_ids)
        
    def _sample_motion_label(self, curr_env_ids):
        num_samples = len(curr_env_ids)
        given_ids = torch.ones((num_samples,), device=self.device).long() * self.num_resets
        if self._enable_contact and self._enable_task_update:
            skill_ids, motion_label, _, _, caption = self._motion_lib_dict.sample_motion_labels_obj_range(num_samples, obj_types=None, given_ids=given_ids)
            self._every_env_skill_id[curr_env_ids] = skill_ids
        else:
            skill_ids, motion_label, _, _, caption = self._motion_lib_dict.sample_motion_labels_obj_range(num_samples, obj_types=None, given_ids=given_ids)
            self._every_env_skill_id[curr_env_ids] = skill_ids
        if self.num_envs == 1 and len(caption) > 0:
            print(caption[0], self.num_resets)
        self.num_resets += 1
        self.captions = caption
        return motion_label
    
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
            target_overlap[target_overlap_ids] = 0
            for i in range(self.num_obj_per_scene):
                target_overlap_obj_i = points_in_polygons(root_pos[target_overlap_ids], ps_global[target_overlap_ids, i, :, :2])
                target_overlap_obj_i = torch.Tensor(target_overlap_obj_i).bool().to(self.device)
                target_overlap[target_overlap_ids] = target_overlap[target_overlap_ids] | target_overlap_obj_i
        self._humanoid_root_states[env_ids, :2] = root_pos

        rand_rot_theta = 2 * np.pi * torch.rand([len(env_ids)], device=self.device)
        axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        root_rot = quat_from_angle_axis(rand_rot_theta, axis)
        self._humanoid_root_states[env_ids, 3:7] = root_rot 

        if self._enable_conditional_disc:
            if len(env_ids):
                self._every_env_motion_label[env_ids] = self._sample_motion_label(env_ids)
                

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
            tar_pos = self.stand_point_chosen

            height_map = self.envs_heightmap_obj.clone().view(-1, self.obj_htmp_scale*self.obj_htmp_scale*self.num_obj_per_scene, 3)

            origin_root_pos = root_states[:, :3].clone()
            origin_root_pos[:, 0] = origin_root_pos[:, 0] - self.x_offset
            origin_root_pos[:, 1] = origin_root_pos[:, 1] - self.y_offset
            tar_dir = self.tar_dir
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_pos = self.stand_point_chosen[env_ids]

            height_map = self.envs_heightmap_obj.clone().view(-1, self.obj_htmp_scale*self.obj_htmp_scale*self.num_obj_per_scene, 3)[env_ids]

            origin_root_pos = root_states[:, :3].clone()
            origin_root_pos[:, 0] = origin_root_pos[:, 0] - self.x_offset[env_ids]
            origin_root_pos[:, 1] = origin_root_pos[:, 1] - self.y_offset[env_ids]
            tar_dir = self.tar_dir[env_ids]

        obs = compute_loco_observations(root_states, tar_pos, self.humanoid_in_mesh, origin_root_pos, self.local_scale, height_map, self.mesh_pos, tar_dir, self.with_heightmap)

        char_root_state = self._humanoid_root_states
        target = self.stand_point_chosen

        tar_speed = torch.ones_like(target[:, 0]) * 1.5
        self.rew_buf[:], self.location_diff_buf[:], self.tar_dir[:, :2] = compute_loco_reward(target, char_root_state, tar_speed, self._prev_root_pos, self.dt, self._is_train)
        # if self.reward_frame % 30 == 0 and self.num_envs == 1:
        #     print('reward:', round(self.rew_buf[0].item(), 4), 'location_diff:', round(self.location_diff_buf[0].item()))
        return obs
    

def compute_loco_reward(target, root_state, tar_speed, prev_root_pos, dt, is_train):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
    dist_threshold = 0.6

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
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0
    vel_reward[dist_mask] = 1

    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)
    facing_reward[dist_mask] = 1

    pos_reward = 0.5 * pos_reward_far + 0.5 * pos_reward_near

    pos_reward_w = 0.6
    vel_reward_w = 0.2
    face_reward_w = 0.1
    stand_still_reward_w = 0.1 * pos_reward

    # #when reaches, restrict speed to 0
    stand_still_reward = torch.exp(-0.5 * torch.sum(root_vel[..., :2] * root_vel[..., :2], dim=-1))
    stand_still_reward[~dist_mask] = 0

    # pos_reward[dist_mask] = 1.0 # already reached
    vel_reward[dist_mask] = 1.0 
    facing_reward[dist_mask] = 1.0

    far_reward = pos_reward_w * pos_reward + vel_reward_w * vel_reward + face_reward_w * facing_reward + stand_still_reward_w * stand_still_reward
    reward = far_reward
    if is_train:
        print('pos_reward:',round(pos_reward.mean().item(), 4), 'vel_reward:', round(vel_reward.mean().item(), 4), 'facing_reward:', round(facing_reward.mean().item(), 4), 'stand_still_reward:', round(stand_still_reward.mean().item(), 4))
    pos_dist = torch.sqrt(pos_err)
    return reward, pos_dist, tar_dir

# @torch.jit.script
def compute_loco_observations(root_states, tar_pos, human_in_mesh, origin_root_pos, local_scale, height_map_pcd, mesh_pos, tar_dir, with_htmp=True):
    # type: (Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)

    mesh_pos = mesh_pos[None] + (root_pos -human_in_mesh)[:, None]
    mesh_dist = (mesh_pos - root_pos[:, None]).reshape(-1, 3)
    heading_rot_for_height = root_rot[:, None].repeat(1, local_scale*local_scale, 1).reshape(-1, 4)
    heading_rot_for_height[:, :2] = 0
    heading_rot_norm = torch.sqrt(1 - heading_rot_for_height[:, 3]*heading_rot_for_height[:, 3])
    heading_rot_for_height[:, 2] = heading_rot_for_height[:, 2] * heading_rot_norm / torch.abs(heading_rot_for_height[:, 2])
    rotated_mesh_dist = quat_rotate(heading_rot_for_height, mesh_dist).reshape(-1, local_scale*local_scale, 3)
    rotated_mesh_origin_pos = rotated_mesh_dist + origin_root_pos[:, None]
    dist = rotated_mesh_origin_pos[..., :2][:, None] - height_map_pcd[..., :2][:, :, None]
    sum_dist = torch.sum(dist * dist, dim=-1)
    shape = sum_dist.shape
    sum_dist = sum_dist.permute(1,0,2).reshape(shape[1], -1)
    min_idx = sum_dist.argmin(0)
    dist_min = sum_dist[min_idx, range(min_idx.shape[0])]
    valid_mask = dist_min < 0.1 ** 2
    height_map = height_map_pcd.reshape(-1, 3)[min_idx, 2]
    height_map[~valid_mask] = 0.0
    height_map = height_map.reshape(shape[0], shape[2]).float()

    obs = torch.cat([local_tar_pos, tar_dir], dim=-1)
    if with_htmp:
        obs = torch.cat([obs, height_map], dim=-1)
    
    return obs
