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
from sims.utils import torch_utils
from sims.utils.torch_utils import quat_rotate, quat_mul, quat_from_angle_axis
# from sims.env.tasks.humanoid_sims import HumanoidSIMS
from sims.env.tasks.humanoid_sims_3dfront import HumanoidSIMS


class HumanoidSIMSTask(HumanoidSIMS):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._enable_task_obs = cfg["env"]['task']["enableTaskObs"]
        self._is_eval = cfg["args"].eval
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self.reward_frame = 0
        if (not self._is_train) and self._is_eval:
            self._success_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.long)
            self._executionTime_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)
            self._precision_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)

            self.max_episode_length = cfg["eval"]["episodeLength"]
            # self._enable_IET = cfg["eval"]["enableIET"]
            self._state_init = HumanoidSIMS.StateInit[cfg["eval"]["stateInit"]]
            self._success_threshold = cfg["eval"]["successThreshold"]

        return
    
    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if (self._enable_task_obs):
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size
    
    def get_task_obs_size(self): # 433
        obs_size = 0 
        if (self._enable_task_obs):
            if self.action_type == 'hsi':
                obs_size = 6 + self.contact_joint_num + self.contact_joint_num * 3 # tar_pos, tar_dir, contact_valid, local_target_pos
            elif self.action_type == 'loco':
                obs_size = 6 # tar_pos, tar_dir
            elif self.action_type == 'hoi':
                obs_size = 3 + 3 + 6 + 8 * 3 # tar_pos, pos 3 + rot 6 + bps 24
            if self.with_heightmap:
                obs_size += self.local_scale*self.local_scale
        return obs_size
    
    def post_physics_step(self):
        super().post_physics_step()
        if (not self._is_train) and self._is_eval:
            self._compute_metrics_evaluation()
            self.extras["success"] = self._success_buf
            self.extras["executionTime"] = self._executionTime_buf
            self.extras["precision"] = self._precision_buf

    def _compute_metrics_evaluation(self):
        pos_err = self.location_diff_buf
        dist_mask = pos_err <= self._success_threshold
        self._success_buf[dist_mask] += 1

        self._precision_buf[dist_mask] = torch.where(pos_err[dist_mask] < self._precision_buf[dist_mask], pos_err[dist_mask], self._precision_buf[dist_mask])

        first_time_success_mask = (self._success_buf == 1)
        self._executionTime_buf[first_time_success_mask] = self.progress_buf[first_time_success_mask] * self.dt

        return
    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_task()
        return

    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return

    def _update_task(self):
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        self._reset_task(env_ids)
        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        
        # if self._enable_IET:
        #     self._IET_step_buf[env_ids] = 0
        #     self._IET_triggered_buf[env_ids] = 0

        if (not self._is_train) and self._is_eval:
            self._success_buf[env_ids] = 0
            self._executionTime_buf[env_ids] = 0
            self._precision_buf[env_ids] = float('Inf')
        return
    
    def _reset_task(self, env_ids):
        return

    def _compute_task_obs(self, env_ids=None):
        raise NotImplemented

    # def process_contact(self):
    #     raise NotImplemented

    def _compute_reward(self, actions):
        return NotImplemented

    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)

        if (self._enable_conditional_disc):
            if (env_ids is None):
                conditions = self._every_env_motion_label
            else:
                conditions = self._every_env_motion_label[env_ids]
            obs = torch.cat([humanoid_obs, conditions], dim=-1)
        else:
            obs = humanoid_obs
            
        if (self._enable_task_obs):
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([obs, task_obs], dim=-1)
        else:
            obs = obs

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return

    def _draw_task(self):
        self.gym.clear_lines(self.viewer)
        cols = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        starts = self.new_rigid_body_pos[0][self._strike_body_ids][self.env_contact_valid[0]]
        ends = self.envs_obj_pcd_buffer[0][range(self.contact_joint_num), self.joint_idx_buff[0]][self.env_contact_valid[0]]

        # joint_pos_buffer = self.new_rigid_body_pos[:, self._strike_body_ids][0]
        # joint_contact_choice = self.env_joint_pairs[0]
        # valid_joint_contact_choice = self.env_joint_pairs_valid[0]
        # joints_contact = joint_pos_buffer.view(-1, 3)[joint_contact_choice.view(-1)].clone()
        # try:
        #     ends[valid_joint_contact_choice.view(-1)[self.env_contact_valid[0]]] = joints_contact[valid_joint_contact_choice.view(-1)]
        # except:
        #     from IPython import embed; embed()
        # obj contact
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()
        cols = cols.repeat(verts.shape[0], axis=0)
        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts
            curr_verts = curr_verts.reshape(-1, 6)
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        # draw self contact
        # cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        # for i, env_ptr in enumerate(self.envs[:1]):

        #     j1_idx, j2_idx = torch.where(self.env_self_contact_mask[i])
        #     if len(j1_idx):
        #         curr_verts = torch.cat([self.new_rigid_body_pos[i][self._strike_body_ids][j1_idx], self.new_rigid_body_pos[i][self._strike_body_ids][j2_idx]], dim=-1).cpu().numpy()
        #         cols = cols.repeat(curr_verts.shape[0], axis=0)
        #         curr_verts = curr_verts.reshape(-1, 6)
        #         self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        starts = self._humanoid_root_states[..., 0:3]
        ends = self.stand_point_chosen
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()
        cols = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        self._update_marker()
        super()._draw_task()
        return

    def _reset_actors(self, env_ids):

        if self._enable_task_update:
            offsets = torch.randint(1, self.num_obj_per_scene, (len(env_ids),), device=self.device)
            self._every_env_target_obj_idx[env_ids] = (self._every_env_target_obj_idx[env_ids] + offsets) % self.num_obj_per_scene
            # self._every_env_target_obj_idx[env_ids] = torch.randint(0, self.num_obj_per_scene, (len(env_ids),), device=self.device)
            self.stand_point_choice[env_ids] = torch.zeros((len(env_ids),), device=self.device).long()
        self.stand_point_chosen[env_ids] = self.scene_stand_point[env_ids, self._every_env_target_obj_idx[env_ids], self.stand_point_choice[env_ids]]
        super()._reset_actors(env_ids)


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


@torch.jit.script
def compute_hoi_observations(root_states, box_states, box_bps, tar_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot) # (num_envs, 4)

    box_pos = box_states[:, 0:3]
    box_rot = box_states[:, 3:7]
    
    local_box_pos = box_pos - root_pos
    local_box_pos = quat_rotate(heading_rot, local_box_pos)

    local_box_rot = quat_mul(heading_rot, box_rot)
    local_box_rot_obs = torch_utils.quat_to_tan_norm(local_box_rot)

    # compute observations for bounding points of the box
    ## transform from object local space to world space
    box_pos_exp = torch.broadcast_to(box_pos.unsqueeze(-2), (box_pos.shape[0], box_bps.shape[1], box_pos.shape[1])) # (num_envs, 3) >> (num_envs, 8, 3)
    box_rot_exp = torch.broadcast_to(box_rot.unsqueeze(-2), (box_rot.shape[0], box_bps.shape[1], box_rot.shape[1])) # (num_envs, 4) >> (num_envs, 8, 4)
    box_bps_world_space = quat_rotate(box_rot_exp.reshape(-1, 4), box_bps.reshape(-1, 3)) + box_pos_exp.reshape(-1, 3) # (num_envs*8, 3)

    ## transform from world space to humanoid local space
    heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(-2), (heading_rot.shape[0], box_bps.shape[1], heading_rot.shape[1]))
    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(-2), (root_pos.shape[0], box_bps.shape[1], root_pos.shape[1]))
    box_bps_local_space = quat_rotate(heading_rot_exp.reshape(-1, 4), box_bps_world_space - root_pos_exp.reshape(-1, 3)) # (num_envs*8, 3)

    # task obs
    local_tar_pos = quat_rotate(heading_rot, tar_pos - root_pos) # 3d xyz

    obs = torch.cat([local_box_pos, local_box_rot_obs, box_bps_local_space.reshape(root_pos.shape[0], -1), local_tar_pos], dim=-1)
    return obs

# @torch.jit.script
def compute_strike_observations(root_states, tar_pos, joint_pos_buffer, pcd_buffer, contact_valid, human_in_mesh, origin_root_pos, local_scale, height_map_pcd, mesh_pos, tar_dir, with_htmp=True):
    # type: (Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)

    # local_heightmap_pcd = obj_heightmap_pcd - root_pos[:, None]
    # local_heightmap_pcd = quat_rotate(heading_rot[:, None].repeat(1, local_heightmap_pcd.shape[1], 1).view(-1, 4), local_heightmap_pcd.view(-1, 3)).view(local_heightmap_pcd.shape)
    # local_heightmap_pcd = local_heightmap_pcd.view(local_heightmap_pcd.shape[0], -1)

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

    obs = torch.cat([local_tar_pos, tar_dir, contact_valid], dim=-1)
    if with_htmp:
        obs = torch.cat([obs, height_map], dim=-1)
    # 3 + 3 + 15  + 81 + 192 = 294
    local_target_pos = pcd_buffer - joint_pos_buffer
    local_target_pos_r = quat_rotate(heading_rot[:, None].repeat(1, local_target_pos.shape[1], 1).view(-1, 4), local_target_pos.view(-1, 3))
    local_target_pos_r = local_target_pos_r.reshape(local_target_pos.shape)
    local_target_pos_r *= contact_valid[..., None]
    obs = torch.cat([obs, local_target_pos_r.view(obs.shape[0], -1),], dim=-1)
    # 294 + 15 = 309
    
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


def compute_heightmap_obs(root_states, human_in_mesh, origin_root_pos, local_scale, height_map_pcd, mesh_pos):
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
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
    return height_map
