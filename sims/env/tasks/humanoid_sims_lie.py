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
from sims.utils import torch_utils
from sims.utils.torch_utils import quat_rotate
from sims.env.tasks.humanoid_sims_sit import HumanoidSIMSSit, compute_strike_observations


class HumanoidSIMSLie(HumanoidSIMSSit):    
    action_type = 'hsi'

    def _compute_task_obs(self, env_ids=None):
        pcd_buffer = []
        self.new_rigid_body_pos = self._rigid_body_pos.clone()

        head = quat_rotate(self._rigid_body_rot[:, 1], self.torso2head)
        head += self._rigid_body_pos[:, 1]
        self.new_rigid_body_pos[:, 2] = head

        torso = quat_rotate(self._rigid_body_rot[:, 0], self.pelvis2torso)
        torso += self._rigid_body_pos[:, 0]
        self.new_rigid_body_pos[:, 1] = torso
        joint_pos_buffer = []
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_pos = self.stand_point_chosen
            pcd_buffer = self.envs_obj_pcd_buffer
            env_num, joint_num, point_num, point_dim = pcd_buffer.shape
            pcd_buffer = pcd_buffer.view(-1, point_num, point_dim)
            pcd_buffer = pcd_buffer[range(pcd_buffer.shape[0]), self.joint_idx_buff.view(-1)]
            pcd_buffer = pcd_buffer.reshape(env_num, joint_num, point_dim)
            joint_pos_buffer = self.new_rigid_body_pos[:, self._strike_body_ids]
            height_map = self.envs_heightmap_obj.clone().view(-1, self.obj_htmp_scale*self.obj_htmp_scale*self.num_obj_per_scene, 3)
            contact_valid = self.env_contact_valid

            origin_root_pos = root_states[:, :3].clone()
            origin_root_pos[:, 0] = origin_root_pos[:, 0] - self.x_offset
            origin_root_pos[:, 1] = origin_root_pos[:, 1] - self.y_offset
            tar_dir = self.tar_dir
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_pos = self.stand_point_chosen[env_ids]
            pcd_buffer = self.envs_obj_pcd_buffer
            env_num, joint_num, point_num, point_dim = pcd_buffer.shape
            pcd_buffer = pcd_buffer.view(-1, point_num, point_dim)
            pcd_buffer = pcd_buffer[range(pcd_buffer.shape[0]), self.joint_idx_buff.view(-1)]
            pcd_buffer = pcd_buffer.reshape(env_num, joint_num, point_dim)
            pcd_buffer = pcd_buffer[env_ids]
            joint_pos_buffer = self.new_rigid_body_pos[:, self._strike_body_ids][env_ids]
            height_map = self.envs_heightmap_obj.clone().view(-1, self.obj_htmp_scale*self.obj_htmp_scale*self.num_obj_per_scene, 3)[env_ids]
            contact_valid = self.env_contact_valid[env_ids]
        
            origin_root_pos = root_states[:, :3].clone()
            origin_root_pos[:, 0] = origin_root_pos[:, 0] - self.x_offset[env_ids]
            origin_root_pos[:, 1] = origin_root_pos[:, 1] - self.y_offset[env_ids]
            tar_dir = self.tar_dir[env_ids]

        obs = compute_strike_observations(root_states, tar_pos, joint_pos_buffer, pcd_buffer, contact_valid, self.humanoid_in_mesh, origin_root_pos, self.local_scale, height_map, self.mesh_pos, tar_dir, self.with_heightmap)

        char_root_state = self._humanoid_root_states
        target = self.stand_point_chosen

        pcd_buffer = self.envs_obj_pcd_buffer
        joint_pos_buffer = self.new_rigid_body_pos[..., self._strike_body_ids, :]
        stand_pos = self.stand_point_chosen
        self.rew_buf[:], self.location_diff_buf[:], self.joint_diff_buff[:], self.joint_idx_buff[:], self.tar_dir[:, :2] = compute_contact_reward(target, char_root_state, pcd_buffer, joint_pos_buffer, self._prev_root_pos, self.dt, self.env_contact_valid, stand_pos, self._is_train)
        # if self.reward_frame % 20 == 0:
        #     print('reward:', round(self.rew_buf[0].item(), 4), 'location_diff:', round(self.location_diff_buf[0].item()))
        self.reward_frame += 1
        return obs

def compute_contact_reward(target, root_state, pcd_buffer, joint_pos_buffer, prev_root_pos, dt,
                           contact_valid, stand_pos, is_train):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
    dist_threshold = 0.5
    obj_contact_threshold = 0.5

    near_pos_err_scale = 5

    root_pos = root_state[..., 0:3]
    root_rot = root_state[..., 3:7]

    near_pos_reward_buf = []
    min_pos_idx_buf = []
    near_pos_err_min_buf = []
    for i in range(pcd_buffer.shape[1]):
        near_pos_diff = pcd_buffer[:, i] - joint_pos_buffer[:, i][:, None]
        near_pos_err = torch.sum(near_pos_diff * near_pos_diff, dim=-1)
        near_pos_err_min, min_pos_idx = near_pos_err.min(-1)

        near_pos_reward_contact = torch.exp(-near_pos_err_scale * near_pos_err_min)

        # reward_dir = torch.nn.functional.normalize(-near_pos_diff[range(near_pos_diff.shape[0]), min_pos_idx], dim=-1) * contact_direction[:,i]
        # reward_dir = reward_dir.sum(-1)
        # reward_dir[contact_direction[:,i].sum(1)==0] = 1
        # reward_dir[reward_dir<0] = 0
        # near_pos_err_min[reward_dir<0.5] += 1 # not fullfill dir
        # near_pos_reward_contact = 0.5 * near_pos_reward_contact + 0.5 * reward_dir

        # near_pos_reward_contact[~contact_valid[:, i]] = 1
        near_pos_reward_contact[~contact_valid[:, i]] = 0

        near_pos_reward_buf.append(near_pos_reward_contact)
        min_pos_idx_buf.append(min_pos_idx)
        near_pos_err_min_buf.append(near_pos_err_min)

    near_pos_err_min_buf = torch.stack(near_pos_err_min_buf, 1)
    near_pos_reward_buf = torch.stack(near_pos_reward_buf, 0)
    min_pos_idx_buf = torch.stack(min_pos_idx_buf, 1)
    near_pos_reward = near_pos_reward_buf.sum(0) / (contact_valid.sum(1) + 1e-6)

    # inter_joint_diff = joint_pos_buffer.view(-1, 15, 1, 3) - joint_pos_buffer.view(-1, 1, 15, 3)
    # joint_diff = torch.sum(inter_joint_diff * inter_joint_diff, dim=-1)
    # self_contact_reward = torch.exp(-0.5 * joint_diff) * self_contact_mask
    # self_contact_reward = (self_contact_reward.sum(-1).sum(-1)) / (self_contact_mask.sum(-1).sum(-1) + 1e-6)

    task_ready_mask = near_pos_err_min_buf[:, 0] < 1.2 **2

    # self_contact_reward[~task_ready_mask] = 0

    task_pos = pcd_buffer[range(pcd_buffer.shape[0]), 0, min_pos_idx_buf[:, 0]]
    stand_point_dir = stand_pos - task_pos
    stand_point_dir[..., 2] = 0
    stand_point_dir /= torch.linalg.norm(stand_point_dir, dim=-1, keepdim=True)
    ref_dir = torch.zeros_like(root_rot[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(root_rot, ref_dir)
    rot_dir[..., 2] = 0
    rot_dir /= torch.linalg.norm(rot_dir, dim=-1, keepdim=True)
    cosine = torch.sum(stand_point_dir * rot_dir, dim=-1)
    rot_reward = torch.exp(-1 * (1 - cosine))

    #########################################################
    pos_diff = target - root_pos
    pos_err = torch.sum(pos_diff[..., :2] * pos_diff[..., :2], dim=-1)
    pos_reward = torch.exp(- 0.5 * pos_err)
    dist_mask = pos_err < 0.7 **2
    # stand_reward, resist the root height to be higher than 0.8m
    stand_reward = 0.8 - root_pos[..., 2]
    stand_reward = torch.clamp(stand_reward, 0, 0.5)
    stand_reward = torch.exp(-10 * stand_reward)
    stand_reward[task_ready_mask] = 1

    tar_dir = torch.nn.functional.normalize(pos_diff[..., 0:2], dim=-1)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_speed = 1.0
    tar_vel_err = tar_speed - tar_dir_speed
    vel_reward = torch.exp(-2 * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0
    vel_reward[task_ready_mask] = 1

    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)
    facing_reward[task_ready_mask] = 1
    pos_reward[task_ready_mask] = 1

    far_reward = 0.5 * pos_reward + 0.2 * vel_reward + 0.1 * stand_reward + 0.2 * facing_reward
    far_reward[task_ready_mask] = 1
    #########################################################

    task_pos = pcd_buffer[range(pcd_buffer.shape[0]), 0, min_pos_idx_buf[:, 0]]

    tar_diff = task_pos - root_pos
    root_contact_err = torch.sum(tar_diff * tar_diff, dim=-1)
    siton_mask = root_contact_err < 0.3 ** 2

    task_contact_err = near_pos_err_min_buf[:, 0]
    task_contact_reward = torch.exp(-5 * task_contact_err)
    task_contact_reward[~task_ready_mask] = 0
    # near_reward = 0.4 * task_contact_reward + 0.2 * near_pos_reward + 0.2 * self_contact_reward + 0.2 * rot_reward
    rot_reward[~task_ready_mask] = 0
    rot_reward[siton_mask] = 1
    near_reward = 0.4 * task_contact_reward + 0.4 * near_pos_reward + 0.2 * rot_reward
    #########################################################
    reward = 0.6 * near_reward + 0.4 * far_reward

    # reward = near_reward
    # print(f'far:{float(pos_reward.mean()):.04f}, task contact:{float(task_contact_reward.mean()):.04f}, rot:{float(rot_reward.mean()):.04f}, stand:{float(stand_reward.mean()):.04f}, facing:{float(facing_reward.mean()):.04f}')
    # if is_train:
    if is_train:
        print(f'far:{float(pos_reward.mean()):.04f}, task contact:{float(task_contact_reward.mean()):.04f}, rot:{float(rot_reward.mean()):.04f}, near:{float(near_pos_reward.mean()):.04f}, facing:{float(facing_reward.mean()):.04f}', 'task_dist:', round(float(torch.sqrt(task_contact_err.mean())), 4))

    # contact_dist = torch.sqrt(root_contact_err)
    contact_dist = torch.abs(torch.sqrt(root_contact_err) - 0.17)

    return reward, contact_dist, near_pos_err_min_buf, min_pos_idx_buf, tar_dir
