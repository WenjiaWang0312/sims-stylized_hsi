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
from isaacgym.torch_utils import quat_from_angle_axis
from sims.utils import torch_utils
from sims.utils.torch_utils import quat_rotate, quat_mul
from sims.env.tasks.humanoid_sims_task import HumanoidSIMSTask, compute_strike_observations
from sims.env.tasks.humanoid_sims_loco import points_in_polygons

def flatten_upper_triangles_torch(tensors):
    n = tensors.size(1)  
    i, j = torch.triu_indices(n, n, offset=1)
    flattened_upper_triangles = tensors[:, i, j]
    return flattened_upper_triangles


class HumanoidSIMSStandup(HumanoidSIMSTask):
    action_type = 'hsi'
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        spacing = 1
        self._tar_pos_dist = torch.distributions.uniform.Uniform(
            torch.tensor([-(spacing - 0.5), -(spacing - 0.5)], device=self.device),
            torch.tensor([(spacing - 0.5), (spacing - 0.5)], device=self.device))
        self.env_contact_valid = torch.zeros([self.num_envs, self.contact_joint_num], device=self.device, dtype=torch.bool)

    def _reset_actors(self, env_ids): 
        if self._enable_task_update:
            # offsets = torch.randint(1, self.num_obj_per_scene, (len(env_ids),), device=self.device)
            # self._every_env_target_obj_idx[env_ids] = (self._every_env_target_obj_idx[env_ids] + offsets) % self.num_obj_per_scene
            self._every_env_target_obj_idx[env_ids] = torch.randint(0, self.num_obj_per_scene, (len(env_ids),), device=self.device)
            self.stand_point_choice[env_ids] = torch.zeros((len(env_ids),), device=self.device).long()
            
            self.stand_point_chosen[env_ids] = self.scene_stand_point[env_ids, self._every_env_target_obj_idx[env_ids], self.stand_point_choice[env_ids]]
            self.stand_point_chosen[:, 2] = 0.86
            self.tar_heightmap = self.envs_heightmap_obj[range(len(self._every_env_target_obj_idx)), self._every_env_target_obj_idx]

            contact_valid_steps = self.env_contact_valid
            threshold = 0.3# if 'sitdown' in self._skills else 0.3
            success = (self.location_diff_buf < threshold) & ~self.big_force
            fulfill = ((contact_valid_steps  & (self.joint_diff_buff < 0.3)) 
                            | (~contact_valid_steps)) & (success[:, None])
            fulfill = torch.all(fulfill, dim=-1)[env_ids]

            reset = ~fulfill
            if len(env_ids[reset]):
                super(HumanoidSIMSTask, self)._reset_actors(env_ids[reset])
            self.still_buf[env_ids[reset|fulfill]] = 0
            
            fulfill_ids = env_ids[fulfill]
            sit_fulfill_ids = fulfill_ids[contact_valid_steps[fulfill_ids, 0] >0 ]
            standup_fulfill_ids = fulfill_ids[contact_valid_steps[fulfill_ids, 0] == 0]
            self.clean_contact(sit_fulfill_ids)
            self._sample_motion_label(standup_fulfill_ids)

    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids

        obj_center = self._every_env_obj_bps[env_ids, self._every_env_target_obj_idx[env_ids], :4].mean(1)
        shift = (self.stand_point_chosen[env_ids] - obj_center) * 0.5
        offset = self._tar_pos_dist.sample([len(env_ids)])
        if not self._is_train:
            shift *= 0
            offset *= 0
        self._humanoid_root_states[env_ids, 0] = self.stand_point_chosen[env_ids, 0]# + shift[:, 0] + offset[:, 0]
        self._humanoid_root_states[env_ids, 1] = self.stand_point_chosen[env_ids, 1] #+ shift[:, 1] + offset[:, 1]

        if not self._is_train:
            deltax = self.stand_point_chosen[env_ids, 0] - obj_center[:, 0]
            deltay = self.stand_point_chosen[env_ids, 1] - obj_center[:, 1]
            rot_theta = torch.arctan2(deltay, deltax)
            axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            root_rot = quat_from_angle_axis(rot_theta, axis)
        else:
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

                if (self._state_init == HumanoidSIMSGetup.StateInit.Random
                    or self._state_init == HumanoidSIMSGetup.StateInit.Hybrid):
                    motion_times = curr_motion_lib.sample_time_rsi(motion_ids) # avoid times with serious self-penetration
                elif (self._state_init == HumanoidSIMSGetup.StateInit.Start):
                    motion_times = torch.zeros(num_envs, device=self.device)
                else:
                    assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

                root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                    = curr_motion_lib.get_motion_state(motion_ids, motion_times)

                root_pos[:, 0] = self.stand_point_chosen[curr_env_ids, 0]
                root_pos[:, 1] = self.stand_point_chosen[curr_env_ids, 1]
                obj_center = self._every_env_obj_bps[curr_env_ids, self._every_env_target_obj_idx[curr_env_ids], :4].mean(1)
                deltax = root_pos[:, 0] - obj_center[:, 0]
                deltay = root_pos[:, 1] - obj_center[:, 1]
                rot_theta = torch.arctan2(deltay, deltax)
                axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
                root_rot = quat_from_angle_axis(rot_theta, axis)
                
                # rot_theta = torch.deg2rad(self.obj_rotate_angle[curr_env_ids]).float()
                # axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
                # # heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
                # # # rot_offset = quat_mul(quat_from_angle_axis(rot_theta, axis), heading_rot)
                # # # root_rot = quat_mul(rot_offset, root_rot)
                # root_rot = quat_from_angle_axis(rot_theta, axis)

                # # rand_rot_theta = 2 * np.pi * torch.rand([len(env_ids)], device=self.device)
                # # axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
                # # root_rot = quat_from_angle_axis(rand_rot_theta, axis)
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

        if self._enable_conditional_disc:
            if len(env_ids):
                self._every_env_motion_label[env_ids] = self._sample_motion_label(env_ids)
        return

    def clean_contact(self, env_ids):
        self.env_contact_valid[env_ids] = 0
        self.envs_obj_pcd_buffer[env_ids] = 0

    def process_contact(self, env_ids, obj_type,  env_obj_contact, caption):
        assert len(env_ids) == len(obj_type) == len(env_obj_contact)
        self.env_contact_valid[env_ids] = 0
        self.envs_obj_pcd_buffer[env_ids] = 0

        for curr_idx, curr_env_idx in enumerate(env_ids):
            curr_children = self._every_env_children[curr_env_idx][self._every_env_target_obj_idx[curr_env_idx]]
            curr_obj_type = obj_type[curr_idx]
            if curr_obj_type == 'bed':
                skill_name = 'liedown'
            else:
                skill_name = 'sitdown'
            # skill_name  = 'sitdown' if 'sitdown' in self._skills else 'liedown'
            task_contact = self.task_contact[skill_name][curr_obj_type]

            if skill_name in ['sitdown', 'liedown']:
                for task_contact_item in task_contact:

                    j1name, partname, contact_type = task_contact_item.split('-')
                    if contact_type == 'up':
                        direction = contact_type
                        contact = 1
                    j1_idx = self.contact_joint_mapping[self.joint_abbr[j1name]]
                    # child_idx = curr_children.index(partname)
                    child_idx = 0
                    curr_part_pcd = self.envs_children_pcd_buffer[curr_env_idx, self._every_env_target_obj_idx[curr_env_idx], child_idx]
                    self.envs_obj_pcd_buffer[curr_env_idx, j1_idx] = curr_part_pcd
                
                    self.env_contact_valid[curr_env_idx, j1_idx] = contact


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

        obs = compute_strike_observations(root_states, tar_pos, joint_pos_buffer, pcd_buffer, contact_valid, self.humanoid_in_mesh, origin_root_pos, self.local_scale, height_map, self.mesh_pos, tar_dir,)

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
    

    def _compute_reset(self):
        self.big_force = (self._contact_forces.abs()>10000).sum((-2,-1))>0

        contact_valid_steps = self.env_contact_valid

        success = (self.location_diff_buf < 0.2) & ~self.big_force

        fulfill = ((contact_valid_steps  & (self.joint_diff_buff < 0.3)) 
                        | (~contact_valid_steps)) & (success[:, None])
        fulfill = torch.all(fulfill, dim=-1)
        # print(contact_valid_steps[0, 0])
        self.still = self.still_buf>10 & fulfill
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights,  self._rigid_body_vel, self.still_buf, fulfill)
        return
#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, _rigid_body_vel, still_buf, fulfill):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, int, bool, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    # terminated = torch.where(big_force, torch.ones_like(reset_buf), terminated) # terminate 
        
    still_buf[_rigid_body_vel.abs().sum(-1).max(-1)[0]<0.6] += 1
    still_buf[_rigid_body_vel.abs().sum(-1).max(-1)[0]>0.6] = 0

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
    reset = torch.where((still_buf>10) & fulfill, torch.ones_like(reset_buf), reset)
    return reset, terminated

def compute_contact_reward(target, root_state, pcd_buffer, joint_pos_buffer, prev_root_pos, dt,
                           contact_valid,  stand_pos, is_train,):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float,) -> Tensor
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


    task_ready_mask = near_pos_err_min_buf[:, 0] < 0.8 **2


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
    pos_reward = torch.exp(- 5 * pos_err)
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
    # vel_reward[task_ready_mask] = 1

    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)
    # facing_reward[task_ready_mask] = 1
    # pos_reward[task_ready_mask] = 1

    far_reward = 0.8 * pos_reward + 0.2 * facing_reward
    # far_reward[task_ready_mask] = 1
    #########################################################

    task_contact_err = near_pos_err_min_buf[:, 0]
    task_contact_reward = torch.exp(-2 * task_contact_err)
    # task_contact_reward[~task_ready_mask] = 0
    rot_reward[~task_ready_mask] = 0
    near_reward = 0.8 * task_contact_reward + 0.2 * rot_reward #+ 0.2 * near_pos_reward
    #########################################################

    not_walking = contact_valid.sum(-1)>0
    reward = far_reward
    reward[not_walking] = near_reward[not_walking]

    # reward = near_reward
    # print(f'far:{float(pos_reward.mean()):.04f}, task contact:{float(task_contact_reward.mean()):.04f}, rot:{float(rot_reward.mean()):.04f}, stand:{float(stand_reward.mean()):.04f}, facing:{float(facing_reward.mean()):.04f}')
    if is_train:
        print(f'far:{float(pos_reward.mean()):.04f}, task contact:{float(task_contact_reward.mean()):.04f}, rot:{float(rot_reward.mean()):.04f}')
    # task_pos = pcd_buffer[range(pcd_buffer.shape[0]), 0, min_pos_idx_buf[:, 0]]
    # task_diff = task_pos - root_pos
    # root_contact_err = torch.sum(task_diff * task_diff, dim=-1)
    # contact_dist = torch.sqrt(root_contact_err)
    # from IPython import embed; embed()
    contact_dist = ((near_pos_err_min_buf * contact_valid).sum(1)) / (contact_valid.sum(1) + 1e-6)
    contact_dist = torch.sqrt(contact_dist)
    return reward, contact_dist, near_pos_err_min_buf, min_pos_idx_buf, tar_dir
