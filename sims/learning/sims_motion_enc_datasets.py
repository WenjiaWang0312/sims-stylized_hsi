from isaacgym.torch_utils import quat_rotate, quat_mul, quat_from_angle_axis, to_torch
import torch
import random
from sims.utils import torch_utils
import numpy as np
from sims.utils.motion_lib_dict import MotionLibCLIP, MotionLibDict
from sims.utils.clip_utils import CLIPExtractor

amp_humanoid_joint_names =  [
    "pelvis",
    "torso",
    "head",
    "right_upper_arm",
    "right_lower_arm",
    "right_hand",
    "left_upper_arm",
    "left_lower_arm",
    "left_hand",
    "right_thigh",
    "right_shin",
    "right_foot",
    "left_thigh",
    "left_shin",
    "left_foot"]


class MotionClipDataset():
    def __init__(self, motion_dict, device='cuda:0', repeat=1, extract_from_text=False) -> None:
        self._local_root_obs = True
        self._root_height_obs = True
        
        self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
        self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
        self._dof_obs_size = 72
        key_bodies = ["right_hand", "left_hand", "right_foot", "left_foot"]
        key_body_ids = [amp_humanoid_joint_names.index(key) for key in key_bodies]
        key_body_ids = to_torch(key_body_ids, torch.int64, device=device)

        self._num_amp_obs_enc_steps = 300
        self.dt = 1.0 / 30.0

        self.device = key_body_ids.device
        self._motion_lib = MotionLibCLIP(motion_dict, self._dof_body_ids, self._dof_offsets, key_body_ids, device, extract_from_text=extract_from_text, max_label_per_motion=5)
    
        num_key_bodies = len(key_bodies)

        self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        self.repeat = repeat

            
    def get_num_enc_amp_obs(self):
        return self._num_amp_obs_enc_steps * self._num_amp_obs_per_step


    def motion_length(self, index):
        
        return self._motion_lib._motion_lengths[index]

    def motion_frame(self, index):
        return int(self._motion_lib._motion_num_frames[index])
    
    def lengths_to_mask(self, lengths):
        mask = torch.arange(self._num_amp_obs_enc_steps).to(self.device) < lengths
        return mask    
        
    def __len__(self):
        return len(self._motion_lib.captions) * self.repeat
    
    @property
    def real_len(self):
        return len(self._motion_lib.captions)
    
    def __getitem__(self, index):
        motion_index = int(index) % self.real_len
        motion_ids = torch.tensor([motion_index], dtype=torch.long, device=self.device)
        obs, mask = self.build_motionclip_repr(motion_index)
        clip_features = self._motion_lib.get_motion_label(motion_ids)
        random_clip_features = self._motion_lib.get_motion_label(motion_ids)
        # mask = self.lengths_to_mask(self._motion_lib._motion_lengths[motion_ids])
        batch = {}
        batch['motion_vector'] = obs.permute(1, 0) 
        batch['mask'] = mask
        batch['clip_features'] = clip_features.view(512)
        batch['extra_clip_features'] = random_clip_features.view(512)
        # batch['captions'] = captions
        return batch

    def fetch_amp_obs_demo_enc(self, index):
        dt = self.dt
        if isinstance(index, int):
            index = torch.Tensor([index]).long().to(self.device)
        elif isinstance(index, torch.Tensor):
            index = index.long().to(self.device)
        elif isinstance(index, list):
            index = torch.tensor(index).long().to(self.device)
        motion_ids = index
        
        enc_window_size = self.dt * (self._num_amp_obs_enc_steps - 1)

        enc_motion_times = self._motion_lib.sample_time(motion_ids, truncate_time=enc_window_size)
        # make sure not to add more than motion clip length, negative amp_obs will show zero index amp_obs instead
        enc_motion_times += torch.clip(self._motion_lib._motion_lengths[motion_ids], max=enc_window_size)

        enc_amp_obs_demo = self.build_amp_obs_demo(motion_ids, enc_motion_times, self._num_amp_obs_enc_steps).view(-1, self._num_amp_obs_enc_steps, self._num_amp_obs_per_step)
        enc_amp_obs_demo_flat = enc_amp_obs_demo.to(self.device)#.view(-1, self.get_num_enc_amp_obs())
        return enc_amp_obs_demo_flat
    
    def build_amp_obs_demo(self, motion_ids, motion_times0, num_steps):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, num_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, num_steps, device=self.device)
        motion_times = torch.clip(motion_times + time_steps, min=0)

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
            = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              dof_pos, dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets)
        return amp_obs_demo
    
    def build_motionclip_repr(self, index):
        motion = self._motion_lib._motions[index]
        gts = motion.global_translation.float()
        grs = motion.global_rotation.float()
        lrs = motion.local_rotation.float()
        grvs = motion.global_root_velocity.float()
        gravs = motion.global_root_angular_velocity.float()
        dvs = motion.dof_vels.float()

        motion_frames = self.motion_frame(index)
        if motion_frames > self._num_amp_obs_enc_steps:
            start = random.randint(0, motion_frames - self._num_amp_obs_enc_steps)
            frame_indices = torch.arange(start, start + self._num_amp_obs_enc_steps)
            mask = torch.ones(self._num_amp_obs_enc_steps, dtype=torch.bool).to(self.device)
        else:
            frame_indices = torch.arange(0, motion_frames)
            frame_indices = torch.cat((frame_indices, torch.zeros(self._num_amp_obs_enc_steps - motion_frames, dtype=torch.int64)))
            mask = self.lengths_to_mask(motion_frames)
        
        root_pos = gts[frame_indices, 0]
        root_rot = grs[frame_indices, 0]
        root_vel = grvs[frame_indices]
        root_ang_vel = gravs[frame_indices]
        local_rot = lrs[frame_indices]
        dof_pos = self._motion_lib._local_rotation_to_dof(local_rot)
        dof_vel = dvs[frame_indices]
        key_pos = gts[frame_indices][ :, self._motion_lib._key_body_ids]

        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              dof_pos, dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets)
        return amp_obs_demo, mask

def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs): #True
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs): #True
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1) # 1 + 6 + 3 + 3 + 72 + 28 + 12 = 125
    return obs

def dof_to_obs(pose, dof_obs_size, dof_offsets):
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose)
        elif (dof_size == 1):
            axis = torch.tensor([0.0, 1.0, 0.0], dtype=joint_pose.dtype, device=pose.device)
            joint_pose_q = quat_from_angle_axis(joint_pose[..., 0], axis)
        else:
            joint_pose_q = None
            assert(False), "Unsupported joint type"

        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q)
        dof_obs[:, (j * joint_obs_size):((j + 1) * joint_obs_size)] = joint_dof_obs

    assert((num_joints * joint_obs_size) == dof_obs_size)
    return dof_obs
