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

import numpy as np
import os
import yaml
from isaacgym.torch_utils import *

from sims.poselib.poselib.skeleton.skeleton3d import SkeletonMotion
from sims.poselib.poselib.core.rotation3d import *
from sims.utils.clip_utils import CLIPExtractor
from sims.utils import torch_utils
import torch

USE_CACHE = True
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy

class DeviceCache:
    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)  
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
        
        print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out


class MotionLib():
    def __init__(self, motion_dict, dof_body_ids, dof_offsets,
                 key_body_ids, device, is_test=False):
        self._dof_body_ids = dof_body_ids
        self._dof_offsets = dof_offsets
        self._num_dof = dof_offsets[-1]
        self._key_body_ids = torch.tensor(key_body_ids, device=device)
        self._device = device
        if not is_test:
            self._load_motions(motion_dict)

            motions = self._motions
            self.gts = torch.cat([m.global_translation for m in motions], dim=0).float()
            self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float()
            self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float() # rotation
            self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float()
            self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float() #self.global_velocity[..., 0, :]
            self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float() #.global_angular_velocity[..., 0, :]
            self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float()
            self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float()

            lengths = self._motion_num_frames
            lengths_shifted = lengths.roll(1)
            lengths_shifted[0] = 0
            self.length_starts = lengths_shifted.cumsum(0)
            self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)
            if len(self._obj_motions) > 0:
                motions = self._obj_motions
                self.obj_gts = torch.cat([m[:, 0:3] for m in motions], dim=0).float()
                self.obj_grs = torch.cat([m[:, 3:7] for m in motions], dim=0).float()
        return

    def num_motions(self):
        return len(self._motions)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._motion_weights, num_samples=n, replacement=True)

        # m = self.num_motions()
        # motion_ids = np.random.choice(m, size=n, replace=True, p=self._motion_weights)
        # motion_ids = torch.tensor(motion_ids, device=self._device, dtype=torch.long)
        return motion_ids

    def sample_time_contact(self, motion_ids, truncate_time=None):
        # this function is designed for reference state init
        # it gives motion times sampled in allowed range
        n = len(motion_ids)
        succ_sample_times = torch.zeros(n, dtype=torch.float32, device=self._device)
        succ_sample_record = torch.zeros(n, dtype=torch.bool, device=self._device)
        while torch.sum(succ_sample_record) < n:
            # print(torch.sum(succ_sample_record))
            mask = (succ_sample_record == False)

            phase = torch.rand(motion_ids[mask].shape, device=self._device)
            
            motion_len = self._motion_lengths[motion_ids[mask]]
            if (truncate_time is not None):
                assert (truncate_time >= 0.0)
                motion_len -= truncate_time

            motion_time = phase * motion_len

            # check if it is within the skipped range
            sample_range = self._motion_contact_ranges[motion_ids[mask]]

            curr_num_samples = motion_time.shape[0]
            # need to sample only within the allowed range
            curr_succ = torch.where(
                torch.logical_and(motion_time <= sample_range[:, 0], motion_time >= sample_range[:, 1]),
                torch.zeros(curr_num_samples, dtype=torch.bool, device=self._device),
                torch.ones(curr_num_samples, dtype=torch.bool, device=self._device),
            )
            curr_succ_inds = torch.nonzero(mask)[curr_succ, 0]
            succ_sample_times[curr_succ_inds] = motion_time[curr_succ]
            succ_sample_record[curr_succ_inds] = True

        return succ_sample_times
    
    def sample_time_rsi(self, motion_ids, truncate_time=None):
        # this function is designed for reference state init
        # it gives motion times sampled in allowed range
        n = len(motion_ids)
        succ_sample_times = torch.zeros(n, dtype=torch.float32, device=self._device)
        succ_sample_record = torch.zeros(n, dtype=torch.bool, device=self._device)
        while torch.sum(succ_sample_record) < n:
            # print(torch.sum(succ_sample_record))
            mask = (succ_sample_record == False)

            phase = torch.rand(motion_ids[mask].shape, device=self._device)
            
            motion_len = self._motion_lengths[motion_ids[mask]]
            if (truncate_time is not None):
                assert (truncate_time >= 0.0)
                motion_len -= truncate_time

            motion_time = phase * motion_len

            # check if it is within the skipped range
            skipped_range = self._motion_rsi_skipped_ranges[motion_ids[mask]]

            curr_num_samples = motion_time.shape[0]
            curr_succ = torch.where(
                torch.logical_and(motion_time >= skipped_range[:, 0], motion_time <= skipped_range[:, 1]),
                torch.zeros(curr_num_samples, dtype=torch.bool, device=self._device),
                torch.ones(curr_num_samples, dtype=torch.bool, device=self._device),
            )
            curr_succ_inds = torch.nonzero(mask)[curr_succ, 0]
            succ_sample_times[curr_succ_inds] = motion_time[curr_succ]
            succ_sample_record[curr_succ_inds] = True

        return succ_sample_times
    
    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert(truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def get_motion_state(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = self._key_body_ids.shape[0]

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]

        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        root_vel = self.grvs[f0l]

        root_ang_vel = self.gravs[f0l]
        
        key_pos0 = self.gts[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_pos1 = self.gts[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]

        dof_vel = self.dvs[f0l]

        vals = [root_pos0, root_pos1, local_rot0, local_rot1, root_vel, root_ang_vel, key_pos0, key_pos1]
        for v in vals:
            assert v.dtype != torch.float64


        blend = blend.unsqueeze(-1)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1

        root_rot = torch_utils.slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
        
        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof(local_rot)

        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos
    
    def get_motion_state_max(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = self._key_body_ids.shape[0]

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        body_pos0 = self.gts[f0l]
        body_pos1 = self.gts[f1l]
        body_rot0 = self.grs[f0l]
        body_rot1 = self.grs[f1l]
        body_vel = self.gvs[f0l]
        body_ang_vel = self.gavs[f0l]

        vals = [body_pos0, body_pos1, body_rot0, body_rot1, body_vel, body_ang_vel]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)
        blend_exp = blend.unsqueeze(-1)

        # linear interpolation in maximal coordinate may not give the right values, see https://github.com/nv-tlabs/ASE/pull/53
        body_pos = (1.0 - blend_exp) * body_pos0 + blend_exp * body_pos1
        body_rot = torch_utils.slerp(body_rot0, body_rot1, blend_exp)

        return body_pos, body_rot, body_vel, body_ang_vel

    def get_obj_motion_state(self, motion_ids, motion_times):

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.obj_gts[f0l]
        root_pos1 = self.obj_gts[f1l]

        root_rot0 = self.obj_grs[f0l]
        root_rot1 = self.obj_grs[f1l]


        vals = [root_pos0, root_pos1]
        for v in vals:
            assert v.dtype != torch.float64


        blend = blend.unsqueeze(-1)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1

        root_rot = torch_utils.slerp(root_rot0, root_rot1, blend)

        return root_pos, root_rot

    def _load_motions(self, motion_dict):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []
        self._motion_rsi_skipped_ranges = []
        self._motion_contact_ranges = []
        self._obj_motions = []

        total_len = 0.0
        motion_files = motion_dict['motion_files']
        motion_weights = motion_dict['motion_weights']
        motion_rsi_skipped_ranges = motion_dict['motion_info']['rsi_skipped_range']
        motion_contact_ranges = motion_dict['motion_info']['contact_range']
        object_files = motion_dict.get('object_files', [])

        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            curr_motion = SkeletonMotion.from_file(curr_file)

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
 
            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels

            # Moving motion tensors to the GPU
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)                
            else:
                curr_motion.tensor = curr_motion.tensor.to(self._device)
                curr_motion._skeleton_tree._parent_indices = curr_motion._skeleton_tree._parent_indices.to(self._device)
                curr_motion._skeleton_tree._local_translation = curr_motion._skeleton_tree._local_translation.to(self._device)
                curr_motion._rotation = curr_motion._rotation.to(self._device)

            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)
            
            curr_weight = motion_weights[f]
            self._motion_weights.append(curr_weight)
            self._motion_files.append(curr_file)

            curr_rsi_skipped_ranges = [frame_id / num_frames * curr_len for frame_id in motion_rsi_skipped_ranges[f]] # times
            self._motion_rsi_skipped_ranges.append(curr_rsi_skipped_ranges)

            contact_ranges = [frame_id / num_frames * curr_len for frame_id in motion_contact_ranges[f]]
            self._motion_contact_ranges.append(contact_ranges)

            # load obj motion data
            if len(object_files) > 0:
                curr_obj_file = object_files[f]
                curr_obj_motion = torch.tensor(np.load(curr_obj_file), device=self._device, dtype=torch.float32)
                self._obj_motions.append(curr_obj_motion)

        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)

        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self._device)
        self._motion_weights /= self._motion_weights.sum()

        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)

        self._motion_rsi_skipped_ranges = torch.tensor(self._motion_rsi_skipped_ranges, device=self._device, dtype=torch.float32)
        self._motion_contact_ranges = torch.tensor(self._motion_contact_ranges, device=self._device, dtype=torch.float32)

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        return

    def _calc_frame_blend(self, time, len_, num_frames, dt):

        phase = time / len_
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        motion = self.get_motion(0)
        num_bodies = motion.num_joints
        return num_bodies

    def _compute_motion_dof_vels(self, motion):
        num_frames = motion.tensor.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            frame_dof_vel = frame_dof_vel
            dof_vels.append(frame_dof_vel)
        
        dof_vels.append(dof_vels[-1])
        dof_vels = torch.stack(dof_vels, dim=0)

        return dof_vels
    
    def _local_rotation_to_dof(self, local_rot):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self._num_dof), dtype=torch.float, device=self._device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_q = local_rot[:, body_id]
                joint_exp_map = torch_utils.quat_to_exp_map(joint_q)
                dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_exp_map
            elif (joint_size == 1):
                joint_q = local_rot[:, body_id]
                joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                joint_theta = joint_theta * joint_axis[..., 1] # assume joint is always along y axis

                joint_theta = normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        dof_vel = torch.zeros([self._num_dof], device=self._device)

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        local_vel = local_vel

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel

            elif (joint_size == 1):
                assert(joint_size == 1)
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset] = joint_vel[1] # assume joint is always along y axis

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_vel

    def get_motion_label(self, motion_ids):
        return self._motion_labels[motion_ids]

    def sample_motion_labels(self, n):
        motion_ids = self.sample_motions(n)
        return self.get_motion_label(motion_ids)
    
    def set_motion_labels(self, motion_labels):
        self._motion_labels = motion_labels
        return


class MotionLibCLIP(MotionLib):
    def __init__(self, motion_dict, dof_body_ids, dof_offsets, key_body_ids, device, extract_from_text=False, motion_encoder=None, max_label_per_motion=1, is_test=False):
        super().__init__(motion_dict, dof_body_ids, dof_offsets, key_body_ids, device, is_test=is_test)
        if extract_from_text:
            self.clip_model = CLIPExtractor(device=device)
        else:
            self.clip_model = None
        self.extract_from_text = extract_from_text
        clip_features, captions = self._fetch_clip_files(motion_dict)
        if motion_encoder:
            motion_encoder = motion_encoder.to(device)
            condition_dim = motion_encoder.latent_dim
        else:
            condition_dim = 512
        self._motion_info_dict = motion_dict['motion_info']
        _motion_labels = []
        for i, clip_feature in enumerate(clip_features):
            curr_caption = captions[i]

            if motion_encoder:
                curr_motion_label = motion_encoder.encode_text(clip_feature)
                if len(curr_motion_label) > max_label_per_motion:
                    curr_motion_label = curr_motion_label[:max_label_per_motion]
                    curr_caption = curr_caption[:max_label_per_motion]
                elif len(curr_motion_label) < max_label_per_motion:
                    curr_motion_label = torch.cat([curr_motion_label, curr_motion_label[0].unsqueeze(0).repeat(max_label_per_motion - len(curr_motion_label), 1)], dim=0)
                    curr_caption = curr_caption + curr_caption[:max_label_per_motion - len(curr_caption)]
            else:
                if len(clip_feature) > max_label_per_motion:
                    clip_feature = clip_feature[:max_label_per_motion]
                    curr_caption = curr_caption[:max_label_per_motion]
                elif len(clip_feature) < max_label_per_motion:
                    clip_feature = torch.cat([clip_feature, clip_feature[0].unsqueeze(0).repeat(max_label_per_motion - len(clip_feature), 1)], dim=0)
                    curr_caption = curr_caption + curr_caption[:max_label_per_motion - len(curr_caption)]
                curr_motion_label = clip_feature
            _motion_labels.append(curr_motion_label)
            captions[i] = curr_caption
        self._motion_labels = torch.stack(_motion_labels, dim=0) # N, max_label_per_motion, latent_dim
        if not is_test:
            assert self._motion_labels.shape[0] == len(self._motions)
            assert self._motion_labels.shape[1] == max_label_per_motion
            assert self._motion_labels.shape[2] == condition_dim
        self.captions = captions
        if is_test:
            self._motion_weights = torch.ones(len(self._motion_labels), device=device) / len(self._motion_labels)
        self.motion_encoder = motion_encoder
        self.max_label_per_motion = max_label_per_motion
        # assert isinstance(self._motion_labels , list)

        self.allowed_obj_types = set()
        for obj_range in self._motion_info_dict['obj_range']:
            for obj_type in obj_range:
                self.allowed_obj_types.add(obj_type)
        self.allowed_obj_indices = dict()
        for obj_type in self.allowed_obj_types:
            for i, obj_range in enumerate(self._motion_info_dict['obj_range']):
                if obj_type in obj_range:
                    if obj_type not in self.allowed_obj_indices:
                        self.allowed_obj_indices[obj_type] = []
                    self.allowed_obj_indices[obj_type].append(i)
            
    def _fetch_clip_files(self, motion_dict):
        clip_feature_all = []
        caption_all = []

        motion_info = motion_dict['motion_info']
        for i in range(len(motion_info['captions'])):

            if 'captions' in motion_info:
                caption = motion_info['captions'][i]
                # caption = caption[0] if isinstance(caption, list) else caption
                caption_all.append(caption)
            else:
                caption_all.append([' '])

            if 'clip_file' in motion_info:
                curr_file = motion_info['clip_file'][i]
            else:
                curr_file = ''
            if self.extract_from_text:
                clip_feature = self.clip_model.encode_text(caption_all[-1]).view(-1, 512)
            elif os.path.isfile(curr_file) and curr_file:
                clip_info = np.load(curr_file, allow_pickle=True)
                clip_embedding = clip_info
                clip_feature = torch.from_numpy(clip_embedding).to(self._device).view(-1, 512)
            else:
                raise FileNotFoundError("Missing CLIP embedding: {}. Precompute embeddings or enable text extraction.".format(curr_file))
            clip_feature_all.append(clip_feature)

        return clip_feature_all, caption_all

    def get_motion_label(self, motion_ids, return_text=False):
        # sampled_motion_labels = []
        captions = []
        if self.max_label_per_motion == 1:
            sampled_motion_labels = self._motion_labels[motion_ids, 0]
        else:
            random_index = torch.randint(low=0, high=self.max_label_per_motion, size=(len(motion_ids),))
            sampled_motion_labels = self._motion_labels[motion_ids, random_index]
        for idx in motion_ids.view(-1):
            # curr_motion_labels = self._motion_labels[idx]
            # # sample one from current motion labels
            # if isinstance(self._motion_labels, list):
            #     if len(curr_motion_labels) > 1:
            #         random_idx = torch.randint(0, len(curr_motion_labels), (1,))
            #         sampled_motion_labels.append(curr_motion_labels[random_idx])
            #         caption = self.captions[idx][random_idx]
            #     else:
            #         sampled_motion_labels.append(curr_motion_labels[0])
            #     if isinstance(caption, list):
            #         caption = caption[0]
            # else:
            #     sampled_motion_labels.append(curr_motion_labels.view(1, self.motion_encoder.latent_dim))
            captions.append(self.captions[idx])
        # sampled_motion_labels = torch.cat(sampled_motion_labels, dim=0)
        if return_text:
            return sampled_motion_labels, captions
        else:
            return sampled_motion_labels
    
    def get_motion_info(self, motion_ids):
        motion_info = [self._motion_info_dict[i] for i in motion_ids]
        return motion_info

    def get_motion_captions(self, motion_ids=None):
        captions = self.captions
        if motion_ids is not None:
            captions = [captions[i] for i in motion_ids]
        return captions
    
    def get_obj_contact(self, motion_ids=None):
        obj_contact = self._motion_info_dict['obj_contact']
        if motion_ids is not None:
            obj_contact = [obj_contact[i] for i in motion_ids]
        return obj_contact
    
    def get_self_contact(self, motion_ids=None):
        self_contact = self._motion_info_dict['self_contact']
        if motion_ids is not None:
            self_contact = [self_contact[i] for i in motion_ids]
        return self_contact

    ############################################################
    def unique_emotions(self):
        return list(set(self._motion_info_dict['emotion/style']))
    
    def unique_actions(self):
        return list(set(self._motion_info_dict['actions']))
    
    def unique_obj_ranges(self):
        return list(set(self._motion_info_dict['obj_range']))
    
    def emotion_action_pairs(self, motion_ids=None):
        emotions = self._motion_info_dict['emotion/style']
        actions = self._motion_info_dict['actions']
        indices = np.arange(len(emotions)) if motion_ids is None else motion_ids
        return [dict(index=i, emotion_style=emotions[i], action=actions[i]) for i in indices]
    
    def sample_motions_obj_range(self, n, obj_types=None):
        if obj_types is None:
            return self.sample_motions(n)
        obj_ranges = self._motion_info_dict['obj_range']
        indices = []
        if isinstance(obj_types, str):
            for i, obj_range in enumerate(obj_ranges):
                if obj_types in obj_range:
                    indices.append(i)
            motion_weights = torch.zeros_like(self._motion_weights)
            motion_weights[indices] = self._motion_weights[indices]
            motion_weights /= motion_weights.sum()
            motion_ids = torch.multinomial(motion_weights, num_samples=n, replacement=True)
        elif isinstance(obj_types, list):
            assert len(obj_types) == n
            motion_ids = torch.zeros(n, dtype=torch.long, device=self._device)
            for j in range(len(obj_types)):
                motion_ids[j] = np.random.choice(self.allowed_obj_indices[obj_types[j]])

        return motion_ids

    def sample_motion_labels_obj_range(self, n, obj_types=None, given_ids=None):
        if given_ids is not None:
            motion_ids = given_ids
            motion_ids = motion_ids % len(self.captions)
        else:
            motion_ids = self.sample_motions_obj_range(n, obj_types)

        motion_labels =  self.get_motion_label(motion_ids)
        self_contact = self.get_self_contact(motion_ids)
        obj_contact = self.get_obj_contact(motion_ids)
        captions = self.get_motion_captions(motion_ids)
        return motion_labels, self_contact, obj_contact, captions
    
    def sample_motions_labels(self, n):
        return self.sample_motion_labels_obj_range(n, None)
    
def concat_dict_list(list_of_dict): # list of dict to dict of list
    output = dict()
    keys = list_of_dict[0].keys()
    for k in keys:
        if isinstance(list_of_dict[0][k], dict):
            output[k] = concat_dict_list(
                [curr_dict[k] for curr_dict in list_of_dict])
        else:
            selected_items = [curr_dict[k] for curr_dict in list_of_dict]
            output[k] = selected_items
    return output

class MotionLibDict():
    def __init__(self, motion_files, dof_body_ids, dof_offsets,
                 key_body_ids, device, motion_encoder=None, allowed_skills=None, condition_dim=64, skill_weights=None, is_test=False, extract_from_text=False):
        self._device = device
        self.is_test = is_test
        if not isinstance(motion_files, list):
            motion_files = [motion_files]
        self._motion_lib_dict = dict()
        self.condition_dim = condition_dim
        motion_dict_skills = dict()
        for motion_file in motion_files:
            curr_motion_dict = self._fetch_motion_files(motion_file)
            for skill in curr_motion_dict.keys():
                if skill in motion_dict_skills:
                    motion_dict_skills[skill]['motion_weights'] += curr_motion_dict[skill]['motion_weights']
                    motion_dict_skills[skill]['motion_files'] += curr_motion_dict[skill]['motion_files']
                    motion_dict_skills[skill]['motion_info'] += curr_motion_dict[skill]['motion_info']
                    motion_dict_skills[skill]['object_files'] += curr_motion_dict[skill]['object_files']
                else:
                    motion_dict_skills[skill] = curr_motion_dict[skill]
        if allowed_skills is not None:
            motion_dict_skills = {k: v for k, v in motion_dict_skills.items() if k in allowed_skills}
            self._skills = allowed_skills
        else:
            self._skills = list(motion_dict_skills.keys())

        allowed_skill_obj_mapping = dict()

        for skill in self._skills:
            if isinstance(motion_encoder, dict):
                motion_enc = motion_encoder[skill]
            else:
                motion_enc = motion_encoder
            motion_dict = motion_dict_skills[skill]
            motion_dict['motion_info'] = concat_dict_list(motion_dict['motion_info'])
            motion_lib = MotionLibCLIP(motion_dict, dof_body_ids, dof_offsets, key_body_ids, device, extract_from_text=extract_from_text, motion_encoder=motion_enc, is_test=is_test)
            allowed_skill_obj_mapping[skill] = motion_lib.allowed_obj_types
            self._motion_lib_dict[skill] = motion_lib
        self.allowed_obj_skill_mapping = dict()
        for skill, objs in allowed_skill_obj_mapping.items():
            for obj in objs:
                if obj not in self.allowed_obj_skill_mapping:
                    self.allowed_obj_skill_mapping[obj] = set()
                self.allowed_obj_skill_mapping[obj].add(skill)

        if skill_weights is not None:
            assert len(skill_weights) == len(self._skills)
        _skill_weights = []    
        for skill in self._skills:
            if skill_weights is not None:
                _skill_weights.append(skill_weights[skill])
            else:
                _skill_weights.append(1.0)
        self._skill_weights = torch.tensor(_skill_weights, device=device)
        self._skill_weights = self._skill_weights / self._skill_weights.sum()

    def keys(self):
        return self._motion_lib_dict.keys()
    
    def __call__(self, skill):
        return self._motion_lib_dict[skill]
    
    @classmethod
    def _fetch_motion_files(cls, motion_file):
        ext = os.path.splitext(motion_file)[1]
        assert (ext == ".yaml")
        dir_name = os.path.dirname(motion_file)

        with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
            motion_config = yaml.load(f, Loader=yaml.SafeLoader)
        motion_dict = dict()
        skills = motion_config['motions'].keys()
        for sk_name in skills:
            motion_dict_curr_skill = dict(motion_weights=[], motion_files=[], motion_info=[], object_files=[])
            motion_list = motion_config['motions'][sk_name]
            for motion_entry in motion_list:
                curr_file = motion_entry['file']
                curr_weight = motion_entry['weight']
                if 'object_file' in motion_entry:
                    obj_file = motion_entry['object_file']
                elif 'obj_file' in motion_entry:
                    obj_file = motion_entry['obj_file']
                else:
                    obj_file = ''
                assert(curr_weight >= 0)

                curr_file = os.path.join(dir_name, curr_file)
                motion_dict_curr_skill['motion_weights'].append(curr_weight)
                motion_dict_curr_skill['motion_files'].append(curr_file)
                if obj_file:
                    obj_file = os.path.join(dir_name, obj_file)
                    motion_dict_curr_skill['object_files'].append(obj_file)

                curr_rsi_skipped_range = motion_entry.get('rsi_skipped_range', [])
                if len(curr_rsi_skipped_range) == 0:
                    curr_rsi_skipped_range = [np.inf, -np.inf]
                else:
                    assert len(curr_rsi_skipped_range) == 2
                    assert curr_rsi_skipped_range[0] < curr_rsi_skipped_range[1]

                contact_range = motion_entry.get('contact_range', [])
                if len(contact_range) == 0:
                    contact_range = [np.inf, -np.inf]
                else:
                    assert len(contact_range) == 2
                    assert contact_range[0] < contact_range[1]

                actions = motion_entry.get('action', '') # string
                emotion_style = motion_entry.get('emotion/style', '') # string
                captions = motion_entry.get('captions', []) # list
                clip_file = motion_entry.get('clip_file', '') # string
                obj_contact = motion_entry.get('obj_contact', []) # list
                self_contact = motion_entry.get('self_contact', []) # list
                obj_range = motion_entry.get('obj_range', []) # list
                clip_file = os.path.join(dir_name, clip_file)
                info = {'actions': actions, 'emotion_style': emotion_style, 'captions': captions, 'clip_file': clip_file, 'obj_contact': obj_contact, 'contact_range': contact_range, 'rsi_skipped_range': curr_rsi_skipped_range, 'self_contact': self_contact, 'obj_range': obj_range, 'skill': sk_name}
                motion_dict_curr_skill['motion_info'].append(info)

            motion_dict[sk_name] = motion_dict_curr_skill

        return motion_dict

    def __getitem__(self, key):
        return self._motion_lib_dict[key]
    
    def __setitem__(self, key, value):
        assert isinstance(value, MotionLib)
        self._motion_lib_dict[key] = value

    def sample_skills(self, n, skill_weight_override=None):
        if skill_weight_override is not None:
            assert len(skill_weight_override) == len(self._skills)
            skill_weights = torch.tensor(skill_weight_override, device=self._device)
            skill_weights = skill_weights / skill_weights.sum()
        else:
            skill_weights = self._skill_weights
        skill_ids = torch.multinomial(skill_weights, num_samples=n, replacement=True)
        return skill_ids

    def sample_skills_obj(self, n, obj_types, skill_weight_override=None):
        if skill_weight_override is not None:
            assert len(skill_weight_override) == len(self._skills)
            skill_weights = torch.tensor(skill_weight_override, device=self._device)
            skill_weights = skill_weights / skill_weights.sum()
        else:
            skill_weights = self._skill_weights.clone()

        if isinstance(obj_types, str):
            allowed_skills = self.allowed_obj_skill_mapping[obj_types]
            allowed_skills_id = [self._skills.index(s) for s in allowed_skills]
            # set weight of skills not in allowed_skills to 0
            for i in range(len(skill_weights)):
                if i not in allowed_skills_id:
                    skill_weights[i] = 0
            skill_weights = skill_weights / skill_weights.sum()
            skill_ids = torch.multinomial(skill_weights, num_samples=n, replacement=True)
        elif isinstance(obj_types, list):
            # there are only several obj_types, split by obj_type and sample from each
            unique_obj_types = set(obj_types)
            skill_ids = torch.zeros(n, dtype=torch.long, device=self._device)
            for obj_type in unique_obj_types:
                allowed_skills = self.allowed_obj_skill_mapping[obj_type]
                allowed_skills_id = [self._skills.index(s) for s in allowed_skills]
                # set weight of skills not in allowed_skills to 0
                for i in range(len(skill_weights)):
                    if i not in allowed_skills_id:
                        skill_weights[i] = 0
                skill_weights = skill_weights / skill_weights.sum()
                sampled_curr_skill_ids = torch.multinomial(skill_weights, num_samples=obj_types.count(obj_type), replacement=True)
                curr_ids = [i for i, x in enumerate(obj_types) if x == obj_type]
                skill_ids[curr_ids] = sampled_curr_skill_ids
                
        return skill_ids
    
    def sample_motion_labels(self, n):
        skill_ids = self.sample_skills(n).int().to(self._device)
        sampled_labels = torch.zeros((n, self.condition_dim), device=self._device)

        for sk_id in torch.unique(skill_ids):
            motion_lib = self._motion_lib_dict[self._skills[int(sk_id)]]
            curr_ids = (skill_ids == sk_id)
            labels = motion_lib.sample_motion_labels(curr_ids.sum())
            sampled_labels[curr_ids] = labels
        return skill_ids, sampled_labels

    def sample_motion_labels_obj_range(self, n, obj_types=None, skill_weight_override=None, given_ids=None):
        # sample motion labels, self_contact, obj_contact based on obj_type
        if obj_types is None:
            skill_ids = self.sample_skills(n, skill_weight_override=skill_weight_override).int().to(self._device)
        else:
            skill_ids = self.sample_skills_obj(n, obj_types, skill_weight_override=skill_weight_override).int().to(self._device)
        sampled_labels = torch.zeros((n, self.condition_dim), device=self._device)
        self_contact = [None] * n
        obj_contact = [None] * n
        captions = [None] * n

        for sk_id in torch.unique(skill_ids):

            motion_lib = self._motion_lib_dict[self._skills[int(sk_id)]]
            curr_ids = (skill_ids == sk_id)
            curr_skill_name = self._skills[int(sk_id)]
            if obj_types is not None:
                if isinstance(obj_types, str):
                    obj_types_skill = obj_types
                elif isinstance(obj_types, list):
                    obj_types_skill = [obj_types[i] for i in curr_ids.nonzero().view(-1)]
            else:
                obj_types_skill = None
            labels, self_contact_curr, obj_contact_curr, captions_curr = motion_lib.sample_motion_labels_obj_range(curr_ids.sum(), obj_types_skill, given_ids=given_ids)

            sampled_labels[curr_ids] = labels
            for idx, curr_idx in enumerate(curr_ids.nonzero().view(-1)):
                self_contact[curr_idx] = self_contact_curr[idx]
                obj_contact[curr_idx] = obj_contact_curr[idx]
                captions[curr_idx] = captions_curr[idx]

        return skill_ids, sampled_labels, self_contact, obj_contact, captions
    
    def sample_motion_labels_skill(self, skill, n):
        if isinstance(skill, str):
            skill_ids = torch.tensor([self._skill.index(skill)]*n).int().to(self._device)
            sampled_labels = self._motion_lib_dict[skill].sample_motion_labels(n)

        elif isinstance(skill, list):
            skill_indexes = [self._skill.index(s) for s in skill]
            skill_ids = torch.multinomial(self._skill_weights[skill_indexes], num_samples=n, replacement=True)
            skill_ids = torch.Tensor(skill_indexes)[skill_ids.int()].int().to(self._device)
            sampled_labels = torch.zeros((n, self.condition_dim), device=self._device)
            for sk_id in torch.unique(skill_ids):
                motion_lib = self._motion_lib_dict[self._skills[int(sk_id)]]
                curr_ids = (skill_ids == sk_id)
                labels = motion_lib.sample_motion_labels(len(curr_ids))
                sampled_labels[curr_ids] = labels
        return skill_ids, sampled_labels
    

class MotionLibDictTest:
    def __init__(self, skills):
        self._skills = skills
