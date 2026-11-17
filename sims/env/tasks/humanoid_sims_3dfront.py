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

import os
import yaml
import math
from enum import Enum
import numpy as np
import copy
import torch
import json
import re
import open3d as o3d

from isaacgym import gymapi
from isaacgym import gymtorch

from sims.env.tasks.humanoid import Humanoid, dof_to_obs
from sims.utils import gym_util
from sims.utils.motion_lib_dict import MotionLibDict
from scipy.spatial.transform import Rotation as scipyR
from isaacgym.torch_utils import *
from sims.learning.sims_motion_enc_model import MotionCLIP

from sims.utils import torch_utils


def get_obb(mesh):
    obb = mesh.get_axis_aligned_bounding_box()
    corners = np.array(obb.get_box_points())
    corners = np.stack([corners[0], corners[1], corners[7], corners[2], corners[5], corners[3], corners[6], corners[4]], 0)
    corners[:4, 2] = 0
    return corners

def load_label(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        label = np.array([int(line) for line in lines], dtype=np.int32)
        return label
    
def get_leaf_node(dic, results, offset):
    for i in range(len(results)):
        result = results[i]
        if 'children' in result.keys():
            get_leaf_node(dic, result['children'], offset)
        else:
            dic[result['name']+str(result['id'])] = result['id'] + offset

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def get_height_map(points: np.ndarray, HEIGHT_MAP_DIM: int=16):
    """ Load region meshes of a scenes

    Args:
        points: scene point cloud
        HEIGHT_MAP_DIM: height map dimension
    
    Return:
        Return the floor height map and axis-aligned scene bounding box
    """

    ## compute floor height map
    minx, miny = points[:, 0].min(), points[:, 1].min()
    maxx, maxy = points[:, 0].max(), points[:, 1].max()

    interval_x = (maxx-minx)/HEIGHT_MAP_DIM
    interval_y = (maxy-miny)/HEIGHT_MAP_DIM
    voxel_idx_x = (points[:, 0]-minx) // interval_x
    voxel_idx_y = (points[:, 1]-miny) // interval_y

    height_map2d = np.zeros((HEIGHT_MAP_DIM,HEIGHT_MAP_DIM))
    for i in range(HEIGHT_MAP_DIM):
        for j in range(HEIGHT_MAP_DIM):
            mask = (voxel_idx_x==i)&(voxel_idx_y==j)
            if mask.sum()==0:
                pass
            else:
                height_map2d[j,i] = points[mask, 2].max()

    x = np.linspace(minx, maxx, HEIGHT_MAP_DIM)
    y = np.linspace(miny, maxy, HEIGHT_MAP_DIM)
    xx, yy = np.meshgrid(x, y)
    # pos2d = np.concatenate([yy[..., None], xx[..., None]], axis=-1)
    pos2d = np.concatenate([xx[..., None], yy[..., None]], axis=-1)

    height_pcd = np.concatenate([pos2d.reshape(-1,2),height_map2d.reshape(-1,1)], axis=-1)
    # >=0
    height_pcd[:, 2] = np.clip(height_pcd[:, 2], 0., 2.0)
    return height_pcd

def VaryPoint(data, axis, degree):
    xyzArray = {
        'X': np.array([[1, 0, 0],
                  [0, np.cos(np.radians(degree)), -np.sin(np.radians(degree))],
                  [0, np.sin(np.radians(degree)), np.cos(np.radians(degree))]]),
        'Y': np.array([[np.cos(np.radians(degree)), 0, np.sin(np.radians(degree))],
                  [0, 1, 0],
                  [-np.sin(np.radians(degree)), 0, np.cos(np.radians(degree))]]),
        'Z': np.array([[np.cos(np.radians(degree)), -np.sin(np.radians(degree)), 0],
                  [np.sin(np.radians(degree)), np.cos(np.radians(degree)), 0],
                  [0, 0, 1]])}
    newData = np.dot(data, xyzArray[axis])
    return newData


class HumanoidSIMS(Humanoid):
    
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    class ObjectType(Enum):
        chair = 0
        armchair = 1
        bed = 2
        stool = 3
        sofa = 4
        table = 5
        desk = 6
        stand = 7
        shelf = 8
        cabinet = 9
        lamp = 10
        wardrobe = 11
    
    object_index = np.array(['chair', 'armchair','bed', 'stool', 'sofa', 'table','desk',  'stand', 'shelf', 'cabinet', 'lamp', 'wardrobe'])
    joint_name = ["pelvis", "left_hip", "left_knee", "left_foot", "right_hip", "right_knee", "right_foot", "torso", 
        "head", "left_shoulder", "left_elbow", "left_hand", "right_shoulder", "right_elbow", "right_hand"]
    joint_mapping = {"pelvis":0, "left_hip":1, "left_knee":2, "left_foot":3, "right_hip":4, "right_knee":5, "right_foot":6, "torso":7, 
        "head":8, "left_shoulder":9, "left_elbow":10, "left_hand":11, "right_shoulder":12, "right_elbow":13, "right_hand":14}

    contact_joint_mapping = {"pelvis": 0, "head": 1, "left_foot": 2, "right_foot": 3, "left_hand": 4, "right_hand": 5}
    
    joint_abbr = {'pelvis': 'pelvis', 'lhip': 'left_hip', 'lknee': 'left_knee', 'lfoot': 'left_foot', 'rhip': 'right_hip', 'rknee': 'right_knee', 'rfoot': 'right_foot', 'torso': 'torso', 'head': 'head', 'lshoulder': 'left_shoulder', 'lelbow': 'left_elbow', 'lhand': 'left_hand', 'rshoulder': 'right_shoulder', 'relbow': 'right_elbow', 'rhand': 'right_hand'}

    task_contact = {
        'sitdown': 
            {'chair':['pelvis-seat-up'], 
             'armchair': ['pelvis-seat-up'], 
             'bed': ['pelvis-seat-up'], 
             'stool': ['pelvis-seat-up'],
             'table': ['pelvis-seat-up'],
             'sofa': ['pelvis-seat-up'],
             'desk': ['pelvis-seat-up'],
             'stand': ['pelvis-seat-up']}, 
        'liedown': 
            {'bed': ['pelvis-seat-up', 'head-seat-up', ], 
             'sofa': ['pelvis-seat-up'],
             'table': ['pelvis-seat-up', 'head-seat-up', ],
             },
             }
    direction_mapping = {
        "up":[0,0,1],
        "down":[0,0,-1],
        "left":[0,1,0],
        "right":[0,-1,0],
        "front":[1,0,0],
        "back":[-1,0,0],
        "none":[0,0,0]
    }
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        self._is_train = cfg["args"].train
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidSIMS.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        strike_body_names = cfg["env"]["strikeBodyNames"]

        self.contact_joint_num = len(strike_body_names)
        #####################################################################
        self._enable_conditional_disc = cfg["env"]['condition']["enableConditionalDisc"]

        self._num_motion_condition_obs = cfg["env"]['condition']["numConditionDim"]
        self._extract_text_features = cfg["env"]['condition'].get("extractTextFeatures", False)

        #####################################################################
        # load objects
        obj_info_category = dict()
        obj_info_path = cfg['env']['obj']['objFile']
        if isinstance(obj_info_path, str):
            obj_info_path = [obj_info_path]
        self.allowed_obj_categories = cfg['env']['obj'].get('allowedObjs', None)
        for path in obj_info_path:
            with open(path) as f:
                curr_dict = json.load(f)
            obj_info_category.update(curr_dict)

        self.obj_info_category = dict()
        if self.allowed_obj_categories is not None:
            for k in self.allowed_obj_categories:
                self.obj_info_category[k] = obj_info_category[k]
        else:
            self.obj_info_category = obj_info_category
        num_obj = 0
        num_obj_dict = dict()
        for k in self.obj_info_category:
            num_obj += len(self.obj_info_category[k])
            num_obj_dict.setdefault(k, 0)
            num_obj_dict[k] += len(self.obj_info_category[k])
        
        self.num_obj = num_obj
        self.num_obj_dict = num_obj_dict
        self.num_scenes_row = cfg['env']['obj']['numRow']
        self.num_scenes_col = cfg['env']['obj']['numCol']
        self.obj_radius_min = cfg['env']['obj']['objRadiusMin']
        self.obj_radius_max = cfg['env']['obj']['objRadiusMax']
        self.standpoint_scale = cfg['env']['obj'].get('standpointScale', 1.0)
        self.num_obj_per_scene = cfg['env']['obj']['numObjPerScene']
        self.obj_angle_offset = cfg['env']['obj']['objAngleOffset']
        self.num_max_children = cfg['env']['obj']['numMaxChildren']
        self.obj_htmp_scale = cfg['env']['obj'].get('objHtmpScale', 12)
        self.is_touch = cfg['env']['obj'].get('isTouch', False)
        self.obj_scale = cfg['env']['obj'].get('objScale', 1)

        #####################################################################
        self.spacing = cfg["env"]["envSpacing"]
        self.with_heightmap = cfg['env']['obj'].get('withHeightmap', True)
        self.height_map_dim = cfg['env']['obj']['heightMapDim']
        self.plan_number = cfg['env']['obj']['numPlans']
        self.env_array = torch.arange(0, cfg["env"]["numEnvs"], device=device_type, dtype=torch.float)
        #####################################################################
        # height map
        self.local_scale = 12
        self.local_interval = 0.15
        self.obj_interval = 0.15
        self.num_pcd_buffer = 400
        #####################################################################
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        #####################################################################
        # motion encoder
        self.motion_enc = MotionCLIP()
        self.motion_enc.load_state_dict(torch.load(cfg['env']['condition']['motion_enc_ckpt']))
        self.motion_enc.to(self.device)
        self.motion_enc.eval()
        for param in self.motion_enc.parameters():
            param.requires_grad = False
        if self._enable_conditional_disc:
            assert self.motion_enc.latent_dim == self._num_motion_condition_obs
        #####################################################################
        # skills
        self._skills = cfg["env"]['skill']["skill"]
        self._skill_disc_prob = torch.tensor(cfg["env"]['skill']["skillDiscProb"], device=self.device, dtype=torch.float) # probs for amp obs demo fetch
        self._skill_init_prob = torch.tensor(cfg["env"]['skill']["skillInitProb"], device=self.device, dtype=torch.float) # probs for amp obs demo fetch
        self._skill_weights = dict(zip(self._skills, cfg["env"]['skill']["skillSampleProb"]))
        motion_file = cfg['env']['motion_file']
        self._every_env_skill_id = torch.multinomial(torch.Tensor(cfg["env"]['skill']["skillSampleProb"]), num_samples=self.num_envs, replacement=True).to(self.device).int()
        self._load_motion(motion_file,)
        if self._is_train:
            assert self._skills == self._motion_lib_dict._skills
        #####################################################################
        # amp
        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None
        #####################################################################
        # if self._enable_conditional_disc:
        self._every_env_motion_label = torch.zeros((self.num_envs, self._num_motion_condition_obs), device=self.device, dtype=torch.float)

        if (not self.headless):
            self._build_marker_state_tensors()

        #####################################################################
        # tensors for enableTrackInitState

        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
    
        self._strike_body_ids = self._build_strike_body_ids_tensor(self.envs[0], self.humanoid_handles[0], strike_body_names)

        self.env_contact_valid = torch.zeros([self.num_envs, self.contact_joint_num], device=self.device, dtype=torch.bool)
        #####################################################################
        # task
        self._enable_task_update = cfg["env"]['task']["enableTaskUpdate"]
        self._enable_contact = cfg["env"]['task']["enableContact"]

        if self._enable_task_update:
            self.location_diff_buf = torch.ones([self.num_envs], device=self.device, dtype=torch.float)
        else:
            self.location_diff_buf = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
            
        if self._enable_task_update and self._enable_contact:
            self.joint_diff_buff = torch.ones([self.num_envs, self.contact_joint_num], device=self.device, dtype=torch.float)
            self.joint_idx_buff = torch.ones([self.num_envs, self.contact_joint_num], device=self.device, dtype=torch.long)
        else:
            self.joint_diff_buff = torch.zeros([self.num_envs, self.contact_joint_num], device=self.device, dtype=torch.float)
            self.joint_idx_buff = torch.zeros([self.num_envs, self.contact_joint_num], device=self.device, dtype=torch.long)

        self.tar_dir = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        self.stand_point_choice = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)
        self.stand_point_chosen = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        #####################################################################

        self.pelvis2torso = torch.tensor([0,0,0.236151]).to(self.device)[None].repeat(self.num_envs, 1)
        self.pelvis2torso[:, 2] += 0.15
        self.torso2head = torch.tensor([0, 0, 0.223894]).to(self.device)[None].repeat(self.num_envs, 1)
        self.torso2head[:, 2] += 0.15
        self.new_rigid_body_pos = torch.ones([self.num_envs, 15, 3], device=self.device, dtype=torch.float)

        self.big_force = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        
        self.still = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        self.still_buf = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self.captions = [''] * self.num_envs

        self._kinematic_humanoid_rigid_body_states = torch.zeros((self.num_envs, self.num_bodies, 13), device=self.device, dtype=torch.float)
        self.num_resets = 0
        return
    
    def _build_strike_body_ids_tensor(self, env_ptr, actor_handle, body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids
    
    def generate_plan_items(self):
        plan_items = []
        rand_radius = np.random.rand(self.plan_number, self.num_obj_per_scene) * (self.obj_radius_max - self.obj_radius_min) + self.obj_radius_min
        rand_angle = []
        stride = 360 / self.num_obj_per_scene
        start_angle = np.random.rand(self.plan_number) * stride - stride / 2
        for i in range(self.num_obj_per_scene):
            rand_angle.append(np.random.rand(1) * self.obj_angle_offset + stride * i + start_angle) 
        # N, N, 4
        rand_angle = np.stack(rand_angle, axis=-1)
        rand_angle = np.deg2rad(rand_angle)
        trans_x = rand_radius * np.cos(rand_angle)
        trans_y = rand_radius * np.sin(rand_angle)
        trans_z = np.zeros_like(trans_y)
        translation = np.stack([trans_x, trans_y, trans_z], -1)
        obj_indices = np.random.randint(0, self.num_obj, (self.plan_number, self.num_obj_per_scene))
        for plan_id in range(self.plan_number):
            curr_plan = []
            for j in range(self.num_obj_per_scene):
                obj_choice = obj_indices[plan_id, j]
                pid = self.obj_ids[obj_choice]
                obj_info = {
                    'id': pid,
                    'transfer': translation[plan_id, j],
                }
                curr_plan.append(obj_info)
            plan_items.append(curr_plan)
        return plan_items

    def _create_ground_plane(self):
        self._create_mesh_ground()
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return
    
    def _create_mesh_ground(self):
        if self.num_obj_per_scene > 0:
            self._load_mesh()
            pcd_id, pcd_plan = self._load_pcd()
            self._get_pcd_parts(pcd_id, pcd_plan)

        _x = np.arange(0, self.local_scale)
        _y = np.arange(0, self.local_scale)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()
        mesh_grid = np.stack([x, y, y], axis=-1) # 3rd dim meaningless
        self.mesh_pos = torch.from_numpy(mesh_grid).to(self.device) * self.local_interval
        self.humanoid_in_mesh = torch.tensor([self.local_interval*(self.local_scale-1)/4, self.local_interval*(self.local_scale-1)/2, 0]).to(self.device)

        _x = np.arange(0, self.obj_htmp_scale)
        _y = np.arange(0, self.obj_htmp_scale)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()
        mesh_grid = np.stack([x, y, y], axis=-1) # 3rd dim meaningless
        self.tar_mesh_pos = torch.from_numpy(mesh_grid).to(self.device) * self.obj_interval
        self.tar_in_mesh = torch.tensor([self.obj_interval*(self.obj_htmp_scale-1)/4, self.obj_interval*(self.obj_htmp_scale-1)/2, 0]).to(self.device)

    def _load_mesh(self):
        self._every_env_obj_cls = torch.zeros([self.num_envs, self.num_obj_per_scene], device=self.device, dtype=torch.long)
        self._every_env_children = []
        self._every_env_obj_bps = torch.zeros([self.num_envs, self.num_obj_per_scene, 8, 3], device=self.device, dtype=torch.float)
        self.obj_info = dict()
        self.obj_ids = list()
        for category in self.obj_info_category:
            for pid in self.obj_info_category[category]:
                curr_obj_info = self.obj_info_category[category][pid]
                curr_obj_info['category'] = category
                # if category == 'bed':

                self.obj_info_category[category][pid] = curr_obj_info
                self.obj_info[pid] = curr_obj_info
                path = 'scene_data/3dfront/obj_models/'+pid+'/mesh.obj'
                if os.path.exists(path):
                    mesh = o3d.io.read_triangle_mesh(path)
                else:
                    raise Exception(f"Object {pid} not found")
                theta = - curr_obj_info['rotate']
                R = np.zeros((3, 3)).astype(np.float32)
                R[0, 0] = np.cos(theta)
                R[0, 1] = -np.sin(theta)
                R[1, 0] = np.sin(theta)
                R[1, 1] = np.cos(theta)
                R[2, 2] = 1.
                mesh.rotate(R, center=(0, 0, 0))
                mesh.scale(curr_obj_info['scale'], center=mesh.get_center())
                mesh_vertices_single = np.asarray(mesh.vertices).astype(np.float32())
                scale = curr_obj_info['scale']
                height = mesh_vertices_single[:, 2].max() - mesh_vertices_single[:, 2].min()
                if not isinstance(scale, list):
                    if not self.is_touch:
                        if category == 'bed':
                            curr_obj_info['scale'] = [scale, scale, 0.8*scale]
                        elif category == 'armchair':
                            curr_obj_info['scale'] = [1.2*scale, 1.2*scale, scale]
                        elif category == 'chair' and height * scale > 1:
                            curr_obj_info['scale'] = [scale, scale, 0.8*scale]
                        else:
                            curr_obj_info['scale'] = [scale, scale, scale]
                    else:
                        curr_obj_info['scale'] = [scale, scale, scale*1.5]
                mesh_vertices_single = mesh_vertices_single * np.array(curr_obj_info['scale']).reshape(1, 3).astype(np.float32) * self.obj_scale
                height = mesh_vertices_single[:, 2].max() - mesh_vertices_single[:, 2].min()
                curr_obj_info['height'] = height
                mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices_single)
                mesh.translate((0,0,-mesh_vertices_single[:, 2].min()))
                self.obj_info[pid]['category'] = category
                self.obj_info[pid]['mesh'] = mesh
                self.obj_ids.append(pid)
                self.obj_info_category[category][pid] = curr_obj_info
        self.plan_items = self.generate_plan_items()
        
        mesh_vertices_list = []
        mesh_triangles_list = []
        verts_obj_idx_list = []
        faces_obj_idx_list = []
        bps_plans = np.zeros([self.plan_number, self.num_obj_per_scene, 8, 3])
        for plan_id, curr_plan in enumerate(self.plan_items): # list of dict(id=0, mesh=mesh, transfer=transfer)
            l = 0
            pn = 0
            mesh_vertices = np.zeros([0, 3]).astype(np.float32)
            mesh_triangles = np.zeros([0, 3]).astype(np.uint32)
            verts_obj_idx = np.zeros([0]).astype(np.uint32)
            faces_obj_idx = np.zeros([0]).astype(np.uint32)

            for j in range(len(curr_plan)):
                curr_obj = curr_plan[j]
                pid = curr_obj['id']
                mesh = copy.deepcopy(self.obj_info[pid]['mesh'])
                mesh.translate(curr_obj['transfer']) #  not collision with init human
                # mesh.translate(obj['multi_obj_offset'])
                mesh_vertices_single = np.asarray(mesh.vertices).astype(np.float32())
                mesh_triangles_single = np.asarray(mesh.triangles).astype(np.uint32)

                mesh_vertices = np.concatenate([mesh_vertices, mesh_vertices_single], axis=0)

                verts_obj_idx_single = np.ones([mesh_vertices_single.shape[0]], dtype=np.int32) * j
                verts_obj_idx = np.concatenate([verts_obj_idx, verts_obj_idx_single], axis=0)

                faces_obj_idx_single = np.ones([mesh_triangles_single.shape[0]], dtype=np.int32) * j
                faces_obj_idx = np.concatenate([faces_obj_idx, faces_obj_idx_single], axis=0)


                mesh_triangles = np.concatenate([mesh_triangles, mesh_triangles_single], axis=0)
                mesh_triangles[l:] += pn
                l = mesh_triangles.shape[0]
                pn = mesh_vertices.shape[0]
                bps = get_obb(mesh)
                bps_plans[plan_id, j] = bps

            verts_obj_idx_list.append(verts_obj_idx)
            faces_obj_idx_list.append(faces_obj_idx)
            mesh_vertices_list.append(mesh_vertices)
            mesh_triangles_list.append(mesh_triangles)


        obj_rotate = np.random.rand(self.num_scenes_row, self.num_scenes_col) * 360.0
        obj_rotate_matrix = np.array([[np.cos(np.radians(obj_rotate)), -np.sin(np.radians(obj_rotate)), obj_rotate*0],
                                    [np.sin(np.radians(obj_rotate)), np.cos(np.radians(obj_rotate)), obj_rotate*0],
                                    [obj_rotate*0, obj_rotate*0, obj_rotate*0+1]])

        self.obj_rotate_matrix = torch.from_numpy(obj_rotate_matrix).to(self.device)
        scene_idx = np.random.randint(0, self.plan_number, (self.num_scenes_row, self.num_scenes_col))
        self.scene_idx = torch.from_numpy(scene_idx).to(self.device)
        self.obj_rotate_angle = torch.from_numpy(obj_rotate).to(self.device)
        self.obj_rotate_angle = self.obj_rotate_angle

        dist_max = 2
        dist_min = 1
        rand_dist_x = (dist_max - dist_min) * np.random.rand(self.num_scenes_row, self.num_scenes_col) + dist_min
        rand_dist_y = (dist_max - dist_min) * np.random.rand(self.num_scenes_row, self.num_scenes_col) + dist_min
        rand_dist_x[0] = 0
        rand_dist_y[0] = 0
        self.rand_dist_x = torch.from_numpy(rand_dist_x).to(self.device)
        self.rand_dist_y = torch.from_numpy(rand_dist_y).to(self.device)
        rand_dist_z = np.random.rand(self.num_scenes_row, self.num_scenes_col) * 1.2 - 0.6
        change_height = 0
        rand_dist_z = rand_dist_z * change_height
        self.rand_dist_z = torch.from_numpy(rand_dist_z).to(self.device)

        for i in range(self.num_scenes_row):
            for j in range(self.num_scenes_col):
                mesh_vertices = mesh_vertices_list[self.scene_idx[i,j]]
                mesh_triangles = mesh_triangles_list[self.scene_idx[i,j]]
                mesh_vertices_offset = mesh_vertices.copy()
                mesh_vertices_offset = VaryPoint(mesh_vertices_offset, 'Z', obj_rotate[i,j]).astype(np.float32)
                mesh_vertices_offset[:, 0] += self.spacing *2 * i + rand_dist_x[i, j]
                mesh_vertices_offset[:, 1] += self.spacing *2 * j + rand_dist_y[i, j]
                mesh_vertices_offset[:, 2] += rand_dist_z[i, j]
                if i ==0 and j == 0:
                    self.mesh_vertices_offset = torch.from_numpy(mesh_vertices_offset).to(self.device)
                    self.mesh_triangles = torch.from_numpy(mesh_triangles.astype(np.int32)).to(self.device)
                    self.verts_obj_idx = torch.from_numpy(verts_obj_idx_list[self.scene_idx[i,j]]).to(self.device)
                    self.faces_obj_idx = torch.from_numpy(faces_obj_idx_list[self.scene_idx[i,j]]).to(self.device)
                curr_bps = bps_plans[self.scene_idx[i,j]]

                tm_params = gymapi.TriangleMeshParams()
                tm_params.nb_vertices = mesh_vertices_offset.shape[0]
                tm_params.nb_triangles = mesh_triangles.shape[0]
                self.gym.add_triangle_mesh(self.sim, mesh_vertices_offset.flatten(order='C'),
                                        mesh_triangles.flatten(order='C'),
                                        tm_params)
                curr_plan = self.plan_items[self.scene_idx[i,j]]
                curr_scene_children = []
                curr_env_idx = i * self.num_scenes_col + j
                for k in range(self.num_obj_per_scene):
                    curr_obj_id = curr_plan[k]['id']
                    curr_category = self.obj_info[curr_obj_id]['category']
                    self._every_env_obj_cls[curr_env_idx, k] = self.ObjectType[curr_category].value
                    self._every_env_obj_bps[curr_env_idx, k] = torch.from_numpy(curr_bps[k]).to(self.device)
                    curr_obj_children = []
                    if 'children' in self.obj_info[curr_obj_id]:
                        for child in self.obj_info[curr_obj_id]['children']:
                            if child:
                                curr_obj_children.append(child[1])
                    else:
                        curr_obj_children.append('')
                    curr_scene_children.append(curr_obj_children)
                    height = self.obj_info[curr_obj_id]['height']
                self._every_env_children.append(curr_scene_children)
                

    def _load_pcd(self,):
        pcds_id = dict()
        for obj_id in self.obj_info:
            curr_obj_info = self.obj_info[obj_id]
            raw_id = curr_obj_info['raw_id']
            pcd_path = f'scene_data/3dfront/obj_models/{raw_id}/point_cloud_seg.ply'
            if os.path.exists(pcd_path):
                pcd = o3d.io.read_point_cloud(pcd_path)
            
            # trans = curr_obj_info['transfer']
            theta = - curr_obj_info['rotate']
            scale = curr_obj_info['scale']

            R = np.zeros((3, 3)).astype(np.float32)
            R[0, 0] = np.cos(theta)
            R[0, 1] = -np.sin(theta)
            R[1, 0] = np.sin(theta)
            R[1, 1] = np.cos(theta)
            R[2, 2] = 1.
            pcd = np.asarray(pcd.points).astype(np.float32)
            pcd = pcd * np.array(scale).reshape(1, 3).astype(np.float32)
            # pcd = pcd @ R.T
            # pcd = pcd + np.array(trans).reshape(1, 3).astype(np.float32)
            label_file = f'scene_data/3dfront/obj_models/{raw_id}/seat_indices.txt'
            if os.path.exists(label_file):
                seat_indices = load_label(label_file)
                pcd_label = np.zeros(pcd.shape[0])
                pcd_label[seat_indices] = 1
            else:
                pcd_label = None

            pcds_id[obj_id] = dict(pcd=pcd, pcd_label=pcd_label)
        pcds_plan = []
        for plan_id, curr_plan in enumerate(self.plan_items):
            pcd_multi = []
            
            for j in range(len(curr_plan)):
                curr_obj = curr_plan[j]
                pid = curr_obj['id']
                pcd = copy.deepcopy(pcds_id[pid]['pcd'])
                transfer = curr_obj['transfer']

                min_x_pcd, min_y_pcd, min_z_pcd = pcd[:,0].min(), pcd[:,1].min(), pcd[:,2].min()
                pcd = pcd + np.array(transfer).reshape(1, 3).astype(np.float32)
                pcd_multi.append(pcd)
            pcd_obj = pcd_multi
            pcd_multi = np.concatenate(pcd_multi, axis=0)
            pcds_plan.append(dict(pcd=pcd_multi, pcd_obj=pcd_obj))
        return pcds_id, pcds_plan

    def _get_pcd_parts(self, pcd_dict, pcd_plans):
        self.height_map_plan = []
        self.height_map_bounding = []
        self.pcd_buffer_plan = torch.zeros([self.plan_number, self.num_obj_per_scene, self.num_max_children, self.num_pcd_buffer, 3]) # N, 4, 4, self.num_pcd_buffer, 3
        self.stand_point_plan = torch.zeros([self.plan_number, self.num_obj_per_scene, 4, 3])        
        from tqdm import tqdm
        for plan_id, curr_plan in tqdm(enumerate(self.plan_items)):
            pcd = pcd_plans[plan_id]['pcd']
            pcd_objs = pcd_plans[plan_id]['pcd_obj']
            curr_height_map_bounding = []
            for j in range(len(curr_plan)):
                curr_obj = curr_plan[j]
                obj_id = curr_obj['id']
                label = pcd_dict[obj_id]['pcd_label']
                pcd_obj = pcd[j*10000: (j+1)*10000]
                children = self.obj_info[obj_id].get('children', '')
                self.obj_info[obj_id]['children'] = children
                transfer = curr_plan[j]['transfer']
                children_pcd = self._get_obj_parts(label, pcd_obj)
                if len(children_pcd):
                    self.pcd_buffer_plan[plan_id, j, :len(children_pcd)] = children_pcd
                obj_scale = self.obj_info[obj_id]['scale']
                standpoint = self.obj_info[obj_id]['standpoint']
                standpoint = np.array(standpoint).reshape(4, 3)
                standpoint = standpoint* self.standpoint_scale
                standpoint[:, 2] = 0.86
                self.stand_point_plan[plan_id, j, :] = torch.tensor(standpoint, device=self.device).view(4, 3) + torch.tensor(transfer, device=self.device) 
                curr_obj_heigh_pcd = get_height_map(pcd_objs[j], HEIGHT_MAP_DIM=self.obj_htmp_scale)
                curr_height_map_bounding.append(torch.from_numpy(curr_obj_heigh_pcd).to(self.device))
            heigh_pcd = get_height_map(pcd, HEIGHT_MAP_DIM=self.height_map_dim)

            self.height_map_plan.append(torch.from_numpy(heigh_pcd).to(self.device))
            curr_height_map_bounding = torch.stack(curr_height_map_bounding, 0)
            self.height_map_bounding.append(curr_height_map_bounding)
        self.height_map_bounding = torch.stack(self.height_map_bounding, 0).to(self.device) # n, num_obj_per_scene, local_scale**2, 3
        self.height_map_plan = torch.stack(self.height_map_plan,0).to(self.device)
        # n, heightmap_dim ** 2, 3
        self.stand_point_plan =  self.stand_point_plan.to(self.device)
        self.pcd_buffer_plan = self.pcd_buffer_plan.to(self.device)

    def process_contact(self, env_ids, obj_type, env_obj_contact, caption):
        pass

    def _get_obj_parts(self, points_label, pcd):

        max_x, min_x, max_y, min_y = pcd[:, 0].max(), pcd[:, 0].min(), pcd[:, 1].max(), pcd[:, 1].min()   

        if points_label is not None:
            part_pcd = pcd[points_label==1]
        else:
            part_pcd = pcd
        max_x, min_x, max_y, min_y, max_z, min_z = part_pcd[:,0].max(), part_pcd[:,0].min(), part_pcd[:,1].max(), part_pcd[:,1].min(), part_pcd[:,2].max(), part_pcd[:,2].min()


        out_part_pcd = part_pcd[(part_pcd[:,0] <= max((max_x-0.1), (max_x-min_x)/5*4+min_x)) & (part_pcd[:,0] >= min(min_x+0.1, (max_x-min_x)/5*1+min_x)) &
                            (part_pcd[:,1] <= max((max_y-0.1), (max_y-min_y)/5*4+min_y)) & (part_pcd[:,1] >= min(min_y+0.1, (max_y-min_y)/5*1+min_y))] # filter edge
        out_part_pcd = torch.from_numpy(out_part_pcd).to(self.device)
        idx = farthest_point_sample(out_part_pcd[None], self.num_pcd_buffer)
        out_part_pcd = out_part_pcd[idx]

        return out_part_pcd
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        self.env_scene_idx_row = (self.env_array % num_per_row % self.num_scenes_row).long()
        self.env_scene_idx_col = (self.env_array // num_per_row % self.num_scenes_col).long()
        self.scene_for_env = self.scene_idx[self.env_scene_idx_row, self.env_scene_idx_col]

        self.x_offset = (self.env_array % num_per_row % self.num_scenes_row) * spacing * 2 - (self.env_array % num_per_row) * spacing * 2
        self.y_offset = (self.env_array // num_per_row % self.num_scenes_col) * spacing * 2- (self.env_array // num_per_row) * spacing * 2

        self.obj_rotate_matrix = self.obj_rotate_matrix.permute(2,3,0,1).float()
        self.scene_stand_point = torch.einsum("nmoe,neg->nmog", self.stand_point_plan[self.scene_for_env], self.obj_rotate_matrix[self.env_scene_idx_row, self.env_scene_idx_col])
        self.obj_rotate_angle = self.obj_rotate_angle[self.env_scene_idx_row, self.env_scene_idx_col]
        self.scene_stand_point[..., 0] += self.x_offset[:,None,None] + self.rand_dist_x[self.env_scene_idx_row, self.env_scene_idx_col][:,None,None]
        self.scene_stand_point[..., 1] += self.y_offset[:,None,None] + self.rand_dist_y[self.env_scene_idx_row, self.env_scene_idx_col][:,None,None]     
        self.scene_stand_point[..., 2] = 0.86

        self._every_env_obj_bps = torch.einsum("nmoe,neg->nmog", self._every_env_obj_bps, self.obj_rotate_matrix[self.env_scene_idx_row, self.env_scene_idx_col])
        self._every_env_obj_bps[..., 0] += self.x_offset[:,None,None] + self.rand_dist_x[self.env_scene_idx_row, self.env_scene_idx_col][:,None,None]
        self._every_env_obj_bps[..., 1] += self.y_offset[:,None,None] + self.rand_dist_y[self.env_scene_idx_row, self.env_scene_idx_col][:,None,None]   

        self.envs_heightmap = self.height_map_plan[self.scene_for_env].float()
        self.envs_heightmap[..., 0] += self.rand_dist_x[self.env_scene_idx_row, self.env_scene_idx_col][..., None]
        self.envs_heightmap[..., 1] += self.rand_dist_y[self.env_scene_idx_row, self.env_scene_idx_col][..., None]
        self.envs_heightmap[..., 2] += self.rand_dist_z[self.env_scene_idx_row, self.env_scene_idx_col][..., None]
        self.envs_heightmap = torch.einsum("nae,neg->nag", self.envs_heightmap, self.obj_rotate_matrix[self.env_scene_idx_row, self.env_scene_idx_col])

        self.envs_heightmap_obj = self.height_map_bounding[self.scene_for_env].float()
        self.envs_heightmap_obj[..., 0] += self.rand_dist_x[self.env_scene_idx_row, self.env_scene_idx_col][..., None, None]
        self.envs_heightmap_obj[..., 1] += self.rand_dist_y[self.env_scene_idx_row, self.env_scene_idx_col][..., None, None]
        self.envs_heightmap_obj[..., 2] += self.rand_dist_z[self.env_scene_idx_row, self.env_scene_idx_col][..., None, None]
        self.envs_heightmap_obj = torch.einsum("nmae,neg->nmag", self.envs_heightmap_obj, self.obj_rotate_matrix[self.env_scene_idx_row, self.env_scene_idx_col])

        self._every_env_target_obj_idx = torch.randint(0, self.num_obj_per_scene, (self.num_envs,), device=self.device, dtype=torch.long)
        # pcd_buffer_plan: n_plan, 4, 4, self.num_pcd_buffer, 3
        # envs_children_pcd_buffer: n_env, 4, 4, self.num_pcd_buffer, 3
        self.envs_children_pcd_buffer = self.pcd_buffer_plan[self.scene_for_env]
        self.envs_children_pcd_buffer = torch.einsum("nmcoe,neg->nmcog", self.envs_children_pcd_buffer, self.obj_rotate_matrix[self.env_scene_idx_row, self.env_scene_idx_col])
        self.envs_children_pcd_buffer[..., 0] += self.x_offset[:, None, None, None] + self.rand_dist_x[self.env_scene_idx_row, self.env_scene_idx_col][..., None, None, None]
        self.envs_children_pcd_buffer[..., 1] += self.y_offset[:, None, None, None] + self.rand_dist_y[self.env_scene_idx_row, self.env_scene_idx_col][..., None, None, None]
        self.envs_children_pcd_buffer[..., 2] += self.rand_dist_z[self.env_scene_idx_row, self.env_scene_idx_col][..., None, None, None]
        self.envs_obj_pcd_buffer = torch.zeros([num_envs, self.contact_joint_num, self.num_pcd_buffer, 3], device=self.device)
        # update every time updating target
        # visualize height map using markers
        if (not self.headless):
            self._height_map_marker_handles = [[] for _ in range(num_envs)]
            self._load_marker_asset()
        
        super()._create_envs(num_envs, spacing, num_per_row)
        return
    
    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if (self._enable_conditional_disc):
            obs_size += self._num_motion_condition_obs
        return obs_size


    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        return


    def _load_marker_asset(self):
        asset_root = "sims/data/assets/mjcf/"
        asset_file = "location_marker.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        if (not self.headless):
            self._build_marker(env_id, env_ptr)
        return
    
    def _build_marker(self, env_id, env_ptr):
        ##### marker to indicate target position
        col_group = self.num_envs + 1
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()

        ##### marker to indicate square height map
        for i in range(self.local_scale):
            for j in range(self.local_scale):

                if j == (self.local_scale - 1): # to indicate the rotation of the height map sensor
                    color = gymapi.Vec3(0.0, 0.0, 1.0) # blue
                else:
                    color = gymapi.Vec3(0.0, 1.0, 0.0) # green

                marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", self.num_envs + 10, 0, 0)
                self.gym.set_actor_scale(env_ptr, marker_handle, 0.2)
                self.gym.set_rigid_body_color(env_ptr, marker_handle, 0,
                                            gymapi.MESH_VISUAL,
                                            color)
                self._height_map_marker_handles[env_id].append(marker_handle)

        # obj markers
        # for i in range(self.obj_htmp_scale):
        #     for j in range(self.obj_htmp_scale):

        #         color = gymapi.Vec3(1.0, 1.0, 1.0) # white

        #         marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", self.num_envs + 10, 0, 0)
        #         self.gym.set_actor_scale(env_ptr, marker_handle, 0.2)
        #         self.gym.set_rigid_body_color(env_ptr, marker_handle, 0,
        #                                     gymapi.MESH_VISUAL,
        #                                     color)
        #         self._height_map_marker_handles[env_id].append(marker_handle)
        return
    
    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs

        # height map markers
        self._height_map_marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1:, :] # actors: humanoid, height map markers
        if self.action_type == 'hoi':
            self._height_map_marker_states = self._height_map_marker_states[:, :-1, :]
        self._height_map_marker_pos = self._height_map_marker_states[..., :3]

        self._height_map_marker_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(self._height_map_marker_handles, dtype=torch.int32, device=self.device)
        self._height_map_marker_actor_ids = self._height_map_marker_actor_ids.flatten()

        return
    
    # def _build_marker_state_tensors(self):
    #     num_actors = self._root_states.shape[0] // self.num_envs
    #     self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., num_actors - 1, :]
    #     self._marker_pos = self._marker_states[..., :3]
        
    #     self._marker_actor_ids = self._humanoid_actor_ids + (num_actors - 1)

    #     return

    def compute_heightmap_marker(self):
        root_pos = self._humanoid_root_states[..., 0:3].clone()
        root_rot = self._humanoid_root_states[..., 3:7].clone()
        local_scale = self.local_scale
        # height_map_pcd = self.envs_heightmap.clone()
        height_map_pcd = self.envs_heightmap_obj.clone().view(-1, self.obj_htmp_scale*self.obj_htmp_scale*self.num_obj_per_scene, 3)

        mesh_pos = self.mesh_pos.clone()
        human_in_mesh = self.humanoid_in_mesh.clone()

        origin_root_pos = root_pos.clone()
        origin_root_pos[:, 0] = origin_root_pos[:, 0] - self.x_offset[:]
        origin_root_pos[:, 1] = origin_root_pos[:, 1] - self.y_offset[:]

        # transform sensors from local space to global space

        mesh_pos = mesh_pos[None] + (root_pos -human_in_mesh)[:, None] # transform height map sensor from local space to global space
        mesh_dist = (mesh_pos - root_pos[:, None]).reshape(-1, 3)

        heading_rot_for_height = root_rot[:, None].repeat(1, local_scale*local_scale, 1).reshape(-1, 4)
        heading_rot_for_height[:, :2] = 0
        heading_rot_norm = torch.sqrt(1 - heading_rot_for_height[:, 3]*heading_rot_for_height[:, 3])
        heading_rot_for_height[:, 2] = heading_rot_for_height[:, 2] * heading_rot_norm / torch.abs(heading_rot_for_height[:, 2])

        rotated_mesh_dist = quat_rotate(heading_rot_for_height, mesh_dist).reshape(-1, local_scale*local_scale, 3)
        rotated_mesh_pos = rotated_mesh_dist + root_pos[:, None]
        rotated_mesh_origin_pos = rotated_mesh_dist + origin_root_pos[:, None]

        # calculate height values
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
        markers_pos3d = rotated_mesh_origin_pos.clone()

        markers_pos3d[..., -1] = height_map # n, 81, 3
        return markers_pos3d
    
    def _update_marker(self):

        markers_pos3d = self.compute_heightmap_marker()
        
        # markers_pos3d = torch.cat([markers_pos3d, obj_markers_pos3d], dim=1)
        # refresh physics tensors   
        self._height_map_marker_pos[:] = markers_pos3d
        env_ids_int32 = self._height_map_marker_actor_ids
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return
    
    def _draw_task(self):
        obj_bps = self._every_env_obj_bps # num_envs, num_obj_per_scene, 8, 3
        verts_tar_box = torch.stack([
            obj_bps[..., 0, :], obj_bps[..., 1, :],
            obj_bps[..., 1, :], obj_bps[..., 2, :],
            obj_bps[..., 2, :], obj_bps[..., 3, :],
            obj_bps[..., 3, :], obj_bps[..., 0, :],

            obj_bps[..., 4, :], obj_bps[..., 5, :],
            obj_bps[..., 5, :], obj_bps[..., 6, :],
            obj_bps[..., 6, :], obj_bps[..., 7, :],
            obj_bps[..., 7, :], obj_bps[..., 4, :],

            obj_bps[..., 0, :], obj_bps[..., 5, :],
            obj_bps[..., 1, :], obj_bps[..., 6, :],
            obj_bps[..., 2, :], obj_bps[..., 7, :],
            obj_bps[..., 3, :], obj_bps[..., 4, :],
        ], dim=2).cpu() # (num_envs, num_obj_per_scene, 24, 3)

        cols = np.array([[.0, .0, .0]], dtype=np.float32)
        cols = cols.repeat(24, axis=0)
        # for i, env_ptr in enumerate(self.envs):
        #     for j in range(self.num_obj_per_scene):
        #         curr_verts = verts_tar_box[i, j]
        #         curr_verts = curr_verts.reshape([12, 6])
        #         self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return
    
    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def fetch_amp_obs_demo(self, num_samples):
        # random sample a skill
        sk_id = torch.multinomial(self._skill_disc_prob, num_samples=1, replacement=True)
        sk_name = self._skills[sk_id]
        curr_motion_lib = self._motion_lib_dict[sk_name]
        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
        motion_ids = curr_motion_lib.sample_motions(num_samples)
        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = curr_motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0, curr_motion_lib)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0, motion_lib):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps
        time_steps = time_steps.unsqueeze(0).unsqueeze(-1)

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              dof_pos, dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets)
        ## double check
        if self._enable_conditional_disc:
            motion_labels = motion_lib.get_motion_label(motion_ids)
            amp_obs_demo = torch.cat([amp_obs_demo, motion_labels], dim=-1)
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return

    def _sample_motion_label(self, curr_env_ids,):
        num_samples = len(curr_env_ids)
        given_ids = torch.ones((num_samples,), device=self.device).long() * self.num_resets
        target_obj_cls = self._every_env_obj_cls[torch.arange(self.num_envs), self._every_env_target_obj_idx][curr_env_ids]
        target_obj_cls = target_obj_cls.view(-1)
        obj_types = self.object_index[target_obj_cls.tolist()].tolist()
        assert len(obj_types) == num_samples
        if self._enable_contact and self._enable_task_update:
            skill_ids, motion_label, self_contact, object_contact, caption = self._motion_lib_dict.sample_motion_labels_obj_range(num_samples, obj_types=obj_types, given_ids=given_ids)
            self._every_env_skill_id[curr_env_ids] = skill_ids

            self.process_contact(env_ids=curr_env_ids, obj_type=obj_types,  env_obj_contact=object_contact, caption=caption)
        else:
            skill_ids, motion_label, _, _, caption = self._motion_lib_dict.sample_motion_labels_obj_range(num_samples, obj_types=obj_types, given_ids=given_ids)
            self._every_env_skill_id[curr_env_ids] = skill_ids
        if self.num_envs == 1 and len(caption) > 0:
            print(caption[0], self.num_resets)
        self.num_resets += 1
        self.captions = caption
        return motion_label
    
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/amp_humanoid_sword_shield.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 31 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)
        self._num_amp_obs_per_step_basic = self._num_amp_obs_per_step
        if self._enable_conditional_disc:
            self._num_amp_obs_per_step += self._num_motion_condition_obs
        
        return
    
    def _load_motion(self, motion_files):
        assert(self._dof_offsets[-1] == self.num_dof)

        self._motion_lib_dict = MotionLibDict(motion_files=motion_files,
                                              allowed_skills=self._skills,
                                              
                                        dof_body_ids=self._dof_body_ids,
                                        dof_offsets=self._dof_offsets,
                                        key_body_ids=self._key_body_ids.cpu().numpy(), 
                                        skill_weights=self._skill_weights,
                                        motion_encoder=self.motion_enc,
                                        condition_dim=self._num_motion_condition_obs,
                                        device=self.device,
                                        extract_from_text=self._extract_text_features,
                                        is_test=not self._is_train,)
        return

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = {}
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}
        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
            self._init_amp_obs(env_ids)
        return    
    
    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidSIMS.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidSIMS.StateInit.Start
              or self._state_init == HumanoidSIMS.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidSIMS.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return

    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        
        if self._enable_conditional_disc:
            if len(env_ids) > 0:
                self._every_env_motion_label[env_ids] = self._sample_motion_label(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._kinematic_humanoid_rigid_body_states[self._reset_default_env_ids] = self._initial_humanoid_rigid_body_states[self._reset_default_env_ids]

    def _reset_ref_state_init(self, env_ids):
        sk_ids = torch.multinomial(self._skill_init_prob, num_samples=env_ids.shape[0], replacement=True)

        for uid, sk_name in enumerate(self._skills):
            curr_motion_lib = self._motion_lib_dict[sk_name]
            curr_env_ids = env_ids[(sk_ids == uid).nonzero().squeeze(-1)] # be careful!!!

            if len(curr_env_ids) > 0:

                num_envs = curr_env_ids.shape[0]
                motion_ids = curr_motion_lib.sample_motions(num_envs)

                if (self._state_init == HumanoidSIMS.StateInit.Random
                    or self._state_init == HumanoidSIMS.StateInit.Hybrid):
                    motion_times = curr_motion_lib.sample_time(motion_ids) # avoid times with serious self-penetration
                elif (self._state_init == HumanoidSIMS.StateInit.Start):
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

        if self._enable_conditional_disc:
            if len(env_ids) > 0:
                self._every_env_motion_label[env_ids] = self._sample_motion_label(env_ids)
        return


    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)
        
        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        for i, sk_name in enumerate(self._skills):
            if self._reset_ref_env_ids.get(sk_name) is not None:
                if (len(self._reset_ref_env_ids[sk_name]) > 0):
                    self._init_amp_obs_ref(self._reset_ref_env_ids[sk_name], self._reset_ref_motion_ids[sk_name],
                                           self._reset_ref_motion_times[sk_name], sk_name)

        return
    
    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times, skill_name):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib_dict[skill_name].get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, 
                                              dof_pos, dof_vel, key_pos, 
                                              self._local_root_obs, self._root_height_obs, 
                                              self._dof_obs_size, self._dof_offsets)
        
        if self._enable_conditional_disc:
            motion_labels = self._every_env_motion_label[env_ids]
            motion_labels = torch.broadcast_to(motion_labels.unsqueeze(-2), [motion_labels.shape[0], self._num_amp_obs_steps - 1, motion_labels.shape[1]])

            amp_obs_demo = torch.cat([amp_obs_demo, motion_labels.reshape(-1, motion_labels.shape[-1])], dim=-1)

        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return
    
    def _compute_amp_observations(self, env_ids=None):

        # print("motion_label: ", self._every_env_motion_label[:4].nonzero()[:, 1])
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if (env_ids is None):
            
            amp_obs = build_amp_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._dof_offsets)
            if self._enable_conditional_disc:
                self._curr_amp_obs_buf[:] = torch.cat([amp_obs, self._every_env_motion_label], dim=-1)
            else:
                self._curr_amp_obs_buf[:] = amp_obs
        else:
            kinematic_rigid_body_pos = self._kinematic_humanoid_rigid_body_states[:, :, 0:3]
            key_body_pos = kinematic_rigid_body_pos[:, self._key_body_ids, :]
            amp_obs = build_amp_observations(self._kinematic_humanoid_rigid_body_states[env_ids, 0, 0:3],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 3:7],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 7:10],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 10:13],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._dof_offsets)
            if self._enable_conditional_disc:
                self._curr_amp_obs_buf[env_ids] = torch.cat([amp_obs, self._every_env_motion_label[env_ids]], dim=-1)
            else:
                self._curr_amp_obs_buf[env_ids] = amp_obs
        
        return
    
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
        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return
    
    def _fetch_humanoid_full_states(self, env_ids=None):
        if env_ids is None:
            env_ids = to_torch(np.arange(self.num_envs), dtype=torch.long, device=self.device)
        full_states = torch.cat((self._rigid_body_pos[env_ids], self._rigid_body_rot[env_ids]), dim=-1) # num_envs, 15, 7
        # For just reset envs, these two variables won't be updated, so read correct values from buffer

        full_states_buffer = self._kinematic_humanoid_rigid_body_states[..., 0:7]
        mask = (self.progress_buf == 0)
        full_states[mask] = full_states_buffer[mask]
        return full_states
    
    def _compute_reset(self):
        self.big_force = (self._contact_forces.abs()>10000).sum((-2,-1))>0
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights)
        return
#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
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
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

@torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs):
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
    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs

  
