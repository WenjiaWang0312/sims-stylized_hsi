
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
from sims.learning.sims_motion_enc_model import MotionCLIP
from isaacgym.torch_utils import *
from sims.utils.motion_lib_dict import MotionLibDictTest
from .humanoid_sims_task import HumanoidSIMSTask, compute_strike_observations, compute_loco_observations, compute_hoi_observations, compute_heightmap_obs
# from .humanoid_sims import HumanoidSIMS, farthest_point_sample, get_height_map, load_label, get_obb

from .humanoid_sims_3dfront import HumanoidSIMS, farthest_point_sample, get_height_map, load_label, get_obb

from sims.utils import torch_utils
from sims.utils.clip_utils import CLIPExtractor
from sims.plans import load_plan_file

    
class Humanoid3DFrontTest(HumanoidSIMSTask):
    skill_flags = {'loco': 1, 'sitdown': 2, 'liedown': 3, 'touch': 4, 'standup': 5, 'getup': 6, 'idle': 7, 'carry': 8}
    hoi_skills = ['carry']
    hsi_skills = ['sitdown', 'liedown', 'standup', 'getup', 'touch']
    loco_skills = ['loco', 'idle']
    action_type = 'hoi'

    class ObjectType(Enum):
        chair = 0
        armchair = 1
        bed = 2
        sofa = 3
        stool = 4
        table = 5
        desk = 6
        stand = 7
        cabinet = 8
        shelf = 9
        wardrobe = 10
        lamp = 11
        
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):  
        self._original_box_size = cfg["env"]["boxSize"]
       
        self._enable_task_obs = cfg["env"]['task']["enableTaskObs"]
        self._max_finish_length = cfg["env"]['task']["maxFinishLengthDict"]

        plan_file = cfg["env"]["task"]["planFile"]
        self.scene_data_root = cfg["env"]["obj"]["sceneDataRoot"]

        plan = load_plan_file(plan_file)
        high_level_order = plan["highLevelOrder"]
        self.init_location = plan.get("initLocation")
        self.box_target = plan.get("boxTarget", '')
        if "boxSize" in plan:
            self._original_box_size = plan["boxSize"]
        layout_json = plan["sceneJson"]
        self.trans_scale = cfg["env"]["obj"].get("transScale", 1.5)
        with open(cfg["env"]["obj"]["objFile"], 'r') as f:
            self.furniture_info = json.load(f)
        room_trans = plan.get("roomTrans", [[0, 0, 0]])
        if isinstance(layout_json, list):
            plan_info = []
            for json_path in layout_json:
                with open(f'{self.scene_data_root}/{json_path}', 'r') as f:
                    plan_info.append(json.load(f))
            plan_info = merge_layouts(plan_info, room_trans)
        else:
            with open(f'{self.scene_data_root}/{layout_json}', 'r') as f:
                plan_info = json.load(f)
        self.layout = self.load_plan(plan_info)

        self._skills = cfg["env"]["skill"]["skill"]
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self._finish_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._success_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.long)
        self._executionTime_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)
        self._precision_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)

        self.max_episode_length = cfg["env"]["episodeLength"]

        if plan.get("maxEpisodeLength", None) is not None:
            self.max_episode_length = plan["maxEpisodeLength"]
        # self._enable_IET = cfg["eval"]["enableIET"]
        self._state_init = HumanoidSIMSTask.StateInit[cfg["env"]["stateInit"]]
        self._success_threshold = cfg["env"]["successThreshold"]
        self.num_obj_per_scene = len(self.layout)
        self._enable_conditional_disc_dict = cfg["env"]['condition']["enableConditionalDisc"]

        self.motion_enc_dict = dict()
        # motion encoder
        for name in ['hsi', 'loco']:
            motion_enc = MotionCLIP()
            motion_enc.load_state_dict(torch.load(cfg['env']['condition'][f'motion_enc_ckpt_{name}']))
            motion_enc.to(self.device)
            motion_enc.eval()
            if self._enable_conditional_disc:
                assert motion_enc.latent_dim == self._num_motion_condition_obs

            for param in motion_enc.parameters():
                param.requires_grad = False
            self.motion_enc_dict[name] = motion_enc
        ##################################################################
        tar_objects = []
        tasks = []
        captions = []
        object_contacts = []
        motion_conditions = []
        stand_choices = []
        
        self.clip_model = CLIPExtractor(device=self.device, cond_mode='text', clip_version='ViT-B/32', latent_dim=512, clip_dim=512, max_text_len=20)
        self._get_task_name = dict(zip(self.skill_flags.values(), self.skill_flags.keys()))

        for task in high_level_order:
            skill = self.skill_flags[task['skill']]
            tasks.append(skill)
            task_name = self._get_task_name[skill]
            print(self.obj_names)
            obj_name = task['object'].split('-')[0]
            if len(task['object'].split('-')) ==2:
                stand_choice = task['object'].split('-')[1]
            else:
                stand_choice = -1
            stand_choice = int(stand_choice)
            tar_objects.append(self.obj_names.index(obj_name))
            stand_choices.append(stand_choice)
            captions.append(task['caption'])
            object_contacts.append(task['object_contact'])
            if task_name in self.loco_skills:
                motion_enc = self.motion_enc_dict['loco']
            elif task_name in self.hsi_skills:
                motion_enc = self.motion_enc_dict['hsi']
            elif task_name in self.hoi_skills:
                motion_enc = self.motion_enc_dict['hoi']
            clip_feature = self.clip_model.encode_text(task['caption'])
            motion_conditions.append(motion_enc.encode_text(clip_feature))
        self._motion_conditions = torch.cat(motion_conditions, 0).to(self.device)
        self._high_level_task_order = to_torch(tasks, dtype=torch.long, device=self.device)
        self._high_level_tar_object_order = to_torch(tar_objects, dtype=torch.long, device=self.device)
        self._high_level_stand_choice_order = to_torch(stand_choices, dtype=torch.long, device=self.device)
        self._pointer = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._task_threshold = cfg["env"]["task"]["taskThreshold"]

        self.captions = captions
        self.object_contact = object_contacts
    
        self._every_env_target_obj_idx = self._high_level_tar_object_order[self._pointer]
        root_pos = self._humanoid_root_states[:, :3]
        root_diff = root_pos - self.scene_stand_point[0, self._every_env_target_obj_idx]
        if self._high_level_stand_choice_order[self._pointer] <0:
            self.stand_point_choice = torch.argmin(torch.sum(root_diff * root_diff, dim=-1), dim=-1)
        else:
            self.stand_point_choice = self._high_level_stand_choice_order[self._pointer]
        self.stand_point_chosen = self.scene_stand_point[0, self._every_env_target_obj_idx, self.stand_point_choice]
        self.stand_point_chosen[:, 2] = 0.86
        self._prev_box_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._build_box_tensors()
        self.frame_idx = 0
        return

    def _load_motion(self, motion_files):
        assert(self._dof_offsets[-1] == self.num_dof)

        self._motion_lib_dict = MotionLibDictTest(skills=self._skills)
        return
    
    ################### build box #####################
    def _build_box_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._box_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., -1, :]
        
        self._box_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + num_actors - 1

        self._initial_box_states = self._box_states.clone()
        self._initial_box_states[:, 7:13] = 0

        self._build_box_bps()

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_box(env_id, env_ptr)

        return

    def _build_box(self, env_id, env_ptr):
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter() + 1
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = self._box_size[env_id, 0] / 2 + 0.4
        default_pose.p.y = 0
        default_pose.p.z = self._box_size[env_id, 2] / 2 # ensure no penetration between box and ground plane
    
        box_handle = self.gym.create_actor(env_ptr, self._box_assets[env_id], default_pose, "box", col_group, col_filter, segmentation_id)
        self._box_handles.append(box_handle)

        return

    def _build_box_bps(self):
        bps_0 = torch.vstack([     self._box_size[:, 0] / 2,      self._box_size[:, 1] / 2, -1 * self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_1 = torch.vstack([-1 * self._box_size[:, 0] / 2,      self._box_size[:, 1] / 2, -1 * self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_2 = torch.vstack([-1 * self._box_size[:, 0] / 2, -1 * self._box_size[:, 1] / 2, -1 * self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_3 = torch.vstack([     self._box_size[:, 0] / 2, -1 * self._box_size[:, 1] / 2, -1 * self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_4 = torch.vstack([     self._box_size[:, 0] / 2,      self._box_size[:, 1] / 2,      self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_5 = torch.vstack([-1 * self._box_size[:, 0] / 2,      self._box_size[:, 1] / 2,      self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_6 = torch.vstack([-1 * self._box_size[:, 0] / 2, -1 * self._box_size[:, 1] / 2,      self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        bps_7 = torch.vstack([     self._box_size[:, 0] / 2, -1 * self._box_size[:, 1] / 2,      self._box_size[:, 2] / 2]).t().unsqueeze(-2)
        self._box_bps = torch.cat([bps_0, bps_1, bps_2, bps_3, bps_4, bps_5, bps_6, bps_7], dim=1).to(self.device) # (num_envs, 8, 3)
        return

    def _load_box_asset(self):
        # rescale

        self._box_density = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)

        self._box_density[:] = 200.0

        self._box_size = torch.tensor(self._original_box_size, device=self.device).reshape(1, 3)  # (num_envs, 3)

        # create asset
        self._box_assets = []
        for i in range(self.num_envs):
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.density = self._box_density[i]
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            self._box_assets.append(self.gym.create_box(self.sim, self._box_size[i, 0], self._box_size[i, 1], self._box_size[i, 2], asset_options))
        
        return
    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_box_pos[:] = self._box_states[..., 0:3]
        return


    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        self._finish_length_buf[env_ids] = 0
        env_ids_int32 = self._box_actor_ids[env_ids].view(-1)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return
    
    def _update_marker(self):

        markers_pos3d = self.compute_heightmap_marker()
        # markers_pos3d = torch.cat([markers_pos3d, obj_markers_pos3d], dim=1)
        # refresh physics tensors   
        self._height_map_marker_pos[:] = markers_pos3d
        # env_ids_int32 = self._height_map_marker_actor_ids
        env_ids_int32 = torch.cat([self._height_map_marker_actor_ids, self._box_actor_ids], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return
    
    ################### build box #####################
    def load_plan(self, plan_info):

        obj_angles = plan_info['obj_angles']
        obj_trans = plan_info['obj_trans']
        obj_raw_ids = plan_info['obj_raw_ids']
        obj_labels = plan_info['obj_labels']
        obj_names = plan_info['obj_names']
        obj_scales = plan_info['obj_scales']
        self.obj_names = obj_names
        self.room_w = plan_info['room_width']
        self.room_l = plan_info['room_length']
        curr_plan = []
        for j in range(len(obj_angles)):
            super_category = self.furniture_info[obj_raw_ids[j]]['super_category']
            standpoint = self.furniture_info[obj_raw_ids[j]]['standpoint']
            scale = obj_scales[j]
            scale_x, scale_y, scale_z = scale
           
            if super_category == 'bed':
                scale = [scale_x, scale_y , scale_z * 0.8]
            # elif super_category == 'chair':
            #     scale = [1.2*scale_x, 1.2*scale_y , scale_z]
            else:
                scale = [scale_x, scale_y, scale_z]
            transfer = obj_trans[j]
            x, y, z = transfer
            transfer = [x * self.trans_scale, y * self.trans_scale, z]

            obj_info = {
                'raw_id': obj_raw_ids[j],
                'transfer': transfer,
                'rotate': obj_angles[j],
                'scale': scale,
                'label': obj_labels[j],
                'name': obj_names[j],
                'standpoint': standpoint,
                'category': self.furniture_info[obj_raw_ids[j]]['category'],
                'super_category': super_category
            }
            curr_plan.append(obj_info)
        return curr_plan

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
        self._load_mesh()
        self.pcd_info = self._load_pcd()
        self._get_pcd_parts()

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
        self._every_env_obj_cls = torch.zeros([self.num_envs, len(self.layout)], device=self.device, dtype=torch.long)
        self._every_env_children = []
        self._every_env_obj_bps = torch.zeros([self.num_envs, len(self.layout), 8, 3], device=self.device, dtype=torch.float)

        mesh_vertices = np.zeros([0, 3]).astype(np.float32)
        mesh_triangles = np.zeros([0, 3]).astype(np.uint32)
    
        l = 0
        pn = 0
        for obj_id, curr_obj_info in enumerate(self.layout):
            category = curr_obj_info['category']
            super_category = curr_obj_info['super_category']
            self._every_env_obj_cls[:, obj_id] = self.ObjectType[super_category].value
            raw_id = curr_obj_info['raw_id']
            path = f'{self.scene_data_root}/obj_models/{raw_id}/mesh.obj'
            if os.path.exists(path):
                mesh = o3d.io.read_triangle_mesh(path)
            else:
                raise Exception(f"Object {raw_id} not found")
            
            trans = curr_obj_info['transfer']
            theta = - curr_obj_info['rotate']
            scale = curr_obj_info['scale']

            R = np.zeros((3, 3)).astype(np.float32)
            R[0, 0] = np.cos(theta)
            R[0, 1] = -np.sin(theta)
            R[1, 0] = np.sin(theta)
            R[1, 1] = np.cos(theta)
            R[2, 2] = 1.
            
            mesh_vertices_single = np.asarray(mesh.vertices).astype(np.float32())
            mesh_vertices_single = mesh_vertices_single * np.array(scale).reshape(1, 3).astype(np.float32)
            mesh_vertices_single = mesh_vertices_single @ R.T
            mesh_vertices_single = mesh_vertices_single + np.array(trans).reshape(1, 3).astype(np.float32)
            mesh_triangles_single = np.asarray(mesh.triangles).astype(np.uint32)
            mesh_vertices = np.concatenate([mesh_vertices, mesh_vertices_single], axis=0)
            mesh_triangles = np.concatenate([mesh_triangles, mesh_triangles_single], axis=0)
            mesh_triangles[l:] += pn
            l = mesh_triangles.shape[0]
            pn = mesh_vertices.shape[0]
            bps = get_obb(mesh)
            self._every_env_obj_bps[:, obj_id] = torch.from_numpy(bps).to(self.device)

            # if 'coffee_table' in curr_obj_info['name']:
            #     from IPython import embed; embed()
                #coffee_table_0: -4.95, -3.39, 0.34+0.2


        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = mesh_vertices.shape[0]
        tm_params.nb_triangles = mesh_triangles.shape[0]
        self.gym.add_triangle_mesh(self.sim, mesh_vertices.flatten(order='C'),
                                mesh_triangles.flatten(order='C'),
                                tm_params)
    
        scene_idx = np.random.randint(0, self.plan_number, (self.num_scenes_row, self.num_scenes_col))
        self.scene_idx = torch.from_numpy(scene_idx).to(self.device)
        return


    def _load_pcd(self,):
        pcds = []
        for obj_id, curr_obj_info in enumerate(self.layout):
            raw_id = curr_obj_info['raw_id']
            pcd_path = f'{self.scene_data_root}/obj_models/{raw_id}/point_cloud_seg.ply'
            if os.path.exists(pcd_path):
                pcd = o3d.io.read_point_cloud(pcd_path)
            
            trans = curr_obj_info['transfer']
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
            pcd = pcd @ R.T
            pcd = pcd + np.array(trans).reshape(1, 3).astype(np.float32)
            label_file = f'{self.scene_data_root}/obj_models/{raw_id}/seat_indices.txt'
            if os.path.exists(label_file):
                seat_indices = load_label(label_file)
                pcd_label = np.zeros(pcd.shape[0])
                pcd_label[seat_indices] = 1
            else:
                pcd_label = None
            pcds.append(dict(pcd=pcd, pcd_label=pcd_label))
        return pcds


    def _get_pcd_parts(self):
        self.height_map_plan = []
        self.height_map_bounding = []
        self.pcd_buffer_plan = torch.zeros([1, len(self.layout), 1, 400, 3]) # N, 4, 1, 200, 3
        self.stand_point_plan = torch.zeros([1, len(self.layout), 4, 3])        

        curr_height_map_bounding = []
        pcd_scene = []
        for j, curr_obj_info in enumerate(self.layout):
            label = self.pcd_info[j]['pcd_label']
            category = curr_obj_info['super_category']
            
            pcd_obj = self.pcd_info[j]['pcd']
            if category == 'lamp':
                pcd_obj*= 0
            transfer = curr_obj_info['transfer']
            children_pcd = self._get_obj_parts(label, pcd_obj)
            self.pcd_buffer_plan[0, j, 0, :, :] = children_pcd
            theta = - curr_obj_info['rotate']

            R = np.zeros((3, 3))
            R[0, 0] = np.cos(theta)
            R[0, 1] = -np.sin(theta)
            R[1, 0] = np.sin(theta)
            R[1, 1] = np.cos(theta)
            R[2, 2] = 1.

            standpoint = curr_obj_info['standpoint']
            standpoint = np.array(standpoint).astype(np.float32).reshape(4, 3)
            standpoint = standpoint @ R.T
            standpoint = standpoint+ np.array(transfer).reshape(1, 3).astype(np.float32)
            standpoint = torch.from_numpy(standpoint).to(self.device)
            self.stand_point_plan[0, j, :] = standpoint
            curr_obj_heigh_pcd = get_height_map(pcd_obj, HEIGHT_MAP_DIM=self.obj_htmp_scale)
            curr_height_map_bounding.append(torch.from_numpy(curr_obj_heigh_pcd).to(self.device))

            pcd_scene.append(pcd_obj)
        pcd_scene = np.concatenate(pcd_scene, axis=0)
        heigh_pcd_scene = get_height_map(pcd_scene, HEIGHT_MAP_DIM=self.height_map_dim)
        self.height_map_plan.append(torch.from_numpy(heigh_pcd_scene).to(self.device))
        curr_height_map_bounding = torch.stack(curr_height_map_bounding, 0)
        self.height_map_bounding.append(curr_height_map_bounding)
        self.height_map_bounding = torch.stack(self.height_map_bounding, 0).to(self.device) # n, num_obj_per_scene, local_scale**2, 3
        self.height_map_plan = torch.stack(self.height_map_plan,0).to(self.device)
        # n, heightmap_dim ** 2, 3
        self.stand_point_plan =  self.stand_point_plan.to(self.device)
        self.pcd_buffer_plan = self.pcd_buffer_plan.to(self.device)
    
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
        idx = farthest_point_sample(out_part_pcd[None], 400)
        out_part_pcd = out_part_pcd[idx]

        return out_part_pcd

    def _create_envs(self, num_envs, spacing, num_per_row):

        self.x_offset = (self.env_array % num_per_row % self.num_scenes_row) * spacing * 2 - (self.env_array % num_per_row) * spacing * 2
        self.y_offset = (self.env_array // num_per_row % self.num_scenes_col) * spacing * 2- (self.env_array // num_per_row) * spacing * 2

        self.scene_stand_point = self.stand_point_plan
        self.scene_stand_point[..., 2] = 0.86

        self.envs_heightmap = self.height_map_plan.float()

        self.envs_heightmap_obj = self.height_map_bounding.float()

        self._every_env_target_obj_idx = torch.zeros(1, device=self.device, dtype=torch.long)
        self.envs_children_pcd_buffer = self.pcd_buffer_plan
        self.envs_obj_pcd_buffer = torch.zeros([num_envs, self.contact_joint_num, 400, 3], device=self.device)

        if (not self.headless):
            self._height_map_marker_handles = [[] for _ in range(num_envs)]
            self._load_marker_asset()

        self._box_handles = []
        self._load_box_asset()

        super(HumanoidSIMS, self)._create_envs(num_envs, spacing, num_per_row)
        return
    

    def _reset_actors(self, env_ids): 
        if self._enable_task_update:
            self._pointer[env_ids] = 0

            self._every_env_target_obj_idx = self._high_level_tar_object_order[self._pointer]
            root_pos = self._humanoid_root_states[:, :3]
            root_diff = root_pos - self.scene_stand_point[0, self._every_env_target_obj_idx]
            if self._high_level_stand_choice_order[self._pointer] <0:
                self.stand_point_choice[env_ids] = torch.argmin(torch.sum(root_diff * root_diff, dim=-1), dim=-1)
            else:
                self.stand_point_choice[env_ids] = self._high_level_stand_choice_order[self._pointer]
            self.stand_point_chosen[env_ids] = self.scene_stand_point[env_ids, self._every_env_target_obj_idx[env_ids], self.stand_point_choice[env_ids]]
            self.stand_point_chosen[:, 2] = 0.86
            
            super(HumanoidSIMSTask, self)._reset_actors(env_ids)


    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        if isinstance(self.init_location, str):
            obj_name, standpoint_choice = self.init_location.split('-')
            standpoint_choice = int(standpoint_choice)
            obj_id = self.obj_names.index(obj_name)
            self._humanoid_root_states[env_ids, 0] = self.stand_point_plan[0, obj_id, standpoint_choice, 0]
            self._humanoid_root_states[env_ids, 1] = self.stand_point_plan[0, obj_id, standpoint_choice, 1]

            obj_center = self._every_env_obj_bps[env_ids, self._every_env_target_obj_idx[env_ids], :4].mean(1)
            deltax = self.stand_point_chosen[env_ids, 0] - obj_center[:, 0]
            deltay = self.stand_point_chosen[env_ids, 1] - obj_center[:, 1]
            rot_theta = torch.arctan2(deltay, deltax)
            axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            root_rot = quat_from_angle_axis(rot_theta, axis)
            self._humanoid_root_states[env_ids, 3:7] = root_rot
        elif isinstance(self.init_location, list):
            self._humanoid_root_states[env_ids, 0] = self.init_location[0]
            self._humanoid_root_states[env_ids, 1] = self.init_location[1]
        else:
            self._humanoid_root_states[env_ids, 0] = - self.room_w * self.trans_scale / 2 - 0.5
            self._humanoid_root_states[env_ids, 1] = - self.room_l * self.trans_scale/ 2 - 0.5
        self._box_states[env_ids, 0] = self._humanoid_root_states[env_ids, 0]
        self._box_states[env_ids, 1] = self._humanoid_root_states[env_ids, 1] - 1
        if (len(self._reset_default_env_ids) > 0):
            self._kinematic_humanoid_rigid_body_states[self._reset_default_env_ids] = self._initial_humanoid_rigid_body_states[self._reset_default_env_ids]

    def process_contact(self, object_contact):
        self.env_contact_valid[:] = 0
        self.envs_obj_pcd_buffer[:] = 0
        curr_obj_type = self.ObjectType(int(self._every_env_obj_cls[:, self._every_env_target_obj_idx[:]])).name

        curr_children = ['seat']

        curr_task = self._high_level_task_order[self._pointer]
        curr_task_name = self._get_task_name[int(curr_task)]

        if object_contact != 'none':
            contact_items = [object_contact] if isinstance(object_contact, str) else object_contact
            for obj_contact_item in contact_items:
                if curr_task_name == 'touch':
                    j1name, partname, contact_type = obj_contact_item, 'seat', 'up'
                else:
                    j1name, partname, contact_type = obj_contact_item.split('-')
                    if partname not in curr_children:
                        if partname == 'pillow':
                            continue
                        else:
                            raise ValueError(f'{partname} not in current object')
                if contact_type in ['up', 'none']:
                    contact = 1
                elif contact_type == 'nocontact':
                    contact = 0
                j1_idx = self.contact_joint_mapping[self.joint_abbr[j1name]]
                child_idx = curr_children.index(partname)
                curr_part_pcd = self.envs_children_pcd_buffer[:, self._every_env_target_obj_idx[:], child_idx]
                self.envs_obj_pcd_buffer[:, j1_idx] = curr_part_pcd
                self.env_contact_valid[:, j1_idx] = contact

        if curr_task_name in ['sitdown', 'liedown']:
            task_contact = self.task_contact[curr_task_name][curr_obj_type]
            for task_contact_item in task_contact:
                if not task_contact_item.replace('-up', '-nocontact') in object_contact:
                    j1name, partname, contact_type = task_contact_item.split('-')
                    # assert j1name == 'pelvis'
                    if partname not in curr_children:
                        if partname == 'pillow':
                            continue
                        else:
                            raise ValueError(f'{partname} not in current {curr_obj_type}')
                    j1_idx = self.contact_joint_mapping[self.joint_abbr[j1name]]
                    child_idx = curr_children.index(partname)
                    curr_part_pcd = self.envs_children_pcd_buffer[:, self._every_env_target_obj_idx[:], child_idx]
                    self.envs_obj_pcd_buffer[:, j1_idx] = curr_part_pcd
                    self.env_contact_valid[:, j1_idx] = 1


    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)

        every_env_task = self._high_level_task_order[self._pointer]
        task_name = self._get_task_name[int(every_env_task)]
        if self._enable_conditional_disc_dict[task_name]:
            motion_condition_obs = self._motion_conditions[self._pointer]
            obs = torch.cat([humanoid_obs, motion_condition_obs], dim=-1)
        else:
            obs = humanoid_obs
        if self._enable_task_update:    
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([obs, task_obs], dim=-1)
            
        if (env_ids is None):
            self.obs_buf = obs
        else:
            self.obs_buf = obs
        self.frame_idx += 1
        if self.frame_idx % 100 == 0:
            print(f'frame_idx: {self.frame_idx}')

        return
    
    def _compute_task_obs(self, env_ids=None):
        pos_diff = self.stand_point_chosen - self._humanoid_root_states[:, :3]
        self.tar_dir[:, :2] = torch.nn.functional.normalize(pos_diff[..., 0:2], dim=-1)
        curr_task = self._high_level_task_order[self._pointer]
        curr_task_name = self._get_task_name[int(curr_task)]
        if curr_task_name in self.hoi_skills:
            task_obs = self._compute_hoi_obs(env_ids)
        elif curr_task_name in self.hsi_skills:
            task_obs = self._compute_hsi_obs(env_ids)
        elif curr_task_name in self.loco_skills:
            task_obs = self._compute_loco_obs(env_ids)
        return task_obs

    def _compute_loco_obs(self, env_ids=None):
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
        curr_task = self._high_level_task_order[self._pointer]
        curr_task_name = self._get_task_name[int(curr_task)]
        if curr_task_name == 'idle':
            tar_pos = root_states[:, :3]

        obs = compute_loco_observations(root_states, tar_pos, self.humanoid_in_mesh, origin_root_pos, self.local_scale, height_map, self.mesh_pos, tar_dir, self.with_heightmap)

        root_pos = root_states[:, :3]
        pos_diff = tar_pos - root_pos
        pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
        pos_dist = torch.sqrt(pos_err)

        mask = pos_dist < self._task_threshold[curr_task_name]
        self._finish_length_buf[mask] += 1
        return obs

    def _compute_hoi_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            box_states = self._box_states
            box_bps = self._box_bps
            tar_pos = self.stand_point_chosen.clone()

            height_map = self.envs_heightmap_obj.clone().view(-1, self.obj_htmp_scale*self.obj_htmp_scale*self.num_obj_per_scene, 3)

            origin_root_pos = root_states[:, :3].clone()
            origin_root_pos[:, 0] = origin_root_pos[:, 0] - self.x_offset
            origin_root_pos[:, 1] = origin_root_pos[:, 1] - self.y_offset
        else:
            root_states = self._humanoid_root_states[env_ids]
            box_states = self._box_states[env_ids]
            box_bps = self._box_bps[env_ids]
            tar_pos = self.stand_point_chosen[env_ids].clone()

            height_map = self.envs_heightmap_obj.clone().view(-1, self.obj_htmp_scale*self.obj_htmp_scale*self.num_obj_per_scene, 3)[env_ids]

            origin_root_pos = root_states[:, :3].clone()
            origin_root_pos[:, 0] = origin_root_pos[:, 0] - self.x_offset[env_ids]
            origin_root_pos[:, 1] = origin_root_pos[:, 1] - self.y_offset[env_ids]
        tar_pos[:, 0] = self.box_target[0]
        tar_pos[:, 1] = self.box_target[1]
        if len(self.box_target) == 3:
            tar_pos[:, 2] = self.box_target[2]
        else:
            tar_pos[:, 2] = self._box_size[:, 2] / 2
        obs = compute_hoi_observations(root_states, box_states, box_bps, tar_pos)
        heightmap_obs = compute_heightmap_obs(root_states, self.humanoid_in_mesh, origin_root_pos, self.local_scale, height_map, self.mesh_pos)
        heightmap_obs[:] = 0
        obs = torch.cat([obs, heightmap_obs], dim=-1)

        curr_task = self._high_level_task_order[self._pointer]
        curr_task_name = self._get_task_name[int(curr_task)]

        box_pos = self._box_states[:, :3]
        pos_diff = tar_pos - box_pos
        pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
        pos_dist = torch.sqrt(pos_err)
        mask = pos_dist < self._task_threshold[curr_task_name]
        self._finish_length_buf[mask] += 1
        return obs
    
    def _compute_hsi_obs(self, env_ids=None):
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

        obs = compute_strike_observations(root_states, tar_pos, joint_pos_buffer, pcd_buffer, contact_valid, self.humanoid_in_mesh, origin_root_pos, self.local_scale, height_map, self.mesh_pos, tar_dir)

        curr_task = self._high_level_task_order[self._pointer]
        curr_task_name = self._get_task_name[int(curr_task)]

        if curr_task_name in ['standup', 'getup']:
            root_pos = root_states[:, :3]
            pos_diff = self.stand_point_chosen - root_pos
            pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
            pos_dist = torch.sqrt(pos_err)
            
        else:
            if curr_task_name in ['sitdown', 'liedown']:
                i = [0]
            elif curr_task_name in ['touch']:
                i = self.env_contact_valid.nonzero()[:, 1]

            pcd_buffer = self.envs_obj_pcd_buffer
            near_diff = pcd_buffer[0, i] - joint_pos_buffer[0, i][:, None]
            near_pos_err = torch.sum(near_diff * near_diff, dim=-1)
            
            near_pos_err_min, near_pos_idx = near_pos_err.min(-1)
            pos_dist = torch.sqrt(near_pos_err_min)
            self.joint_idx_buff[:, i] = near_pos_idx

        mask = pos_dist < self._task_threshold[curr_task_name]
        self._finish_length_buf[mask] += 1

        return obs
    
    def get_obs_size(self):
        obs_size = self._num_obs  
        return obs_size

    def get_task_obs_size(self, task): # 433
        obs_size = 0 
        if (self._enable_task_obs):
            if task in self.hsi_skills:
                obs_size = 6 + self.contact_joint_num + self.contact_joint_num * 3 # tar_pos, tar_dir, contact_valid, local_target_pos
            elif task in self.loco_skills:
                obs_size = 6 # tar_pos, tar_dir
            elif task in self.hoi_skills:
                obs_size = 3 + 3 + 6 + 8 * 3 # tar_pos, pos 3 + rot 6 + bps 24
            if self.with_heightmap:
                obs_size += self.local_scale*self.local_scale
        if self._enable_conditional_disc_dict[task]:
            obs_size += self._num_motion_condition_obs
        return obs_size

    # def get_task_obs_size(self, task):
    #     obs_size = 6 + self.joint_num * 3 + self.joint_num + self.local_scale * self.local_scale
    #     if self._enable_conditional_disc_dict[task]:
    #         obs_size += self._num_motion_condition_obs
    #     return obs_size

    def _fetch_humanoid_full_states(self, env_ids=None):
        if env_ids is None:
            env_ids = to_torch(np.arange(self.num_envs), dtype=torch.long, device=self.device)
        full_states = torch.cat((self._rigid_body_pos[env_ids], self._rigid_body_rot[env_ids]), dim=-1) # num_envs, 15, 7
        # For just reset envs, these two variables won't be updated, so read correct values from buffer

        full_states_buffer = self._kinematic_humanoid_rigid_body_states[..., 0:7]
        mask = (self.progress_buf == 0)
        full_states[mask] = full_states_buffer[mask]
        return full_states

    def _fetch_every_env_task(self):
        return self._high_level_task_order[self._pointer]

    def _update_task(self):
        env_ids = to_torch(np.arange(self.num_envs), dtype=torch.long, device=self.device)

        every_env_task = self._high_level_task_order[self._pointer]
        
        every_env_max_finish_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        for skill in self._skills:
            curr_task_mask = (every_env_task == self.skill_flags[skill])
            every_env_max_finish_lengths[curr_task_mask] = self._max_finish_length[skill]
        # print(self._finish_length_buf, every_env_max_finish_lengths)
        finished_task_mask = (self._finish_length_buf == every_env_max_finish_lengths - 1) # Update only once, otherwise Getup's tarpos keeps changing
        num_total_tasks = len(self._high_level_task_order)
        ongoing_mask = self._pointer < num_total_tasks - 1

        update_mask = torch.logical_and(finished_task_mask, ongoing_mask)
        self._finish_length_buf[update_mask] = 0
        self._pointer[update_mask] += 1
        self._every_env_target_obj_idx = self._high_level_tar_object_order[self._pointer]
        root_pos = self._humanoid_root_states[:, :3]
        root_diff = root_pos - self.scene_stand_point[0, self._every_env_target_obj_idx]
        if self._high_level_stand_choice_order[self._pointer] <0:
            self.stand_point_choice[env_ids] = torch.argmin(torch.sum(root_diff * root_diff, dim=-1), dim=-1)
        else:
            self.stand_point_choice[env_ids] = self._high_level_stand_choice_order[self._pointer]
        self.stand_point_chosen[env_ids] = self.scene_stand_point[env_ids, self._every_env_target_obj_idx[env_ids], self.stand_point_choice[env_ids]]
        self.stand_point_chosen[:, 2] = 0.86

        object_contact = self.object_contact[self._pointer]
        self.process_contact(object_contact=object_contact)
        return

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_task()
        self._prev_box_pos[:] = self._box_states[..., 0:3]
        return
    
    def get_amp_obs_size(self, task):
        key_bodies = self.cfg["env"]["keyBodies"]
        num_key_bodies = len(key_bodies)

        _num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]

        if self._enable_conditional_disc_dict[task]:
            _num_amp_obs_per_step += self._num_motion_condition_obs
        return _num_amp_obs_per_step
    

def merge_layouts(rooms, trans):
    merged_layout = {
        'obj_labels': [],
        'obj_names': [],
        'obj_bboxes': [],
        'obj_angles': [],
        'obj_trans': [],
        'obj_scales': [],
        'obj_raw_ids': [],
        'room_length': 0,
        'room_width': 0
    }

    label_counters = {}
    num_rooms = len(rooms)
    if len(trans) != num_rooms - 1:
        raise ValueError("The length of trans should be (number of rooms - 1)")

    for idx, room in enumerate(rooms):
        if idx == 0:
            translation = [0, 0, 0]
        else:
            translation = trans[idx - 1]

        room_length_with_offset = room.get('room_length', 0) + abs(translation[0])
        room_width_with_offset = room.get('room_width', 0) + abs(translation[1])
        merged_layout['room_length'] = max(merged_layout['room_length'], room_length_with_offset)
        merged_layout['room_width'] = max(merged_layout['room_width'], room_width_with_offset)

        for i in range(len(room['obj_labels'])):
            label = room['obj_labels'][i]
            name = room['obj_names'][i]
            bbox = room['obj_bboxes'][i]
            angle = room['obj_angles'][i]
            obj_trans = room['obj_trans'][i]
            scale = room['obj_scales'][i]
            raw_id = room['obj_raw_ids'][i]

            adjusted_trans = [
                obj_trans[0] + translation[0],
                obj_trans[1] + translation[1],
                obj_trans[2] + translation[2]
            ]
            adjusted_bbox = [
                [
                    bbox[0][0] + translation[0],
                    bbox[0][1] + translation[1],
                    bbox[0][2] + translation[2]
                ],
                [
                    bbox[1][0] + translation[0],
                    bbox[1][1] + translation[1],
                    bbox[1][2] + translation[2]
                ]
            ]
            if label not in label_counters:
                existing_ids = [int(n.split('_')[-1]) for n, l in zip(merged_layout['obj_names'], merged_layout['obj_labels']) if l == label and '_' in n and n.split('_')[-1].isdigit()]
                label_counters[label] = max(existing_ids, default=-1) + 1
            else:
                label_counters[label] += 1

            new_name = f"{label}_{label_counters[label]}"
            merged_layout['obj_labels'].append(label)
            merged_layout['obj_names'].append(new_name)
            merged_layout['obj_bboxes'].append(adjusted_bbox)
            merged_layout['obj_angles'].append(angle)
            merged_layout['obj_trans'].append(adjusted_trans)
            merged_layout['obj_scales'].append(scale)
            merged_layout['obj_raw_ids'].append(raw_id)
    return merged_layout
