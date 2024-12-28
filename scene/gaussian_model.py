#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
# from lietorch import SO3, SE3, Sim3, LieGroupParameter
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scipy.spatial.transform import Rotation as R
from utils.pose_utils import rotation2quad, get_tensor_from_camera
from utils.graphics_utils import getWorld2View2
from icecream import ic
import open3d as o3d

def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion to a rotation matrix.
    
    Parameters:
    - quaternion: A tensor of shape (..., 4) representing quaternions.
    
    Returns:
    - A tensor of shape (..., 3, 3) representing rotation matrices.
    """
    # Ensure quaternion is of float type for computation
    quaternion = quaternion.float()
    
    # Normalize the quaternion to unit length
    quaternion = quaternion / quaternion.norm(p=2, dim=-1, keepdim=True)
    
    # Extract components
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    
    # Compute rotation matrix components
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    
    # Assemble the rotation matrix
    R = torch.stack([
        torch.stack([1 - 2 * (yy + zz),     2 * (xy - zw),     2 * (xz + yw)], dim=-1),
        torch.stack([    2 * (xy + zw), 1 - 2 * (xx + zz),     2 * (yz - xw)], dim=-1),
        torch.stack([    2 * (xz - yw),     2 * (yz + xw), 1 - 2 * (xx + yy)], dim=-1)
    ], dim=-2)
    
    return R


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.confidence = torch.empty(0)
        self.first_major= True

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.P,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.P) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    def compute_relative_world_to_camera(self, R1, t1, R2, t2):
        # Create a row of zeros with a one at the end, for homogeneous coordinates
        zero_row = np.array([[0, 0, 0, 1]], dtype=np.float32)

        # Compute the inverse of the first extrinsic matrix
        E1_inv = np.hstack([R1.T, -R1.T @ t1.reshape(-1, 1)])  # Transpose and reshape for correct dimensions
        E1_inv = np.vstack([E1_inv, zero_row])  # Append the zero_row to make it a 4x4 matrix

        # Compute the second extrinsic matrix
        E2 = np.hstack([R2, -R2 @ t2.reshape(-1, 1)])  # No need to transpose R2
        E2 = np.vstack([E2, zero_row])  # Append the zero_row to make it a 4x4 matrix

        # Compute the relative transformation
        E_rel = E2 @ E1_inv

        return E_rel

    def init_RT_seq(self, cam_list, trainable_index):        
        poses =[]
        for cam in cam_list[1.0]:
            p = get_tensor_from_camera(cam.world_view_transform.transpose(0, 1)) # R T -> quat t
            poses.append(p)
        poses = torch.stack(poses)
        self.P = poses[:trainable_index].cuda().requires_grad_(True)
        self.non_trainable_P = poses.cuda()
        # self.P = poses.cuda().requires_grad_(True)

    def init_RT_seq_pose(self, extrinsics, trainable_index, num_poses):        
        poses =[]
        for extrin in extrinsics:
            p = get_tensor_from_camera(extrin) # R T -> quat t
            poses.append(p)
        if extrinsics.shape[0]<num_poses:
            iden_pose = torch.tensor([1,0,0,0,0,0,0]).cuda()
            for ii in range(num_poses-extrinsics.shape[0]):
                poses.append(iden_pose)
        poses = torch.stack(poses)
        self.P = poses[:trainable_index].cuda().requires_grad_(True)
        self.non_trainable_P = poses.cuda()
    
    def update_RT_seq(self, cam, index):
        self.non_trainable_P[index] = get_tensor_from_camera(cam.world_view_transform.transpose(0, 1))
        d = {"pose": self.non_trainable_P[index:index+1],}
        optimizable_tensors = self.cat_pose_tensors_to_optimizer(d)
        self.P = optimizable_tensors["pose"]

    def update_RT_seq_by_pose(self, pose, index):
        self.non_trainable_P[index] = pose
        d = {"pose": self.non_trainable_P[index:index+1],}
        optimizable_tensors = self.cat_pose_tensors_to_optimizer(d)
        self.P = optimizable_tensors["pose"]
        print(f"update pose {index}", self.P.requires_grad)

        # new_trainable = self.non_trainable_P[index:index+1].clone().detach().requires_grad_(True)
        # self.P = torch.cat([self.P, new_trainable], dim=0)
        # self.P = self.P.detach().requires_grad_(True)
        # self.l_cam = [{'params': [self.P], 'lr': 0.001 * 0.1, "name": "pose"}]
        # self.optimizer = torch.optim.Adam(self.l+self.l_cam)

    def get_RT(self, idx, learn_1=True):
        if learn_1:
            if idx==0:
                pose = self.P[idx:idx+1].detach()
            else:
                pose = self.P[idx:idx+1]
        else:
            if idx<=1:
                pose = self.P[idx:idx+1].detach()
            else:
                pose = self.P[idx:idx+1]
        # pose = self.P[idx:idx+1]
        # print(idx, pose)
        return pose

    # def get_RT_test(self, idx):
    #     pose = self.test_P[idx]
    #     return pose

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            print(self.active_sh_degree)

    def create_from_pcd(self, pcd : BasicPointCloud, conf=None, scale_gaussian=None, spatial_lr_scale : float= None):
        self.spatial_lr_scale = spatial_lr_scale
        self.rgb = np.asarray(pcd.colors)
        conf = torch.tensor(conf).float().cuda()
        self.conf = torch.clip(conf, 0, conf.max())
        print("confidence:", self.conf.shape)
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        # breakpoint()
        # if scale_gaussian is not None:
        #     mean3_sq_dist = torch.from_numpy(scale_gaussian**2).float().cuda()
        #     scales = torch.log(torch.sqrt(mean3_sq_dist))[..., None]
        # else:
        #     dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        #     scales = torch.log(torch.sqrt(dist2))[...,None]
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        if scale_gaussian is not None:
            print("Use depth to initialize scaling")
            mean3_sq_dist = torch.from_numpy(scale_gaussian**2).float().cuda()
            dist2 = torch.min(mean3_sq_dist, dist2)
        scales = torch.log(torch.sqrt(dist2))[...,None]

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.confidence = torch.ones_like(opacities, device="cuda")


    def create_from_pcd_separate(self, points, colors, conf, scale_gaussian=None, spatial_lr_scale : float= None):
        self.spatial_lr_scale = spatial_lr_scale
        self.rgb = colors
        conf = torch.tensor(conf).float().cuda()
        self.conf = torch.clip(conf, 0, conf.max())
        print("confidence:", self.conf.shape)
        fused_point_cloud = torch.tensor(points).float().cuda()
        fused_color = RGB2SH(torch.tensor(colors).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
                
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(points).float().cuda()), 0.0000001)
        if scale_gaussian is not None:
            mean3_sq_dist = torch.from_numpy(scale_gaussian**2).float().cuda()
            dist2 = torch.min(mean3_sq_dist, dist2)
        scales = torch.log(torch.sqrt(dist2))[...,None]

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.confidence = torch.ones_like(opacities, device="cuda")

    # New added method, called when new view is added.
    def add_points_from_xyz_rgb(self, xyz, rgb, scale_gaussian=None, spatial_lr_scale: float = None):
        # need to update cameras_extent as well.
        if spatial_lr_scale is not None:
            ic(
                "warning: Set spatial_lr_scale, not setting updated spatial_lr_scale might lead to underperformace "
            )
            self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(xyz).float().cuda()
        fused_color = RGB2SH(torch.tensor(rgb).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points added : ", fused_point_cloud.shape[0])


        # if scale_gaussian is not None:
        #     mean3_sq_dist = scale_gaussian**2
        #     scales = torch.tile(torch.log(torch.sqrt(torch.from_numpy(mean3_sq_dist).float().cuda()))[..., None], (1, 1))
        # else:
        #     xyz_in =torch.from_numpy(np.asarray(xyz)).float().cuda()
        #     xyz_all = torch.cat((xyz_in, self._xyz), 0)
        #     dist2 = torch.clamp_min(distCUDA2(xyz_all), 0.0000001)[:xyz_in.shape[0]]
        #     scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 1)
        
        xyz_in =torch.from_numpy(np.asarray(xyz)).float().cuda()
        xyz_all = torch.cat((xyz_in, self._xyz), 0)
        dist2 = torch.clamp_min(distCUDA2(xyz_all), 0.0000001)[:xyz_in.shape[0]]

        if scale_gaussian is not None:
            mean3_sq_dist = torch.from_numpy(scale_gaussian**2).float().cuda()
            dist2 = torch.min(mean3_sq_dist, dist2)
            
        scales = torch.log(torch.sqrt(dist2))[..., None]

            
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(self._xyz.detach().cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(self.rgb)
        # o3d.io.write_point_cloud(f"points_{self._xyz.shape[0]}_wait_add.ply", pcd)

        # pcd = o3d.geometry.PointCloud()
        # points3d = torch.cat((self._xyz, fused_point_cloud))
        # colors3d = np.concatenate((self.rgb, rgb))
        # pcd.points = o3d.utility.Vector3dVector(points3d.detach().cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(colors3d)
        # o3d.io.write_point_cloud(f"points_{points3d.shape[0]}.ply", pcd)
        # breakpoint()

        self.rgb= np.concatenate((self.rgb, rgb))

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        # new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # new_features_dc = nn.Parameter(
        #     features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        # )
        # new_features_rest = nn.Parameter(
        #     features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        # )
        # new_scaling = nn.Parameter(scales.requires_grad_(True))
        # new_rotation = nn.Parameter(rots.requires_grad_(True))
        # new_opacities = nn.Parameter(opacities.requires_grad_(True))
        # new_pose = nn.Parameter(torch.zeros(0).cuda().requires_grad_(True))
        new_xyz = fused_point_cloud
        new_features_dc = features[:, :, 0:1].transpose(1, 2).contiguous()
        new_features_rest = features[:, :, 1:].transpose(1, 2).contiguous()
        new_scaling = scales
        new_rotation = rots
        new_opacities = opacities
        new_pose = torch.zeros(0).cuda()

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_pose,
        )
        del new_xyz
        del new_features_dc
        del new_features_rest
        del new_scaling
        del new_rotation
        del new_opacities
        del new_pose

    def add_local_gaussian(self, xyz, rot, local_gaussian, masks, inverse_apply=False):
        masks = (np.array(masks))<1
        masks = masks.flatten() 
        if not inverse_apply:
            if xyz.shape[0]>masks.shape[0]:
                masks_padding = np.ones((xyz.shape[0]-masks.shape[0]))>0
                masks = np.concatenate([masks, masks_padding])
            # breakpoint()
            new_xyz = xyz[masks]
            new_features_dc = local_gaussian._features_dc[masks]
            new_features_rest = local_gaussian._features_rest[masks]
            new_opacities = local_gaussian._opacity[masks]
            new_scaling = local_gaussian._scaling[masks]
            new_rotation = rot[masks]
            self.conf= torch.concat((self.conf, local_gaussian.conf[masks]))
            print(new_xyz.shape, new_features_dc.shape, new_features_rest.shape, new_opacities.shape, new_scaling.shape, new_rotation.shape)
            print("confidence:", self.conf.shape)
            new_pose = torch.zeros(0).cuda()
        else:
            if self._xyz.shape[0]>masks.shape[0]:
                masks_padding = np.ones((self._xyz.shape[0]-masks.shape[0]))>0
                masks = np.concatenate([masks_padding, masks])
            # breakpoint()
            # self._xyz = self._xyz[masks]
            # self._features_dc = self._features_dc[masks]
            # self._features_rest = self._features_rest[masks]
            # self._opacity = self._opacity[masks]
            # self._scaling = self._scaling[masks]
            # self._rotation = self._rotation[masks]
            optimizable_tensors = self._prune_optimizer(masks)

            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            new_xyz = xyz
            new_features_dc = local_gaussian._features_dc
            new_features_rest = local_gaussian._features_rest
            new_opacities = local_gaussian._opacity
            new_scaling = local_gaussian._scaling
            new_rotation = rot
            self.conf= torch.concat((self.conf[masks], local_gaussian.conf))
            print(new_xyz.shape, new_features_dc.shape, new_features_rest.shape, new_opacities.shape, new_scaling.shape, new_rotation.shape)
            print("confidence:", self.conf.shape)
            new_pose = torch.zeros(0).cuda()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_pose,
        )
        del new_xyz
        del new_features_dc
        del new_features_rest
        del new_scaling
        del new_rotation
        del new_opacities
        del new_pose
        


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        NN=10
        print("xyz training lr", training_args.position_lr_init)
        self.l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr*NN, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr*NN / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr*NN, "name": "rotation"},
        ]

        print("camera training lr", training_args.rotation_lr)
        self.l_cam = [{'params': [self.P],'lr': training_args.rotation_lr *0.1, "name": "pose"},]
        # self.l_cam = [{'params': [self.P],'lr': 0, "name": "pose"},]

        # self.l += l_cam
        # +self.l_cam

        self.optimizer = torch.optim.Adam(self.l+self.l_cam, lr=0.0, eps=1e-15)
        # self.cam_optimizer = torch.optim.Adam(self.l_cam)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.cam_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr*0.1,
                                                    lr_final=training_args.rotation_lr*0.001,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
    # def get_optimizer(self, optimize_pose=False):
    #     if optimize_pose:
    #         return self.cam_optimizer
    #     else:
    #         return self.optimizer
        

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "pose":
                lr = self.cam_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
        # for param_group in self.optimizer.param_groups:
        #     print(param_group["name"], param_group['lr'])

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        # for i in range(self._scaling.shape[1]):
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()        
        if self._scaling.shape[1] == 1:
            scale = torch.tile(self._scaling, (1, 3)).detach().cpu().numpy()
        else:
            scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


    def reduce_opacity(self):
        opacities_new = inverse_sigmoid(self.get_opacity*0.1)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


    def reset_scaling(self):
        scaling_new = torch.log(torch.min(self.get_scaling, torch.ones_like(self.get_scaling)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(scaling_new, "scaling")
        self._scaling = optimizable_tensors["scaling"]

    def transform_gaussian(self, xyz, rot):
        optimizable_tensors1 = self.replace_tensor_to_optimizer(xyz, "xyz")
        self._xyz = optimizable_tensors1["xyz"]
        optimizable_tensors2 = self.replace_tensor_to_optimizer(rot, "rotation")
        self._rotation = optimizable_tensors2["rotation"]

    # def reduce_scaling(self, value):
    #     scaling_new = torch.log(torch.min(self.get_scaling, torch.ones_like(self.get_scaling)*value))
    #     optimizable_tensors = self.replace_tensor_to_optimizer(scaling_new, "scaling")
    #     self._scaling = optimizable_tensors["scaling"]

    def reduce_scaling(self, mask, value):
        scaling_new = torch.log(self.get_scaling)
        # breakpoint()
        scaling_new[mask] = torch.log(torch.min(self.get_scaling[mask], value))
        # scaling_new[mask] = torch.log(torch.min(self.get_scaling[mask], (torch.ones_like(self.get_scaling)*value)[mask]))
        optimizable_tensors = self.replace_tensor_to_optimizer(scaling_new, "scaling")
        self._scaling = optimizable_tensors["scaling"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"]=="pose":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        if torch.sum(mask)>0:
            print("pruned point number", torch.sum(mask))
        valid_points_mask = ~mask
        self.conf = self.conf[valid_points_mask.cpu().numpy()]
        # breakpoint()
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.confidence = self.confidence[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def cat_pose_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"]=="pose":
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_pose
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "pose": new_pose
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        print("Total number of points : ", self._xyz.shape[0])

        del d
        del optimizable_tensors

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.confidence = torch.cat([self.confidence, torch.ones(new_opacities.shape, device="cuda")], 0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        # grads = self.xyz_gradient_accum / self.denom
        # grads[grads.isnan()] = 0.0

        # self.densify_and_clone(grads, max_grad, extent)
        # self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if prune_mask.sum()>0:
            print(f"pruning {prune_mask.sum()} points due to small opacity")
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # conf_mask = self.conf < 0.01
        # prune_mask = torch.logical_or(prune_mask, conf_mask.squeeze())
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
