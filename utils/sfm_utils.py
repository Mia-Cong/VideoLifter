import math
import os
import time
import scipy
import torch
import cv2
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
from plyfile import PlyData, PlyElement
import torchvision.transforms as tvf
import roma
import re
from pathlib import Path
from typing import List, NamedTuple, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm  # Add this import at the top of your file

# from dust3r.utils.image import _resize_pil_image
# from mast3r.retrieval.processor import Retriever
# from dust3r.utils.device import to_numpy
import trimesh
from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, rotmat2qvec
# from utils.dust3r_utils import storePly
from utils.utils_flow.matching import global_correlation_softmax, local_correlation_softmax
from utils.utils_flow.geometry import coords_grid
from utils.utils_flow.flow_viz import flow_to_image, flow_to_color, save_vis_flow_tofile
import torchvision.transforms.functional as tf


# from mast3r.utils.matching import global_correlation_softmax, local_correlation_softmax
# from mast3r.utils.geometry import coords_grid


try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def save_time(time_dir, process_name, sub_time):
    if isinstance(time_dir, str):
        time_dir = Path(time_dir)
    time_dir.mkdir(parents=True, exist_ok=True)
    minutes, seconds = divmod(sub_time, 60)
    formatted_time = f"{int(minutes)} min {int(seconds)} sec"  
    with open(time_dir / f'train_time.txt', 'a') as f:
        f.write(f'{process_name}: {formatted_time}\n')

def split_train_test_org(image_files, llffhold=8, n_views=None, scene=None):
    print(">> Spliting Train-Test Set: ")
    test_indices = [idx for idx in range(len(image_files)) if idx % llffhold == 0]
    non_test_indices = [idx for idx in range(len(image_files)) if idx % llffhold != 0]
    if n_views is None or n_views == 0:
        n_views = len(non_test_indices)
    sparse_indices = np.linspace(0, len(non_test_indices) - 1, n_views, dtype=int)
    train_indices = [non_test_indices[i] for i in sparse_indices]
    # print(" - sparse_indexs:  ", sparse_indices)
    print(" - train_indices:  ", train_indices)
    print(" - test_indices:   ", test_indices)
    train_img_files = [image_files[i] for i in train_indices]
    test_img_files = [image_files[i] for i in test_indices]

    return train_img_files, test_img_files


def split_train_test(image_files, llffhold=8, n_views=None, scene=None):
    print(">> Spliting Train-Test Set: ")
    ids = np.arange(len(image_files))
    llffhold = 2 if scene=="Family" else 8
    test_indices = ids[int(llffhold/2)::llffhold]
    non_test_indices = np.array([i for i in ids if i not in test_indices])
    # breakpoint()
    if n_views is None or n_views == 0:
        n_views = len(non_test_indices)
    sparse_indices = np.linspace(0, len(non_test_indices) - 1, n_views, dtype=int)
    train_indices = [non_test_indices[i] for i in sparse_indices]
    print(" - sparse_idx:         ", sparse_indices, len(sparse_indices))
    print(" - train_set_indices:  ", train_indices, len(train_indices))
    print(" - test_set_indices:   ", test_indices, len(test_indices))
    train_img_files = [image_files[i] for i in train_indices]
    test_img_files = [image_files[i] for i in test_indices]

    return train_img_files, test_img_files

        # sample_rate = 2 if "Family" in os.path else 8
        # # sample_rate = 8
        # ids = np.arange(len(cam_infos))
        # i_test = ids[int(sample_rate/2)::sample_rate]
        # i_train = np.array([i for i in ids if i not in i_test])
        # train_cam_infos = [cam_infos[i] for i in i_train]
        # test_cam_infos = [cam_infos[i] for i in i_test]


def get_sorted_image_files(image_dir: str) -> Tuple[List[str], List[str]]:
    """
    Get sorted image files from the given directory.

    Args:
        image_dir (str): Path to the directory containing images.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - List of sorted image file paths
            - List of corresponding file suffixes
    """
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.JPG', '.PNG'}
    image_path = Path(image_dir)
    
    def extract_number(filename):
        match = re.search(r'\d+', filename.stem)
        return int(match.group()) if match else float('inf')
    
    image_files = [
        str(f) for f in image_path.iterdir()
        if f.is_file() and f.suffix.lower() in allowed_extensions
    ]
    
    sorted_files = sorted(image_files, key=lambda x: extract_number(Path(x)))
    suffixes = [Path(file).suffix for file in sorted_files]
    
    return sorted_files, suffixes[0]
    # filenames = sorted(glob(inference_dir + '/*.png') + glob(inference_dir + '/*.jpg'))


# def get_scene_graph(scenegraph_type, winsize, refid, win_cyclic, retrieval_model, image_files, model, device):
#     scene_graph_params = [scenegraph_type]
#     if scenegraph_type in ["swin", "logwin"]:
#         scene_graph_params.append(str(winsize))
#     elif scenegraph_type == "oneref":
#         scene_graph_params.append(str(refid))
#     elif scenegraph_type == "retrieval":
#         scene_graph_params.append(str(winsize))  # Na
#         scene_graph_params.append(str(refid))  # k

#     if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
#         scene_graph_params.append('noncyclic')
#     scene_graph = '-'.join(scene_graph_params)

#     sim_matrix = None
#     if 'retrieval' in scenegraph_type:
#         assert retrieval_model is not None
#         retriever = Retriever(retrieval_model, backbone=model, device=device)
#         with torch.no_grad():
#             sim_matrix = retriever(image_files)

#         # Cleanup
#         del retriever
#         torch.cuda.empty_cache()

#     return scene_graph, sim_matrix


def rigid_points_registration(pts1, pts2, conf=None):
    R, T, s = roma.rigid_points_registration(
        pts1.reshape(-1, 3), pts2.reshape(-1, 3), weights=conf, compute_scaling=True)
    return s, R, T  # return un-scaled (R, T)

# def round_python3(number):
#     rounded = round(number)
#     if abs(number - rounded) == 0.5:
#         return 2.0 * round(number / 2.0)
#     return rounded

def init_filestructure(save_path, n_views=None):
    if n_views is not None and n_views != 0:        
        sparse_0_path = save_path / f'sparse_{n_views}/0'    
        sparse_1_path = save_path / f'sparse_{n_views}/1'       
        print(f'>> Doing {n_views} views reconstrution!')
    elif n_views is None or n_views == 0:
        sparse_0_path = save_path / 'sparse_0/0'    
        sparse_1_path = save_path / 'sparse_0/1'
        print(f'>> Doing full views reconstrution!')

    save_path.mkdir(exist_ok=True, parents=True)
    sparse_0_path.mkdir(exist_ok=True, parents=True)    
    sparse_1_path.mkdir(exist_ok=True, parents=True)
    return save_path, sparse_0_path, sparse_1_path


# def load_images(folder_or_list, size, square_ok=False, verbose=True):
#     """ open and convert all images in a list or folder to proper input format for DUSt3R
#     """
#     if isinstance(folder_or_list, str):
#         if verbose:
#             print(f'>> Loading images from {folder_or_list}')
#         root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

#     elif isinstance(folder_or_list, list):
#         if verbose:
#             print(f'>> Loading a list of {len(folder_or_list)} images')
#         root, folder_content = '', folder_or_list

#     else:
#         raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

#     supported_images_extensions = ['.jpg', '.jpeg', '.png', '.JPG', 'PNG']
#     if heif_support_enabled:
#         supported_images_extensions += ['.heic', '.heif']
#     supported_images_extensions = tuple(supported_images_extensions)

#     imgs = []
#     for path in folder_content:
#         if not path.lower().endswith(supported_images_extensions):
#             continue
#         img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
#         W1, H1 = img.size
#         if size == 224:
#             # resize short side to 224 (then crop)
#             img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
#         else:
#             # resize long side to 512
#             img = _resize_pil_image(img, size)
#         W, H = img.size
#         cx, cy = W//2, H//2
#         if size == 224:
#             half = min(cx, cy)
#             img = img.crop((cx-half, cy-half, cx+half, cy+half))
#         else:
#             halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
#             if not (square_ok) and W == H:
#                 halfh = 3*halfw/4
#             img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

#         W2, H2 = img.size
#         if verbose:
#             print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
#         imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
#             [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

#     assert imgs, 'no images foud at '+root
#     if verbose:
#         print(f' (Found {len(imgs)} images)')
#     return imgs, (W1,H1)








# Save image transformations
# def save_extrinsic(extrinsics_w2c, img_files, sparse_path, image_suffix):
#     images_file = sparse_path / 'images.txt'
#     with open(images_file, 'w') as f:
#         f.write("# Image list with two lines of data per image:\n")
#         f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
#         f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
#         for i in range(extrinsics_w2c.shape[0]):
#             name = Path(img_files[i]).stem
#             rotation_matrix = extrinsics_w2c[i, :3, :3]
#             qw, qx, qy, qz = rotmat2qvec(rotation_matrix)
#             tx, ty, tz = extrinsics_w2c[i, :3, 3]
#             f.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {name}{image_suffix}\n\n")

import collections
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])
      
# def save_extrinsic(sparse_path, extrinsics_w2c, img_files, image_suffix):
#     images_bin_file = sparse_path / 'images.bin'
#     images_txt_file = sparse_path / 'images.txt'
#     images = {}
    
#     for i, (w2c, img_file) in enumerate(zip(extrinsics_w2c, img_files), start=1):  # Start enumeration from 1
#         name = Path(img_file).stem + image_suffix
#         rotation_matrix = w2c[:3, :3]
#         qvec = rotmat2qvec(rotation_matrix)
#         tvec = w2c[:3, 3]
        
#         images[i] = BaseImage(
#             id=i,
#             qvec=qvec,
#             tvec=tvec,
#             camera_id=i,
#             name=name,
#             xys=[],  # Empty list as we don't have 2D point information
#             point3D_ids=[]  # Empty list as we don't have 3D point IDs
#         )
    
#     write_images_binary(images, images_bin_file)
#     write_images_text(images, images_txt_file)


# def save_intrinsics(sparse_path, focals, org_imgs_shape, imgs_shape, save_focals=False):
#     org_width, org_height = org_imgs_shape
#     scale_factor_x = org_width / imgs_shape[2]
#     scale_factor_y = org_height / imgs_shape[1]
#     cameras_bin_file = sparse_path / 'cameras.bin'
#     cameras_txt_file = sparse_path / 'cameras.txt'

#     cameras = {}
#     for i, focal in enumerate(focals, start=1):  # Start enumeration from 1
#         cameras[i] = Camera(
#             id=i,
#             model="PINHOLE",
#             width=org_width,
#             height=org_height,
#             params=[focal*scale_factor_x, focal*scale_factor_y, org_width/2, org_height/2]
#         )    
#     print(f' - scaling focal: ({focal}, {focal}) --> ({focal*scale_factor_x}, {focal*scale_factor_y})' )
#     write_cameras_binary(cameras, cameras_bin_file)
#     write_cameras_text(cameras, cameras_txt_file)
#     if save_focals:
#         np.save(sparse_path / 'non_scaled_focals.npy', focals)


# def save_points3D(sparse_path, imgs, pts3d, masks, use_masks=True, save_all_pts=False):
#     points3D_bin_file = sparse_path / 'points3D.bin'
#     points3D_txt_file = sparse_path / 'points3D.txt'
#     points3D_ply_file = sparse_path / 'points3D.ply'
    

#     # Convert inputs to numpy arrays
#     imgs = to_numpy(imgs)
#     pts3d = to_numpy(pts3d)
#     masks = to_numpy(masks)

#     # Process points and colors
#     if use_masks:
#         pts = np.concatenate([p[m] for p, m in zip(pts3d, masks.reshape(masks.shape[0], -1))])
#         col = np.concatenate([p[m] for p, m in zip(imgs, masks)])
#     else:
#         pts = np.array(pts3d)
#         col = np.array(imgs)
#     pts = pts.reshape(-1, 3)
#     col = col.reshape(-1, 3) * 255.

#     # points3D = {}
#     # for i, (pt, color) in enumerate(zip(pts, col), start=1):  # Start enumeration from 1
#     #     points3D[i] = Point3D(
#     #         id=i,
#     #         xyz=pt,
#     #         rgb=color,
#     #         error=0,
#     #         image_ids=[],  # Empty list as we don't have image IDs
#     #         point2D_idxs=[]  # Empty list as we don't have 2D point indices
#     #     )

#     # write_points3D_binary(points3D, points3D_bin_file)
#     # write_points3D_text(points3D, points3D_txt_file)
#     storePly(points3D_ply_file, pts, col)
#     if save_all_pts:
#         np.save(sparse_path / 'points3D_all.npy', pts3d)
    
#     # Write pts_num.txt
#     pts_num_file = sparse_path / f'pts_num_{pts.shape[0]}.txt'  # New file for pts_num
#     with open(pts_num_file, 'w') as f:
#         f.write(f"After downsampling: {pts.shape[0]}\n")
#         f.write(f"Vanilla points num: {pts3d.reshape(-1, 3).shape[0]}\n")
    
#     return pts.shape[0]

# Save images and masks
def save_pair_confs_masks(img_files, masks, masks_path, image_suffix):
    for i, (name, mask) in enumerate(zip(img_files, masks)):
        imgname = Path(name).stem
        mask_save_path = masks_path / f"{imgname}{image_suffix}"
        mask = np.repeat(np.expand_dims(mask, -1), 3, axis=2) * 255
        PIL.Image.fromarray(mask.astype(np.uint8)).save(mask_save_path)


# Save images and masks
def save_images_and_masks(sparse_0_path, n_views, imgs, global_conf_masks, pair_conf_masks, co_vis_masks, combined_masks, overlapping_masks, image_files, image_suffix):

    images_path       = sparse_0_path / f'imgs_{n_views}'    
    global_conf_masks_path   = sparse_0_path / f'global_conf_masks_{n_views}'
    pair_conf_masks_path = sparse_0_path / f'pair_conf_masks_{n_views}'
    co_vis_masks_path = sparse_0_path / f'co_vis_masks_{n_views}'
    combined_masks_path  = sparse_0_path / f'combined_masks_{n_views}'
    overlapping_masks_path = sparse_0_path / f'overlapping_masks_{n_views}'

    images_path.mkdir(exist_ok=True, parents=True)
    global_conf_masks_path.mkdir(exist_ok=True, parents=True)
    pair_conf_masks_path.mkdir(exist_ok=True, parents=True)
    co_vis_masks_path.mkdir(exist_ok=True, parents=True)
    combined_masks_path.mkdir(exist_ok=True, parents=True)
    overlapping_masks_path.mkdir(exist_ok=True, parents=True)

    for i, (image, name, global_conf_mask, pair_conf_mask, co_vis_mask, combined_mask, overlapping_mask) in enumerate(zip(imgs, image_files, global_conf_masks, pair_conf_masks, co_vis_masks, combined_masks, overlapping_masks)):
        imgname = Path(name).stem
        image_save_path = images_path / f"{imgname}{image_suffix}"
        global_conf_mask_save_path = global_conf_masks_path / f"{imgname}{image_suffix}"
        pair_conf_mask_save_path = pair_conf_masks_path / f"{imgname}{image_suffix}"
        co_vis_mask_save_path = co_vis_masks_path / f"{imgname}{image_suffix}"
        combined_mask_save_path = combined_masks_path / f"{imgname}{image_suffix}"
        overlapping_mask_save_path = overlapping_masks_path / f"{imgname}{image_suffix}"

        # Save overlapping masks
        overlapping_mask = np.repeat(np.expand_dims(overlapping_mask, -1), 3, axis=2) * 255
        PIL.Image.fromarray(overlapping_mask.astype(np.uint8)).save(overlapping_mask_save_path)

        # Save images   
        rgb_image = cv2.cvtColor(image * 255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(image_save_path), rgb_image)

        # Save conf masks
        global_conf_mask = np.repeat(np.expand_dims(global_conf_mask, -1), 3, axis=2) * 255
        PIL.Image.fromarray(global_conf_mask.astype(np.uint8)).save(global_conf_mask_save_path)
        pair_conf_mask = np.repeat(np.expand_dims(pair_conf_mask, -1), 3, axis=2) * 255
        PIL.Image.fromarray(pair_conf_mask.astype(np.uint8)).save(pair_conf_mask_save_path)

        # Save co-vis masks
        co_vis_mask = np.repeat(np.expand_dims(co_vis_mask, -1), 3, axis=2) * 255
        PIL.Image.fromarray(co_vis_mask.astype(np.uint8)).save(co_vis_mask_save_path)

        # Save combined masks
        combined_mask = np.repeat(np.expand_dims(combined_mask, -1), 3, axis=2) * 255
        PIL.Image.fromarray(combined_mask.astype(np.uint8)).save(combined_mask_save_path)


def read_focal_from_cameras_txt(file_path):
    """
    Reads focal lengths from a cameras.txt file where the camera model is 'PINHOLE'.

    Args:
        file_path (str): Path to the cameras.txt file.

    Returns:
        List[float]: A list of focal lengths for 'PINHOLE' camera models.
    """
    focals = []
    with open(file_path, 'r') as file:
        for line in file:
            # Skip comment lines
            if line.startswith('#'):
                continue
            
            # Split the line into parts
            parts = line.strip().split()
            
            # Check if the line has enough parts and the model is 'PINHOLE'
            if len(parts) >= 5 and parts[1] == 'PINHOLE':
                # Extract the focal length (5th element, index 4)
                focal_length = float(parts[4])
                focals.append(focal_length)
    
    return focals


























# # Save point cloud with normals
# def save_pointcloud_with_normals(imgs, pts3d, masks, sparse_path):
#     pc = get_point_cloud(imgs, pts3d, masks)
#     default_normal = [0, 1, 0]
#     vertices = pc.vertices
#     colors = pc.colors
#     normals = np.tile(default_normal, (vertices.shape[0], 1))
#     save_path = sparse_path / 'points3D.ply'
#     header = """ply
# format ascii 1.0
# element vertex {}
# property float x
# property float y
# property float z
# property uchar red
# property uchar green
# property uchar blue
# property float nx
# property float ny
# property float nz
# end_header
# """.format(len(vertices))
#     with open(save_path, 'w') as f:
#         f.write(header)
#         for vertex, color, normal in zip(vertices, colors, normals):
#             f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {int(color[0])} {int(color[1])} {int(color[2])} {normal[0]} {normal[1]} {normal[2]}\n")

# Generate point cloud
# def get_point_cloud(imgs, pts3d, mask):
#     imgs = to_numpy(imgs)
#     pts3d = to_numpy(pts3d)
#     mask = to_numpy(mask)
#     pts = np.concatenate([p[m] for p, m in zip(pts3d, mask.reshape(mask.shape[0], -1))])
#     col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
#     pts = pts.reshape(-1, 3)[::3]
#     col = col.reshape(-1, 3)[::3]
#     normals = np.tile([0, 1, 0], (pts.shape[0], 1))
#     pct = trimesh.PointCloud(pts, colors=col)
#     pct.vertices_normal = normals
#     return pct









# Save camera information
# def save_cameras(focals, principal_points, sparse_path, org_imgs_shape, imgs_shape):
#     org_width, org_height = org_imgs_shape
#     scale_factor_x = org_width  / imgs_shape[2]
#     scale_factor_y = org_height  / imgs_shape[1]
#     cameras_file = sparse_path / 'cameras.txt'
#     with open(cameras_file, 'w') as f:
#         f.write("# Camera list with one line of data per camera:\n")
#         f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
#         for i, (focal, pp) in enumerate(zip(focals, principal_points)):
#             f.write(f"{i} PINHOLE {org_width} {org_height} {focal*scale_factor_x} {focal*scale_factor_y} {org_width/2} {org_height/2}\n")


def compute_global_correspondence(feature0, feature1, pred_bidir_flow=False):
    """
    Compute global correspondence between two feature maps.
    
    Args:
        feature0 (torch.Tensor): First feature map of shape [B, C, H, W]
        feature1 (torch.Tensor): Second feature map of shape [B, C, H, W]
    
    Returns:
        torch.Tensor: Correspondence map of shape [B, 2, H, W]
    """
    # Compute flow and probability
    flow, prob = global_correlation_softmax(feature0, feature1, pred_bidir_flow=pred_bidir_flow)
    
    # Get initial grid
    b, _, h, w = feature0.shape
    init_grid = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
    
    # Compute correspondence
    correspondence = flow + init_grid
    
    return correspondence, flow

def save_flow_visualization(flow, save_path):
    """
    Convert flow to color representation and save as an image.

    Args:
        flow (torch.Tensor): Flow map of shape [B, 2, H, W]
        save_path (str or Path): Path to save the flow visualization
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert flow to numpy and create visualization
    flow_np = flow.cpu().numpy()
    flow_rgb = flow_to_color(flow_np[0].transpose(1, 2, 0))
    
    # Save the flow visualization
    cv2.imwrite(str(save_path), cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR))

def compute_local_correspondence(feature0, feature1, local_radius=4, chunk_size=32):
    """
    Compute local correspondence and flow between two feature maps using a sliding window approach.
    
    Args:
        feature0 (torch.Tensor): First feature map of shape [B, C, H, W]
        feature1 (torch.Tensor): Second feature map of shape [B, C, H, W]
        local_radius (int): Radius for local correlation window
        chunk_size (int): Number of rows to process at once to save memory
    
    Returns:
        tuple: (correspondence, flow)
            - correspondence (torch.Tensor): Correspondence map of shape [B, 2, H, W]
            - flow (torch.Tensor): Flow map of shape [B, 2, H, W]
    """
    b, c, h, w = feature0.shape
    device = feature0.device
    
    # Initialize the output correspondence and flow maps
    correspondence = torch.zeros(b, 2, h, w, device=device)
    flow = torch.zeros(b, 2, h, w, device=device)
    
    # Process the image in chunks to save memory
    for i in range(0, h, chunk_size):
        end = min(i + chunk_size, h)
        
        # Extract chunks from both feature maps
        chunk0 = feature0[:, :, i:end, :]
        chunk1 = feature1[:, :, max(0, i-local_radius):min(h, end+local_radius), :]
        
        # Compute local correlation for the chunk
        flow_chunk, _ = local_correlation_softmax(chunk0, chunk1, local_radius)
        
        # Convert flow to correspondence
        init_grid_chunk = coords_grid(b, end-i, w, device=device)
        correspondence_chunk = flow_chunk + init_grid_chunk
        
        # Update the correspondence and flow maps
        correspondence[:, :, i:end, :] = correspondence_chunk
        flow[:, :, i:end, :] = flow_chunk
    
    return correspondence, flow

# You can add any additional utility functions here if needed


def save_feature_visualization(feature, save_path, title='Feature Visualization'):
    """
    Save the visualization of feature maps to a file.

    Parameters:
    - feature: torch.Tensor, the feature map to visualize, expected shape (C, H, W)
    - save_path: str, the path to save the visualization image
    - title: str, the title for the visualization
    """
    feature_np = feature.cpu().numpy()  # Convert to numpy array
    num_channels = feature_np.shape[0]
    plt.figure(figsize=(num_channels * 3, 3))
    for i in range(num_channels):
        plt.subplot(1, num_channels, i + 1)
        plt.imshow(feature_np[i], cmap='viridis')
        plt.axis('off')
    plt.suptitle(title)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def save_merged_feature_visualization(feature, save_path, title='Merged Feature Visualization', method='mean'):
    """
    Save the merged visualization of feature maps to a file.

    Parameters:
    - feature: torch.Tensor, the feature map to visualize, expected shape (C, H, W)
    - save_path: str, the path to save the visualization image
    - title: str, the title for the visualization
    - method: str, the method to merge channels ('mean' or 'max')
    """
    feature_np = feature.cpu().numpy()  # Convert to numpy array

    if method == 'mean':
        merged_feature = np.mean(feature_np, axis=0)
    elif method == 'max':
        merged_feature = np.max(feature_np, axis=0)
    else:
        raise ValueError("Method must be 'mean' or 'max'.")

    plt.figure(figsize=(5, 5))
    plt.imshow(merged_feature, cmap='viridis')
    plt.axis('off')
    plt.title(title)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def feature_save(tensor,path_name,name):
    # tensor = torchvision.utils.make_grid(tensor.transpose(1,0))
    tensor = torch.mean(tensor,dim=1)
    inp = tensor.detach().cpu().numpy().transpose(1,2,0)
    inp = inp.squeeze(2)
    inp = (inp - np.min(inp)) / (np.max(inp) - np.min(inp))
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    inp = cv2.applyColorMap(np.uint8(inp * 255.0),cv2.COLORMAP_JET)
    cv2.imwrite(path_name  + name, inp)


def cal_co_vis_mask(points, depths, curr_depth_map, depth_threshold, camera_intrinsics, extrinsics_w2c):

    h, w = curr_depth_map.shape
    overlapping_mask = np.zeros((h, w), dtype=bool)
    # Project 3D points to image j
    points_2d, _ = project_points(points, camera_intrinsics, extrinsics_w2c)
    
    # Check if points are within image bounds
    valid_points = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                   (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        
    # Check depth consistency using vectorized operations
    valid_points_2d = points_2d[valid_points].astype(int)
    valid_depths = depths[valid_points]

    # Extract x and y coordinates
    x_coords, y_coords = valid_points_2d[:, 0], valid_points_2d[:, 1]

    # Compute depth differences
    depth_differences = np.abs(valid_depths - curr_depth_map[y_coords, x_coords])

    # Create a mask for points where the depth difference is below the threshold
    consistent_depth_mask = depth_differences < depth_threshold

    # Update the overlapping masks using the consistent depth mask
    overlapping_mask[y_coords[consistent_depth_mask], x_coords[consistent_depth_mask]] = True

    return overlapping_mask

def normalize_depth(depth_map):
    """Normalize the depth map to a range between 0 and 1."""
    return (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

def compute_overlapping_mask_2(sorted_conf_indices, depthmaps, pointmaps, camera_intrinsics, extrinsics_w2c, image_sizes, depth_threshold=0.1):

    num_images, h, w, _ = image_sizes
    pointmaps = pointmaps.reshape(num_images, h, w, 3)
    overlapping_masks = np.zeros((num_images, h, w), dtype=bool)
    
    for i, curr_map_idx in tqdm(enumerate(sorted_conf_indices), total=len(sorted_conf_indices)):

        # if frame_idx is 0, set its occ_mask to be all False
        if i == 0:
            continue

        # get before and after curr_frame's indices
        idx_before = sorted_conf_indices[:i]
        # idx_after = sorted_conf_indices[i+1:]

        # get partial pointmaps and depthmaps
        points_before = pointmaps[idx_before].reshape(-1, 3)
        depths_before = depthmaps[idx_before].reshape(-1)    
        # points_after = pointmaps[idx_after].reshape(-1, 3)        
        # depths_after = depthmaps[idx_after].reshape(-1)
        # get current frame's depth map
        curr_depth_map = depthmaps[curr_map_idx].reshape(h, w)

        # normalize depth for comparison
        depths_before = normalize_depth(depths_before)
        # depths_after = normalize_depth(depths_after)
        curr_depth_map = normalize_depth(curr_depth_map)

        # before_mask = overlapping_masks[idx_before]
        # after_mask = overlapping_masks[idx_after]
        # curr_mask = before_mask & after_mask
        

        before_mask = cal_co_vis_mask(points_before, depths_before, curr_depth_map, depth_threshold, camera_intrinsics[curr_map_idx], extrinsics_w2c[curr_map_idx])
        # after_mask = cal_co_vis_mask(points_after, depths_after, camera_intrinsics[i], extrinsics_w2c[i], curr_depth_map, depth_threshold)
        
        # white/True means co-visible area: we need to remove
        # black/False means occulusion/bad geometry area: we need to keep
        # 白白=白
        # 白黑=白
        # 黑白=黑
        # 黑黑=黑
        overlapping_masks[curr_map_idx] = before_mask# & after_mask
        
    return overlapping_masks

def compute_overlapping_mask(depthmaps, pointmaps, camera_intrinsics, extrinsics_w2c, image_sizes, depth_threshold=0.1):
    num_images, h, w, _ = image_sizes
    pointmaps = pointmaps.reshape(num_images, h, w, 3)
    overlapping_masks = np.zeros((num_images, h, w), dtype=bool)
    
    for i in range(num_images):
        # Exclude the current pointmap
        points_3d = pointmaps[np.arange(num_images) != i].reshape(-1, 3)        
        depths = depthmaps[np.arange(num_images) != i].reshape(-1)
        depth_map_i = depthmaps[i].reshape(h, w)

        # normalize depth for comparison
        depths = normalize_depth(depths)
        depth_map_i = normalize_depth(depth_map_i)
        
        for j in range(num_images):
            if i == j:
                continue
            
            # Project 3D points to image j
            points_2d, _ = project_points(points_3d, camera_intrinsics[j], extrinsics_w2c[i])
            
            # Check if points are within image bounds
            valid_points = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                           (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
            
            # Check depth consistency using vectorized operations
            valid_points_2d = points_2d[valid_points].astype(int)
            valid_depths = depths[valid_points]

            # Extract x and y coordinates
            x_coords, y_coords = valid_points_2d[:, 0], valid_points_2d[:, 1]

            # Compute depth differences
            depth_differences = np.abs(valid_depths - depth_map_i[y_coords, x_coords])

            # Create a mask for points where the depth difference is below the threshold
            consistent_depth_mask = depth_differences < depth_threshold

            # Update the overlapping masks using the consistent depth mask
            overlapping_masks[i][y_coords[consistent_depth_mask], x_coords[consistent_depth_mask]] = True
    return overlapping_masks


# def compute_overlapping_mask(depthmaps, pointmaps, camera_intrinsics, extrinsics_w2c, image_sizes, depth_threshold=0.1):
#     num_images, h, w, _ = image_sizes
#     pointmaps = pointmaps.reshape(num_images, h, w, 3)
#     overlapping_masks = np.zeros((num_images, h, w), dtype=bool)
    
#     for i in range(num_images):
#         # points_3d = pointmaps[i].reshape(-1, 3)
#         # Exclude the current pointmap
#         points_3d = pointmaps[np.arange(num_images) != i].reshape(-1, 3)
        
#         for j in range(num_images):
#             if i == j:
#                 continue
            
#             # Project 3D points to image j
#             points_2d, depths = project_points(points_3d, camera_intrinsics[j], extrinsics_w2c[i])
            
#             # Check if points are within image bounds
#             valid_points = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
#                            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
            
#             # Check depth consistency
#             # if depth so wrong, it maybe occ_area / background area

#             # depths = depthmaps[j]
#             depths = depthmaps[np.arange(num_images) != i].reshape(-1)
#             depth_map_j = depthmaps[i].reshape(h, w)

#             # normalize depth for comparison
#             depths = (depths - np.min(depths)) / (np.max(depths) - np.min(depths))
#             depth_map_j = (depth_map_j - np.min(depth_map_j)) / (np.max(depth_map_j) - np.min(depth_map_j))

#             # depth_map_j = pointmaps[j][:, :, 2]
#             for point, depth in zip(points_2d[valid_points], depths[valid_points]):
#                 x, y = int(point[0]), int(point[1])
#                 if abs(depth - depth_map_j[y, x]) < depth_threshold:
#                 # if abs(depth - depth_map_j[y, x]) < depth_threshold:
#                     overlapping_masks[i][y, x] = True
#                     # overlapping_masks[j][y, x] = True
    
#     return overlapping_masks

def project_points(points_3d, intrinsics, extrinsics):
    # Convert to homogeneous coordinates
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    
    # Apply extrinsic matrix
    points_camera = np.dot(extrinsics, points_3d_homogeneous.T).T
    
    # Apply intrinsic matrix
    points_2d_homogeneous = np.dot(intrinsics, points_camera[:, :3].T).T
    
    # Convert to 2D coordinates
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]
    depths = points_camera[:, 2]
    
    return points_2d, depths

def read_colmap_gt_pose(gt_pose_path, llffhold=8):
    # colmap_cam_extrinsics = read_extrinsics_binary(gt_pose_path + '/triangulated/images.bin')
    colmap_cam_extrinsics = read_extrinsics_binary(gt_pose_path + '/sparse/0/images.bin')
    all_pose=[]
    print("Loading colmap gt train pose:")
    for idx, key in enumerate(colmap_cam_extrinsics):
        # if idx % llffhold == 0:
        extr = colmap_cam_extrinsics[key]
        # print(idx, extr.name)
        # R = np.transpose(qvec2rotmat(extr.qvec))
        R = np.array(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        pose = np.eye(4,4)
        pose[:3, :3] = R
        pose[:3, 3] = T
        all_pose.append(pose)
    colmap_pose = np.array(all_pose)
    return colmap_pose

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = PIL.Image.open(renders_dir / fname)
        gt = PIL.Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def align_pose(pose1, pose2):
    mtx1 = np.array(pose1, dtype=np.double, copy=True)
    mtx2 = np.array(pose2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = scipy.linalg.orthogonal_procrustes(mtx1, mtx2)
    mtx2 = mtx2 * s
    print("scale", s)

    return mtx1, mtx2, R


# colmap_train_poses = []
# for view in scene.getTrainCameras():
#     pose = view.world_view_transform.transpose(0, 1)
#     colmap_train_poses.append(pose)

# colmap_test_poses = []
# for view in scene.getTestCameras():
#     pose = view.world_view_transform.transpose(0, 1)
#     colmap_test_poses.append(pose)
# train_poses = np.load(os.path.join(dataset.model_path,"pose/pose_{}.npy".format(iteration)))
# train_poses = torch.tensor(train_poses)
# test_poses_learned, scale_a2b = align_scale_c2b_use_a2b(colmap_train_poses, train_poses, colmap_test_poses)