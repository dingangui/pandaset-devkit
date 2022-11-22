from matplotlib import pyplot as plt

import pandaset
import os
from pandaset import geometry
import matplotlib.cm as cm
import numpy as np
import mmcv
import cv2
import json
import transforms3d as t3d

def _heading_position_to_mat(heading, position):
    quat = np.array([heading["w"], heading["x"], heading["y"], heading["z"]])
    pos = np.array([position["x"], position["y"], position["z"]])
    # t3d.affines.compose(T, R, Z): 组合 平移(T), 旋转(R), 缩放(Z)矩阵
    transform_matrix = t3d.affines.compose(np.array(pos),
                                           t3d.quaternions.quat2mat(quat),
                                           [1.0, 1.0, 1.0])
    return transform_matrix
    
def projection(lidar_points, camera_data, camera_pose, camera_intrinsics, filter_outliers=True):
    camera_heading = camera_pose['heading']
    camera_position = camera_pose['position']
    camera_pose_mat = _heading_position_to_mat(camera_heading, camera_position)

    trans_lidar_to_camera = np.linalg.inv(camera_pose_mat)
    points3d_lidar = lidar_points
    # 不通过齐次坐标实现旋转 + 平移
    points3d_camera = trans_lidar_to_camera[:3, :3] @ (points3d_lidar.T) + \
                        trans_lidar_to_camera[:3, 3].reshape(3, 1)
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = camera_intrinsics.fx
    K[1, 1] = camera_intrinsics.fy
    K[0, 2] = camera_intrinsics.cx
    K[1, 2] = camera_intrinsics.cy

    inliner_indices_arr = np.arange(points3d_camera.shape[1]) # 表示 points3d_camera 所有点云的 index
    if filter_outliers:
        condition = points3d_camera[2, :] > 0.0 # 某个位置 depth 是否大于 0 (condition 是 true or false)
        points3d_camera = points3d_camera[:, condition] # 过滤掉 depth 小于 0 的数据
        inliner_indices_arr = inliner_indices_arr[condition] # 得到 depth 大于 0 的点云 index

    points2d_camera = K @ points3d_camera #  相机内参 @ 相机坐标 -> 像素坐标(按深度缩放后)
    points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T # 像素坐标除以深度, 得到实际像素坐标

    if filter_outliers:
        image_w, image_h = camera_data.size
        # 某个位置像素坐标是否超过图像边界 (condition 是 true or false)
        condition = np.logical_and(
            (points2d_camera[:, 1] < image_h) & (points2d_camera[:, 1] > 0),
            (points2d_camera[:, 0] < image_w) & (points2d_camera[:, 0] > 0)) 
        points2d_camera = points2d_camera[condition] # 过滤超出图像边界的像素
        points3d_camera = (points3d_camera.T)[condition] # 过滤投影到像素平面后超出图像边界的 3d 点云
        inliner_indices_arr = inliner_indices_arr[condition] # 过滤投影到像素平面后超出图像边界的点云 index
    return points2d_camera, points3d_camera, inliner_indices_arr

def create_output(vertices, colors, filename):
    """
    输出 ply 文件, 用于 meshlab 可视化
    os.makedirs(f'visualized_results/Lidar_Coordinate_Points_0', exist_ok=True)
    create_output(points_0_concat, np.ones_like(points_0_concat) * 255, f'visualized_results/Lidar_Coordinate_Points_0/{filename}.ply')
    """
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')     # 必须先写入，然后利用write()在头部插入ply header
    ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
            \n
            '''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)

def gen_depth():
    """
    生成 pandaset 数据集的 depth map
    """
    # load dataset
    dataset = pandaset.DataSet("/hdd/1000w/pandaset")

    # filefold_seqs: 数据集目录下拥有的文件夹列表
    for root,dir,_ in os.walk("/hdd/1000w/pandaset"): 
        filefold_seqs = dir
        break

    camera_names = ['back_camera', 'front_camera', 'right_camera', 'left_camera', 'front_right_camera', 'front_left_camera']

    finished_jobs = json.load(open('log/finished_jobs.json', 'r'))

    # 遍历所有文件夹
    for filefold in filefold_seqs:

        # if filefold in finished_jobs:
        #     continue
        seq = dataset[filefold] # 某个文件夹下的全部帧
        seq.load() # 从磁盘 load 数据到内存
        lidar = seq.lidar # 某个文件夹全部帧下的全部激光雷达数据
        
        # 遍历所有摄像头
        for camera_name in camera_names:    
            choosen_camera = seq.camera[camera_name] # 含 intrinsics poses timestamps
            # 遍历所有帧
            for seq_idx in range(len(lidar.data)):
                points3d_lidar_xyz = lidar.data[seq_idx].to_numpy()[:, :3] # 当前帧世界坐标系下的激光雷达点云坐标(含全部视角)
                os.makedirs(f'visualized_results/points3d_lidar_xyz', exist_ok=True)
                create_output(points3d_lidar_xyz, np.ones_like(points3d_lidar_xyz) * 255, f'visualized_results/points3d_lidar_xyz/{filefold}_{str(seq_idx).zfill(2)}.ply')
                """
                #### 2.Use projection function in pandaset.geometry to get projection 2d-points on image.
                - ***geometry.projection***
                    - input
                        - ***lidar_points***(np.array(\[N, 3\])): lidar points in the world coordinates.
                        - ***camera_data***(PIL.Image): image for one camera in one frame.
                        - ***camera_pose***: pose in the world coordinates for one camera in one frame.
                        - ***camera_intrinsics***: intrinsics for one camera in one frame.
                        - ***filter_outliers***(bool): filtering projected 2d-points out of image.
                    - output
                        - ***projection_points2d***(np.array(\[K, 2\])): projected 2d-points in pixels.
                        - ***camera_points_3d***(np.array(\[K, 3\])): 3d-points in pixels in the camera frame.
                        - ***inliner_idx***(np.array(\[K, 2\])): the indices for *lidar_points* whose projected 2d-points are inside image.
                """
                projected_points2d, camera_points_3d, inner_indices = projection(lidar_points=points3d_lidar_xyz, 
                                                                                        camera_data=choosen_camera[seq_idx],
                                                                                        camera_pose=choosen_camera.poses[seq_idx],
                                                                                        camera_intrinsics=choosen_camera.intrinsics,
                                                                                        filter_outliers=True)
                # print("projection 2d-points inside image count:", projected_points2d.shape)
                
                """
                projected_points2d: 点云对齐相机坐标后的 x,y 坐标
                    x: projected_points2d[:, 0]
                    y: projected_points2d[:, 1]

                camera_points_3d: shape [points_num, 3]

                distances: 欧几里得距离
                z_depth: 
                """
                ori_image = seq.camera[camera_name][seq_idx]
                # distances = np.sqrt(np.sum(np.square(camera_points_3d), axis=-1))
                z_depth = camera_points_3d[:,2]


                # 全 0 背景 + 点云深度 + scale 200 倍 + 300 米以上做截断
                h, w = ori_image.height, ori_image.width
                depth = np.zeros((h, w))
                depth[projected_points2d[:, 1].astype(np.int32), projected_points2d[:, 0].astype(np.int32)] = z_depth
                depth *= 200.
                depth[np.where(depth > 200. * 300.)] = 0

                depth_map_path = f'/hdd/1000w/pandaset/{filefold}/depth/{camera_name}/{str(seq_idx).zfill(2)}.png'
                print(depth_map_path)
                # mmcv.imwrite(depth.astype(np.uint16), depth_map_path)
                
                rows = projected_points2d[:, 1]
                cols = projected_points2d[:, 0]
                colors = cm.jet(z_depth / np.max(z_depth))
                fig, ax = plt.subplots(1, 1, figsize=(20, 20))
                fig.tight_layout()
                ax.axis('off')
                ax.imshow(ori_image)
                ax.scatter(cols, rows, c=colors, s=3)
                os.makedirs(f'visualized_results/scat_points/', exist_ok=True)
                plt.savefig(f'visualized_results/scat_points/{filefold}_{camera_name}_{str(seq_idx).zfill(2)}.png', bbox_inches='tight',pad_inches = 0)
                
                os.makedirs(f'visualized_results/camera_points_3d', exist_ok=True)
                create_output(camera_points_3d, np.ones_like(camera_points_3d) * 255, f'visualized_results/camera_points_3d/{filefold}_{camera_name}_{str(seq_idx).zfill(2)}.ply')
                
                break
        break

        print(f"{filefold} finished")
        finished_jobs.append(f'{filefold}')

        with open('log/finished_jobs.json', 'w') as outfile:
            json.dump(finished_jobs, outfile, indent = 4)
    
        
    # """
    # 判断生成的 depth map 是否准确
    # """

    # depthmap_0 = mmcv.imread("depthmap_0.png", -1)

    # ori_image = mmcv.imread("img_0.png",flag='color', channel_order='rgb', backend='cv2')

    # plt.figure(figsize=(20, 12))
    # plt.imshow(ori_image)
    # colors = cm.jet(depthmap_0[np.where(depthmap_0 > 0)] / np.max(depthmap_0))

    # plt.gca().scatter(np.where(depthmap_0 > 0)[1], np.where(depthmap_0 > 0)[0], color=colors, s=1)
    # plt.savefig(f'plt_0_new.png')

if __name__ == '__main__':
    gen_depth()