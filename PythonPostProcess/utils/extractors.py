from .calculations import *
import cv2

import numpy as np

from scipy.spatial.transform import Rotation as R
def get_part_kp_arr(ms, by_ms_by_car_hash,vehicle_hash,dummy):
    pts_arr = []
    rot2 = R.from_euler('xyz', [0,0,0])

    for i, part in enumerate(by_ms_by_car_hash[ms][vehicle_hash]):
        x, y, z = float(part['x']), float(part['y']), float(part['z'])
        yaw, roll , pitch = 0,0,72
        if 'yaw' in part:

            yaw, roll, pitch = float(part['yaw']), float(part['roll']), float(part['pitch'])
        pt = np.array([x, y, z])
        new_pt = norm_dummy(pt, dummy)
        rot = R.from_euler('xyz', [yaw, roll, pitch])
        rot2 = R.from_euler('xyz', [-yaw, -roll, -pitch])

        new_pt = rot.apply(new_pt)
        # new_pt = restor   e_dummy(new_pt,dummy)
        # print('before', pt)
        # print('after', new_pt)
        pts_arr.append(new_pt)
    pts_arr = np.array(pts_arr)
    return pts_arr,rot2
def get_edges_pts(edges_rows_by_ms_by_car_hash, cam_rows_by_ms, vehicle_hash, ms, im, new_bbox_f):
    # gets all the pts that form the 3d bbox in the original model
    # not tight enough
    pts = []
    maxz = -1
    minz = 1e10
    for idx, part in enumerate(edges_rows_by_ms_by_car_hash[ms][vehicle_hash]):

        data = {}
        maxz = max(maxz, float(part['z']))
        minz = min(minz, float(part['z']))
        data['fov'] = 50
        if 'nan' in ''.join([part['x'],part['y'],part['z']]):
            continue
        data['3D_x'], data['3D_y'], data['3D_z'] = float(part['x']), float(part['y']), float(
            part['z'])  # 268.568,1122.32,219.804#265.292 , 1122.89 , 219.81
        data['cam_rot_z'] = cam_rows_by_ms[ms]['rotz']
        data['cam_rot_y'] = cam_rows_by_ms[ms]['roty']
        data['cam_rot_x'] = cam_rows_by_ms[ms]['rotx']
        data['cam_3D_x'] = cam_rows_by_ms[ms]['x']
        data['cam_3D_y'] = cam_rows_by_ms[ms]['y']
        data['cam_3D_z'] = cam_rows_by_ms[ms]['z']

        h, w, _ = im.shape

        x, y = calc_2d(data, w, h)
        if x is None:
            continue
        center = (x, y)
        #         print(center)
        pts.append(center)

        new_bbox_f.writerow({'ms': ms, 'idx': idx, 'x': x, 'y': y})
        if x > w or x < 0 or y < 0 or y > h:
            continue
        cv2.putText(im, f"{idx}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
    return pts, im,maxz,minz
def get_dummy(by_ms,vehicle_hash,cam_rows_by_ms,ms,w,h):
    dummy = None
    for part in by_ms[ms]:
        data = {}
        data['fov'] = 50

        vehicle_hash_index = f"{part['vehicle_hash']}_{part['vehicle_index']}"
        if vehicle_hash != vehicle_hash_index:
            continue
        #         print
        if part['z'] is None:
            continue
        data['3D_x'], data['3D_y'], data['3D_z'] = float(part['x']), float(part['y']), float(
            )  # 268.568,1122.32,219.804#265.292 , 1122.89 , 219.81
        data['cam_rot_z'] = cam_rows_by_ms[ms]['rotz']
        data['cam_rot_y'] = cam_rows_by_ms[ms]['roty']
        data['cam_rot_x'] = cam_rows_by_ms[ms]['rotx']
        data['cam_3D_x'] = cam_rows_by_ms[ms]['x']
        data['cam_3D_y'] = cam_rows_by_ms[ms]['y']
        data['cam_3D_z'] = cam_rows_by_ms[ms]['z']

        x, y = calc_2d(data, w, h)
        center = (x, y)
        if part['part'] == 'chassis_dummy':
            dummy = (float(part['x']), float(part['y']), float(part['z']))
        if part['occlusion'] == '1':
            pass
        else:
            pass
    return dummy
