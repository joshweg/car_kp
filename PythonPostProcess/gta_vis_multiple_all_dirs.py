# visuzlize edges
import shutil

import cv2
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import csv
from collections import defaultdict

from utils.parsers import parse_csvs

p = Path(r'\\caffe-srv\projects\David\kps\MULTIPLE\SECOND')
from utils.calculations import *
from utils.extractors import *

dbg_path = p / 'debug_images'
dbg_path.mkdir(exist_ok=True)

import csv
to_rmv = set()



for run in p.glob('*'):

    if not 'CARKPMULTIPLE_DEBUG_' in str(run):
        continue
    try:
        print(f' working on {run}')
        dbg = run /'debug_images'
        # if dbg.is_dir():
        #     shutil.rmtree(dbg)

        for car_hash in run.glob('*'):
            if not car_hash.is_dir():
                continue
            if car_hash.stem == 'debug_images':
                continue
            heights=  car_hash.glob('*')
            # if car_hash.stem in list(dbg_path.glob('*')):
            #     continue
            for height in heights:
                radiuses = height.glob('*')
                for radius in radiuses:
                    pics = radius.glob('*')

                    new_bbox_f = csv.DictWriter(open(radius / 'new_bbox.csv', 'w'), fieldnames=['ms', 'idx', 'y', 'x'],lineterminator='\n')
                    new_bbox_f.writeheader()
                    two_d_bbox = csv.DictWriter(open(radius / '2d_bbox.csv', 'w'), fieldnames=['ms', 'y', 'x'],lineterminator='\n')
                    two_d_bbox.writeheader()
                    # read csvs
                    by_ms, by_ms_by_car_hash, cam_rows_by_ms, edges_rows_by_ms_by_car_hash = parse_csvs(radius)
                    for sec in pics:
                        # read pics
                        if not 'bmp' in sec.name and not 'jpg' in sec.name:
                            continue
                        ms = int(sec.stem)  #
                        if ms not in cam_rows_by_ms:
                            continue
                        if (radius / f'{ms}.bmp').exists():
                            im = cv2.imread(str(radius / f'{ms}.bmp'))
                        else:
                            im = cv2.imread(str(radius / f'{ms}.jpg'))

                        w, h = im.shape[:2]
                        # curr_cam_coords = cam_rows_by_ms[ms]
                        # if any([float(curr_cam_coords[k])==0 for k in curr_cam_coords if k!='ms']):
                        #     continue
                        pts = []

                        for _,vehicle_hash in enumerate(edges_rows_by_ms_by_car_hash[ms].keys()):
                            dummy = get_dummy(by_ms,vehicle_hash, cam_rows_by_ms, ms, w, h)

                            pts,im,maxz,minz = get_edges_pts(edges_rows_by_ms_by_car_hash, cam_rows_by_ms, vehicle_hash, ms, im, new_bbox_f)

                            bbox = extract_2d_bbox(pts, w, h)
                            for pt in bbox:
                                two_d_bbox.writerow({'ms': ms, 'x': pt[0], 'y': pt[1]})
                                cv2.circle(im, pt, 10, (150, 100, 128), -1)

                            pts1 = pts[:len(pts) // 2]
                            pts_arr,rot2 = get_part_kp_arr(ms, by_ms_by_car_hash,vehicle_hash,dummy)


                            # print(pts_arr.shape)
                            if  pts_arr.size ==0:
                                continue

                            new_bbox =  calc_new_bbox(pts_arr,dummy,maxz,minz)

                            #     print(tight_pts)
                            cnt = 0
                            tight_pts2 = []


                            for pt in new_bbox:
                                data = {}
                                data['fov'] = 50
                                pt =rot2.apply(pt)
                                data['3D_x'], data['3D_y'], data['3D_z'] =tuple(restore_dummy(pt, dummy))
                                # print('data', data)
                                data['cam_rot_z'] = cam_rows_by_ms[ms]['rotz']
                                data['cam_rot_y'] = cam_rows_by_ms[ms]['roty']
                                data['cam_rot_x'] = cam_rows_by_ms[ms]['rotx']
                                data['cam_3D_x'] = cam_rows_by_ms[ms]['x']
                                data['cam_3D_y'] = cam_rows_by_ms[ms]['y']
                                data['cam_3D_z'] = cam_rows_by_ms[ms]['z']

                                h, w, _ = im.shape
                                x, y = calc_2d(data, w, h)
                                center = (x, y)
                                # print('center', center)
                                # cv2.circle(im, center, 20, (50, 150, 128), -1)
                                tight_pts2.append(center)

                                cnt += 1

                            #plotting
                            for cnt, pt in enumerate(tight_pts2):

                                cv2.putText(im, str(cnt), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                            for cs1, cs2 in list(zip(pts1, pts1[1:] + pts1[:1])):
                                cv2.line(im, cs1, cs2, (0, 255, 0), 1)
                            connects = (2, 4), (1, 7), (3, 5), (0, 6), (7, 4), (5, 4), (5, 6), (6, 7), (3, 0), (1, 2), (2, 3), (0, 1)
                            for x in connects:
                                cs1, cs2 = pts[x[0]], pts[x[1]]
                                cs3, cs4 = tight_pts2[x[0]], tight_pts2[x[1]]
                                cv2.line(im, cs1, cs2, (0, 255, 0), 1)
                                # cv2.line(im, cs3, cs4, (255, 0, 0), 1)
                        for part in by_ms[ms]:
                            data = {}
                            data['fov'] = 50
                            #         print(part['part'])
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
                            center = (x, y)

                            if part['part'] == 'chassis_dummy':
                                dummy = (float(part['x']), float(part['y']), float(part['z']))
                            if part['occlusion'] == '1':
                                cv2.circle(im, (x, y), 1, (0, 0, 255), 3)
                                #         cv2.putText(im,part['part'], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
                            else:
                                cv2.circle(im, (x, y), 1, (0, 255, 0), 3)
                        dbg_path = run /'debug_images' / car_hash.stem / height.stem / radius.stem
                        dbg_path.mkdir(exist_ok=True,parents=True)
                        cv2.imwrite(str(dbg_path /f'{ms}.png'), im)
                        # plt.imshow(im)
                        # plt.show
    except Exception as e:
        print(e.__str__())
# print(len(to_rmv))
# import shutil
# for t in to_rmv:
#     shutil.rmtree(str(t))