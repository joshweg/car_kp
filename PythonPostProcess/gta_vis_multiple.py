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

p = Path(r'D:\SteamLibrary\steamapps\common\Grand Theft Auto V\JTA\CarKPData\CARKPMULTIPLE_DEBUG_h1_14_05_2020_01_02_00')
from utils.calculations import *
from utils.extractors import *

dbg_path = p / 'debug_images'
dbg_path.mkdir(exist_ok=True)
rot = R.from_euler('z', -72)
rot2 = R.from_euler('z', 72)
import csv
to_rmv = set()


for car_hash in p.glob('*'):
    print(f"working on {car_hash}")
    if not car_hash.is_dir():
        continue
    # if car_hash.stem!='2261744861':
    #     continue
    if car_hash.stem == 'debug_images':
        continue
    heights=  car_hash.glob('*')
    # if car_hash.stem in list(dbg_path.glob('*')):
    #     continue
    for height in heights:
        radiuses = height.glob('*')
        for radius in radiuses:
            pics = radius.glob('*')
            # if ((radius / 'new_bbox.csv').exists()):
            #
            #     ROWS =  list(csv.DictReader(open(radius / 'new_bbox.csv', 'r')))
            #     # if len(ROWS)>1:
            #     # # print(ROWS)
            #     #     continue
            # ROWS = list(csv.DictReader(open(radius / 'edges.csv', 'r')))
            # if len(ROWS) ==0:
            #     to_rmv.add(car_hash)
            #     # print(ROWS)
            #     continue
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
                for k in  cam_rows_by_ms[ms]:
                    if k=='ms':
                        continue
                    if float(cam_rows_by_ms[ms][k])==0:
                        cam_rows_by_ms[ms][k] ='0.000000426887'

                pts = []
                for _,vehicle_hash in enumerate(edges_rows_by_ms_by_car_hash[ms].keys()):

                    dummy = get_dummy(by_ms, vehicle_hash,cam_rows_by_ms, ms, w, h)

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
                    tight_pts2 =  get_tight_bbox(im,new_bbox,cam_rows_by_ms,ms,rot2,dummy)
                    #plotting
                    for cnt, pt in enumerate(tight_pts2):

                        cv2.putText(im, str(cnt), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255))
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
                dbg_path_loc = dbg_path/ car_hash.stem / height.stem / radius.stem
                dbg_path_loc.mkdir(exist_ok=True,parents=True)
                cv2.imwrite(str(dbg_path_loc /f'{ms}.png'), im)
                # plt.imshow(im)
                # plt.show
# print(len(to_rmv))
# import shutil
# for t in to_rmv:
#     shutil.rmtree(str(t))