# visuzlize edges

import cv2
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import csv
from collections import defaultdict

p = Path(r'\\caffe-srv\projects\David\kps\CURRENT_RUN3')
def extract_2d_bbox(pts,w,h):
    minx= 1e10
    miny = 1e10
    maxx = 0
    maxy= 0
    for pt in pts:
        minx = min(pt[0],minx)
        miny = min(pt[1], miny)
        maxx = max(pt[0], maxx)
        maxy = max(pt[1], maxy)

    new_pts = [(minx,miny),(maxx,maxy),(maxx,miny),(minx,maxy) ]
    new_pts = [(min(x,w), min(y,h) ) for (x,y) in new_pts ]
    new_pts = [(max(x, 0), max(y, h)) for (x, y) in new_pts]
    return new_pts
def calc_2d( data,w,h):
    x = float(data['3D_x'])
    y = float(data['3D_y'])
    z = float(data['3D_z'])

    a = np.radians(float(data['cam_rot_x']))
    b = np.radians(float(data['cam_rot_y']))
    c = np.radians(float(data['cam_rot_z']))
#     a = np.radians(-4.6)
#     b = np.radians(0)
#     c = np.radians(-84)

    cam_x = float(data['cam_3D_x'])
    cam_y = float(data['cam_3D_y'])
    cam_z = float(data['cam_3D_z'])

    fov = int(data['fov'])

    # Image Size
    W = w
    H = h

    rot_mat = np.array([[np.cos(b)*np.cos(c), -np.cos(a)*np.sin(c)+np.sin(a)*np.sin(b)*np.cos(c), np.sin(a)*np.sin(c)+np.cos(a)*np.sin(b)*np.cos(c)],
                     [np.cos(b)*np.sin(c), np.cos(a)*np.cos(c)+np.sin(a)*np.sin(b)*np.sin(c), -np.sin(a)*np.cos(c)+np.cos(a)*np.sin(b)*np.sin(c)],
                     [-np.sin(b), np.sin(a)*np.cos(b), np.cos(a)*np.cos(b)]])

    obj_3d = np.array([x, y, z])
    cam_3d = np.array([cam_x, cam_y, cam_z])

    coor_vec = np.matmul((obj_3d-cam_3d),rot_mat)

    focal = (H/2)*(1/(np.tan(np.radians(fov/2))))
    # focal = (H/2.0)* math.degrees(np.tan(np.radians(25)))

    x = focal*(coor_vec[0]/coor_vec[1]) + W/2
    y = H/2 - focal*(coor_vec[2]/coor_vec[1])

    return (int(x),int(y))

def norm_dummy(pt, dummy):
    new_pt = pt.copy()
    new_pt[0] = pt[0] - dummy[0]
    new_pt[1] = pt[1] - dummy[1]
    new_pt[2] = pt[2] - dummy[2]
    return new_pt


def restore_dummy(pt, dummy):
    new_pt = pt.copy()
    new_pt[0] = pt[0] + dummy[0]
    new_pt[1] = pt[1] + dummy[1]
    new_pt[2] = pt[2] + dummy[2]
    return new_pt




dbg_path = p / 'debug_images'
dbg_path.mkdir(exist_ok=True)
rot = R.from_euler('z', -72)
rot2 = R.from_euler('z', 72)
import csv
to_rmv = set()
for car_hash in p.glob('*'):
    # print (car_hash)
    if car_hash.stem != '2261744861':
        continue
    heights=  car_hash.glob('*')
    # if car_hash.stem in list(dbg_path.glob('*')):
    #     continue
    for height in heights:
        if height.stem!='15':
            continue
        radiuses = height.glob('*')
        for radius in radiuses:
            pics = radius.glob('*')
            if ((radius / 'new_bbox.csv').exists()):

                ROWS =  list(csv.DictReader(open(radius / 'new_bbox.csv', 'r')))
                # if len(ROWS)>1:
                # # print(ROWS)
                #     continue
            ROWS = list(csv.DictReader(open(radius / 'edges.csv', 'r')))
            if len(ROWS) ==0:
                to_rmv.add(car_hash)
                # print(ROWS)
                continue
            new_bbox_f = csv.DictWriter(open(radius / 'new_bbox.csv', 'w'), fieldnames=['ms', 'idx', 'y', 'x'],lineterminator='\n')
            new_bbox_f.writeheader()
            two_d_bbox = csv.DictWriter(open(radius / '2d_bbox.csv', 'w'), fieldnames=['ms', 'y', 'x'],lineterminator='\n')
            two_d_bbox.writeheader()
            for sec in pics:
                if not 'bmp' in sec.name and not 'jpg' in sec.name:
                    continue
                ms = sec.stem  # '1581513309477'
                if (radius / f'{ms}.bmp').exists():
                    im = cv2.imread(str(radius / f'{ms}.bmp'))
                else:
                    im = cv2.imread(str(radius / f'{ms}.jpg'))

                edges_rows = list(csv.DictReader(open(radius / 'edges.csv')))
                cam_rows = list(csv.DictReader(open(radius / 'cam_coords.csv')))
                rows = list(csv.DictReader(open(radius / 'coords.csv')))

                by_ms = defaultdict(list)
                for row in rows:
                    by_ms[row['ms']].append(row)
                # by_ms = defaultdict(list)
                # for row in rows :
                #     by_ms[row['ms']].append(row)
                # im =  cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                cam_rows_by_ms = {}
                for row in cam_rows:


                    cam_rows_by_ms[row['ms']]=row



                # print(by_ms[f'{ms}'])
                # print(by_ms)
                pts = []
                edges_rows_by_ms = defaultdict(list)
                for row in edges_rows:
                    edges_rows_by_ms[row['ms']].append(row)

                for part in by_ms[f'{ms}']:
                    data = {}
                    data['fov'] = 50
                    #         print(part['part'])
                    data['3D_x'], data['3D_y'], data['3D_z'] = float(part['x']), float(part['y']), float(
                        part['z'])  # 268.568,1122.32,219.804#265.292 , 1122.89 , 219.81
                    data['cam_rot_z'] = cam_rows_by_ms[f'{ms}']['rotz']
                    data['cam_rot_y'] = cam_rows_by_ms[f'{ms}']['roty']
                    data['cam_rot_x'] = cam_rows_by_ms[f'{ms}']['rotx']
                    data['cam_3D_x'] = cam_rows_by_ms[f'{ms}']['x']
                    data['cam_3D_y'] = cam_rows_by_ms[f'{ms}']['y']
                    data['cam_3D_z'] = cam_rows_by_ms[f'{ms}']['z']

                    h, w, _ = im.shape
                    x, y = calc_2d(data, w, h)
                    center = (x, y)
                    if part['part'] == 'chassis_dummy':
                        dummy = (float(part['x']), float(part['y']), float(part['z']))
                    if part['occlusion'] == '1':
                        pass
                    else:
                        pass
                maxz = -1
                minz = 1e10
                for idx, part in enumerate(edges_rows_by_ms[f'{ms}']):
                    data = {}
                    maxz = max(maxz, float(part['z']))
                    minz = min(minz, float(part['z']))
                    data['fov'] = 50

                    data['3D_x'], data['3D_y'], data['3D_z'] = float(part['x']), float(part['y']), float(
                        part['z'])  # 268.568,1122.32,219.804#265.292 , 1122.89 , 219.81
                    data['cam_rot_z'] = cam_rows_by_ms[f'{ms}']['rotz']
                    data['cam_rot_y'] = cam_rows_by_ms[f'{ms}']['roty']
                    data['cam_rot_x'] = cam_rows_by_ms[f'{ms}']['rotx']
                    data['cam_3D_x'] = cam_rows_by_ms[f'{ms}']['x']
                    data['cam_3D_y'] = cam_rows_by_ms[f'{ms}']['y']
                    data['cam_3D_z'] = cam_rows_by_ms[f'{ms}']['z']

                    h, w, _ = im.shape

                    x, y = calc_2d(data, w, h)
                    center = (x, y)
                    #         print(center)
                    pts.append(center)

                    # cv2.circle(im, (x,y), 5, (0, 0, 255), -1)
                    new_bbox_f.writerow({'ms':ms,'idx':idx, 'x':x,'y':y})
                    if x > w or x<0 or y<0 or y>h:
                        continue
                    cv2.putText(im, f"{idx}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
                # print(pts,len(pts)/2)
                bbox = extract_2d_bbox(pts, w, h)
                for pt in bbox:
                    two_d_bbox.writerow({'ms': ms, 'x': pt[0], 'y': pt[1]})
                    cv2.circle(im, pt, 10, (150, 100, 128), -1)

                pts1 = pts[:len(pts) // 2]
                pts_arr = []
                for i, part in enumerate(by_ms[f'{ms}']):
                    x, y, z = float(part['x']), float(part['y']), float(part['z'])
                    pt = np.array([x, y, z])
                    new_pt = norm_dummy(pt, dummy)
                    new_pt = rot.apply(new_pt)
                    # new_pt = restor   e_dummy(new_pt,dummy)
                    # print('before', pt)
                    # print('after', new_pt)
                    pts_arr.append(new_pt)

                pts_arr = np.array(pts_arr)
                # print(pts_arr)
                maxes = np.argmax(pts_arr, axis=0)
                mins = np.argmin(pts_arr, axis=0)
                maxesa = np.amax(pts_arr, axis=0)
                minsa = np.amin(pts_arr, axis=0)
                #     print(maxes,mins)
                adj_maxz = maxz - dummy[2]
                adj_minz = minz - dummy[2]
                # maxesa[-1] = adj_maxz
                # minsa[-1] = adj_minz
                e1 = [maxesa[0], minsa[1], adj_maxz]
                e2 = [minsa[0], maxesa[1], adj_minz]
                e3 = [maxesa[0], minsa[1], adj_minz]
                e4 = [minsa[0], maxesa[1], adj_maxz]

                e5 = [maxesa[0], maxesa[1], adj_minz]
                e6 = [minsa[0], minsa[1], adj_maxz]
                #     print(pts_arr[maxes[0]]),print(pts_arr[mins[0]])
                tight_pts = [pts_arr[maxes[0]], pts_arr[maxes[1]], pts_arr[mins[0]], pts_arr[mins[1]]]

                e0 = minsa
                e1= [minsa[0], minsa[1], maxesa[2]]
                e2 = [minsa[0], maxesa[1], minsa[2]]
                e3 = [minsa[0], maxesa[1], maxesa[2]]
                e4 = [maxesa[0], minsa[1], minsa[2]]
                e5 = [maxesa[0], minsa[1], maxesa[2]]
                e6 = [maxesa[0], maxesa[1], minsa[2]]
                e7 = maxesa

                # new_bbox = [minsa,maxesa,e1,e2,e3,e4,e5,e6]
                new_bbox = [minsa, e2, e4, e6, maxesa, e1, e3, e5]
                new_bbox = [e0,e1 ,e2,e3, e4,e5, e6,e7]
                #     print(tight_pts)
                cnt = 0
                tight_pts2 = []


                for pt in new_bbox:
                    data = {}
                    data['fov'] = 50
                    pt =rot2.apply(pt)
                    data['3D_x'], data['3D_y'], data['3D_z'] =tuple(restore_dummy(pt, dummy))
                    # print('data', data)
                    data['cam_rot_z'] = cam_rows_by_ms[f'{ms}']['rotz']
                    data['cam_rot_y'] = cam_rows_by_ms[f'{ms}']['roty']
                    data['cam_rot_x'] = cam_rows_by_ms[f'{ms}']['rotx']
                    data['cam_3D_x'] = cam_rows_by_ms[f'{ms}']['x']
                    data['cam_3D_y'] = cam_rows_by_ms[f'{ms}']['y']
                    data['cam_3D_z'] = cam_rows_by_ms[f'{ms}']['z']

                    h, w, _ = im.shape
                    x, y = calc_2d(data, w, h)
                    center = (x, y)
                    # print('center', center)
                    # cv2.circle(im, center, 20, (50, 150, 128), -1)
                    tight_pts2.append(center)
                    cv2.putText(im, str(cnt), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cnt += 1

                for cs1, cs2 in list(zip(pts1, pts1[1:] + pts1[:1])):
                    cv2.line(im, cs1, cs2, (0, 255, 0), 1)
                connects = (2, 4), (1, 7), (3, 5), (0, 6), (7, 4), (5, 4), (5, 6), (6, 7), (3, 0), (1, 2), (2, 3), (0, 1)
                for x in connects:
                    cs1, cs2 = pts[x[0]], pts[x[1]]
                    cs3, cs4 = tight_pts2[x[0]], tight_pts2[x[1]]
                    cv2.line(im, cs1, cs2, (0, 255, 0), 1)
                    # cv2.line(im, cs3, cs4, (255, 0, 0), 1)
                for part in by_ms[f'{ms}']:
                    data = {}
                    data['fov'] = 50
                    #         print(part['part'])
                    data['3D_x'], data['3D_y'], data['3D_z'] = float(part['x']), float(part['y']), float(
                        part['z'])  # 268.568,1122.32,219.804#265.292 , 1122.89 , 219.81
                    data['cam_rot_z'] = cam_rows_by_ms[f'{ms}']['rotz']
                    data['cam_rot_y'] = cam_rows_by_ms[f'{ms}']['roty']
                    data['cam_rot_x'] = cam_rows_by_ms[f'{ms}']['rotx']
                    data['cam_3D_x'] = cam_rows_by_ms[f'{ms}']['x']
                    data['cam_3D_y'] = cam_rows_by_ms[f'{ms}']['y']
                    data['cam_3D_z'] = cam_rows_by_ms[f'{ms}']['z']

                    h, w, _ = im.shape
                    x, y = calc_2d(data, w, h)
                    center = (x, y)

                    if part['part'] == 'chassis_dummy':
                        dummy = (float(part['x']), float(part['y']), float(part['z']))
                    if part['occlusion'] == '1':
                        cv2.circle(im, (x, y), 1, (0, 0, 255), -1)
                        #         cv2.putText(im,part['part'], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
                    else:
                        cv2.circle(im, (x, y), 1, (255, 0, 255), -1)
                dbg_path = p /'debug_images' / car_hash.stem / height.stem / radius.stem
                dbg_path.mkdir(exist_ok=True,parents=True)
                cv2.imwrite(str(dbg_path /f'{ms}.png'), im)
                # plt.imshow(im)
                # plt.show
# print(len(to_rmv))
# import shutil
# for t in to_rmv:
#     shutil.rmtree(str(t))