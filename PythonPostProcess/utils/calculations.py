import cv2
import numpy as np
def calc_new_bbox(pts_arr,dummy,maxz,minz):
    #calculates the new bbox based on the car kps and the height of the origin al bbbox
    # it sucks but it works. kind of
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
    e1 = [minsa[0], minsa[1], maxesa[2]]
    e2 = [minsa[0], maxesa[1], minsa[2]]
    e3 = [minsa[0], maxesa[1], maxesa[2]]
    e4 = [maxesa[0], minsa[1], minsa[2]]
    e5 = [maxesa[0], minsa[1], maxesa[2]]
    e6 = [maxesa[0], maxesa[1], minsa[2]]
    e7 = maxesa

    # new_bbox = [minsa,maxesa,e1,e2,e3,e4,e5,e6]
    new_bbox = [minsa, e2, e4, e6, maxesa, e1, e3, e5]
    new_bbox = [e0, e1, e2, e3, e4, e5, e6, e7]
    return new_bbox
def get_pts(edges_rows_by_ms_by_car_hash,cam_rows_by_ms,vehicle_hash,ms,im,new_bbox_f ):
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
def get_dummy(by_ms,cam_rows_by_ms,ms):
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
            pass
        else:
            pass
    return dummy
def get_tight_bbox(im,new_bbox,cam_rows_by_ms,ms,rot2,dummy):
    cnt = 0
    tight_pts2 = []
    for pt in new_bbox:
        data = {}
        data['fov'] = 50
        pt = rot2.apply(pt)
        data['3D_x'], data['3D_y'], data['3D_z'] = tuple(restore_dummy(pt, dummy))
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
        cv2.circle(im, center, 1, (0, 255, 255), 4)
        tight_pts2.append(center)

        cnt += 1
    return tight_pts2

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
    if np.isnan(x) or np.isnan(y):
        return None,None
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


