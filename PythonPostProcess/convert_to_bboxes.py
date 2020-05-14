from matplotlib import pyplot as plt
import os,csv
from collections import defaultdict


def euclidean(v1, v2):
    return sum((p - q) ** 2 for p, q in zip(v1, v2)) ** .5


# root = r'D:\SteamLibrary\steamapps\common\Grand Theft Auto V\JTA\Similarite\CLEAR_h8_0_09_09_2019_08_44_32'
# root= r'\\BRIEFCAM-GTA\Grand Theft Auto V\JTA\Similarite\CLEAR_h8_0_10_09_2019_10_48_17'
root = r'D:\SteamLibrary\steamapps\common\Grand Theft Auto V\JTA\Crowd_Counting'
for seq in os.listdir(root):
    print(seq)
    # if 'seq' not in seq:
    #     continue
    rows = list(csv.DictReader(open(os.path.join(root, seq, 'coords.csv'), 'r')))
    headers= ['ped_hash','frame','bbox_head_max_x','bbox_head_max_y','bbox_head_min_x','bbox_head_min_y' ,
              'bbox_body_max_x', 'bbox_body_max_y', 'bbox_body_min_x', 'bbox_body_min_y']
    wr = csv.DictWriter(open(os.path.join(root, seq, 'bbox_coords.csv'), 'w') , fieldnames=headers,lineterminator = '\n')
    wr.writeheader()
    relevant = []
    kp_dict = defaultdict(lambda : defaultdict(dict))




    radius = 3
    for row in rows:

        frame =int(row['frame'])
        x = int(float(row['2D_x']))
        y = int(float(row['2D_y']))

        kp_dict[frame][row['pedestrian_id']][int(row['joint_type'])] = (x, y)


    for frame in kp_dict.keys():

        for ped_hash in kp_dict[frame]:
            # determine bbox edges for body
            max_x = 0
            min_x = 1e100
            max_y = 0
            min_y = 1e100
            detection ={'ped_hash': ped_hash,'frame':frame}
            for kp in kp_dict[frame][ped_hash]:
                x, y = kp_dict[frame][ped_hash][kp]
                # print('xy', x, y)
                # cv2.circle(im, (x, y), 1, (255, 0, 0), thickness=1, lineType=8, shift=0)
                max_x = max(x, max_x)
                max_y = max(y, max_y)
                min_x = int(min(x, min_x))
                min_y = int(min(y, min_y))
                dist_x = max_x - min_x
                dist_y = max_y - min_y

                # print(dist_x, dist_y, offset_x, offset_y)
            if 1 in kp_dict[frame][ped_hash] and 3 in kp_dict[frame][ped_hash]:
                radius = 1.5 * euclidean(kp_dict[frame][ped_hash][1], kp_dict[frame][ped_hash][3])

            detection['bbox_body_min_x'], detection['bbox_body_min_y'] = min_x - int(radius / 2), min_y - int(radius / 2)
            detection['bbox_body_max_x'], detection['bbox_body_max_y'] = max_x+ int(radius / 2),max_y + int(radius / 2)
            kp_chosen = 1
            if kp_chosen not in kp_dict[frame][ped_hash]:
                kp_chosen = 2
            # determine bbox edges for dickhead
            if kp_chosen not in kp_dict[frame][ped_hash]:
                kp_chosen = 3
            if kp_chosen not in kp_dict[frame][ped_hash]:
                wr.writerow(detection)
                continue

            new_x = int(kp_dict[frame][ped_hash][kp_chosen][0])
            new_y = int(kp_dict[frame][ped_hash][kp_chosen][1])
            detection['bbox_head_max_x'], detection['bbox_head_max_y'] = max_x+ int(radius / 2), max_y+ int(radius / 2)
            detection['bbox_head_min_x'], detection['bbox_head_min_y'] = min_x- int(radius / 2), min_y- int(radius / 2)
            wr.writerow(detection)


                # cv2.rectangle(im, (new_x - int(radius / 2), new_y - int(radius / 2)),
                #               (new_x + int(radius / 2), new_y + int(radius / 2)), (0, 0, 255))
                # cv2.rectangle(im, (min_x - int(radius / 2), min_y - int(radius / 2)),
                #               (max_x + int(radius / 2), max_y + int(radius / 2)), (0, 255, 0))
                # plt.imshow(im)
                # # plt.imsave(r'D:\fuck.jpg',im)
                # plt.show()
                # cv2.imwrite(r'D:\tmp_dbg\{}.jpg'.format(seq), im)
