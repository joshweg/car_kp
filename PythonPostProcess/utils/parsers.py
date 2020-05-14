import csv
from collections import defaultdict


def parse_csvs(radius):
    edges_rows = list(csv.DictReader(open(radius / 'edges.csv')))
    cam_rows = list(csv.DictReader(open(radius / 'cam_coords.csv')))
    rows = list(csv.DictReader(open(radius / 'coords.csv')))

    cam_coords_filtered = csv.DictWriter(open(radius / 'cam_coords_filtered.csv', 'w'), fieldnames=cam_rows[0].keys(),
                                         lineterminator='\n')
    cam_coords_filtered.writeheader()
    by_ms = defaultdict(list)
    for row in rows:
        by_ms[int(row['ms'])].append(row)

    by_ms_by_car_hash = defaultdict(lambda: defaultdict(list))
    for row in rows:
        vehicle_hash_index = f"{row['vehicle_hash']}_{row['vehicle_index']}"
        by_ms_by_car_hash[int(row['ms'])][vehicle_hash_index].append(row)

    cam_rows_by_ms = {}

    for row in cam_rows:
        # get rid of faulty camera positions, there are plenty
        # if any([float(row[k]) == 0 for k in row if k != 'ms']):
        #     continue
        for k in row.keys():
            if k == 'ms':
                continue
            if float(row[k]) == 0:
                row[k] = '0.000000426887'
        cam_coords_filtered.writerow(row)
        cam_rows_by_ms[int(float(row['ms']))] = row
    edges_rows_by_ms_by_car_hash = defaultdict(lambda: defaultdict(list))
    for row in edges_rows:
        # edges_rows_by_ms[row['ms']].append(row)
        vehicle_hash_index = f"{row['vehicle_hash']}_{row['vehicle_index']}"
        edges_rows_by_ms_by_car_hash[int(row['ms'])][vehicle_hash_index].append(row)
    return by_ms,by_ms_by_car_hash,cam_rows_by_ms,edges_rows_by_ms_by_car_hash