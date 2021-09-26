"""

"""


# Built-in
import os

# Libs

# Own modules
from utils import misc_utils

city = [# 'NZ_Dunedin_{}',
        # 'NZ_Gisborne_{}',
        # 'NZ_Palmerston-North_{}',
        # 'NZ_Rotorua_{}',
        # 'NZ_Tauranga_{}',
        # 'AZ_Tucson_{}',
        'KS_Colwich-Maize_{}'
        ]
idx = [
    # list(range(1, 7)),
    # list(range(1, 7)),
    # list(range(1, 15)),
    # list(range(1, 8)),
    # list(range(1, 7)),
    # list(range(1, 27)),
    list(range(1, 49)),
]


def main():
    data_dir = r'/media/ei-edl01/data/transmission/eccv/img'
    total_area = 0
    for city_name, id_list in zip(city, idx):
        for city_id in id_list:
            img_name = os.path.join(data_dir, '{}.jpg'.format(city_name.format(city_id)))
            img = misc_utils.load_file(img_name)
            total_area += img.shape[0] * 0.3 * 1e-3 * img.shape[1] * 0.3 * 1e-3
    print(total_area)


if __name__ == '__main__':
    main()
