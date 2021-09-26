"""
This file defines handy function for making the dataset
"""


# Built-in
import math

# Libs
import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from scipy.ndimage.morphology import distance_transform_edt

# Own modules
from data import graph_utils, sknw


def get_center_point(ymin, xmin, ymax, xmax):
    return (ymin+ymax)/2, (xmin+xmax)/2


def csv_to_annotation(file_name):
    node_info = []
    adj_mat = []
    adj_flag = False
    with open(file_name, 'r') as f:
        line = f.readline()
        while line:
            if adj_flag:
                adj_mat.append([int(a) for a in line.strip().split(',')[:-1]])
            else:
                box = tuple([int(a) for a in line.strip().split(',')[:-1]])
                y, x = get_center_point(*box)
                node_info.append({'box': box, 'center': [int(y), int(x)]})
            line = f.readline()
            if line == '\n':
                adj_flag = True
                line = f.readline()
    adj_mat = np.array(adj_mat)
    return node_info, adj_mat


def render_line_graph(img_size, nodes, adj_matrix, thickness=30):
    line_map = np.zeros(img_size)
    line_pts = []
    x_idx, y_idx = np.where(adj_matrix==1)
    for x, y in zip(x_idx, y_idx):
        assert adj_matrix[x][y] == 1
        pt1 = tuple(nodes[x]['center'])
        pt2 = tuple(nodes[y]['center'])
        cv2.line(line_map, pt1[::-1], pt2[::-1], color=1, thickness=thickness)

        # get points on the line
        temp_map = np.zeros(img_size)
        cv2.line(temp_map, pt1[::-1], pt2[::-1], color=1, thickness=thickness)
        x_idx, y_idx = np.where(temp_map == 1)
        pts = np.stack([x_idx, y_idx], axis=1)
        line_pts.append(pts)

    return line_map, line_pts


def render_angle_map(line_map, nodes, adj_matrix, line_pts, multi=False):
    # create graph
    graph = nx.MultiDiGraph() if multi else nx.Graph()
    for i in range(len(nodes)):
        graph.add_node(i, pts=[nodes[i]], o=nodes[i])
    x_idx, y_idx = np.where(adj_matrix == 1)
    for cnt, (x, y) in enumerate(zip(x_idx, y_idx)):
        ln = np.linalg.norm(np.array(nodes[x]['center'])-np.array(nodes[y]['center']))
        graph.add_edge(x, y, pts=line_pts[cnt], weight=ln)

    # make distance array
    distance_array = distance_transform_edt(1 - line_map)
    std = 15
    distance_array = np.exp(-0.5 * (distance_array * distance_array) / (std * std))
    distance_array *= 255

    keypoints = getKeypoints(distance_array, smooth_dist=4)
    h, w = line_map.shape
    vecmap_euclidean, orienation_angles = getVectorMapsAngles((h, w), keypoints, theta=10, bin_size=10)

    return vecmap_euclidean, orienation_angles


def getKeypoints(mask, thresh=0.8, is_gaussian=True, is_skeleton=False, smooth_dist=4):
    """
    Generate keypoints for binary prediction mask.
    @param mask: Binary road probability mask
    @param thresh: Probability threshold used to cnvert the mask to binary 0/1 mask
    @param gaussian: Flag to check if the given mask is gaussian/probability mask
                    from prediction
    @param is_skeleton: Flag to perform opencv skeletonization on the binarized
                        road mask
    @param smooth_dist: Tolerance parameter used to smooth the graph using
                        RDP algorithm
    @return: return ndarray of road keypoints
    """

    if is_gaussian:
        mask /= 255.0
        mask[mask < thresh] = 0
        mask[mask >= thresh] = 1

    h, w = mask.shape
    if is_skeleton:
        ske = mask
    else:
        ske = skeletonize(mask).astype(np.uint16)
    graph = sknw.build_sknw(ske, multi=True)

    segments = graph_utils.simplify_graph(graph, smooth_dist)
    linestrings_1 = graph_utils.segmets_to_linestrings(segments)
    linestrings = graph_utils.unique(linestrings_1)

    keypoints = []
    for line in linestrings:
        linestring = line.rstrip("\n").split("LINESTRING ")[-1]
        points_str = linestring.lstrip("(").rstrip(")").split(", ")
        ## If there is no road present
        if "EMPTY" in points_str:
            return keypoints
        points = []
        for pt_st in points_str:
            x, y = pt_st.split(" ")
            x, y = float(x), float(y)
            points.append([x, y])

            x1, y1 = points[0]
            x2, y2 = points[-1]
            zero_dist1 = math.sqrt((x1) ** 2 + (y1) ** 2)
            zero_dist2 = math.sqrt((x2) ** 2 + (y2) ** 2)

            if zero_dist2 > zero_dist1:
                keypoints.append(points[::-1])
            else:
                keypoints.append(points)
    return keypoints


def getVectorMapsAngles(shape, keypoints, theta=5, bin_size=10):
    """
    Convert Road keypoints obtained from road mask to orientation angle mask.
    Reference: Section 3.1
        https://anilbatra2185.github.io/papers/RoadConnectivityCVPR2019.pdf
    @param shape: Road Label/PIL image shape i.e. H x W
    @param keypoints: road keypoints generated from Road mask using
                        function getKeypoints()
    @param theta: thickness width for orientation vectors, it is similar to
                    thicknes of road width with which mask is generated.
    @param bin_size: Bin size to quantize the Orientation angles.
    @return: Retun ndarray of shape H x W, containing orientation angles per pixel.
    """

    im_h, im_w = shape
    vecmap = np.zeros((im_h, im_w, 2), dtype=np.float32)
    vecmap_angles = np.zeros((im_h, im_w), dtype=np.float32)
    vecmap_angles.fill(360)
    height, width, channel = vecmap.shape
    for j in range(len(keypoints)):
        for i in range(1, len(keypoints[j])):
            a = keypoints[j][i - 1]
            b = keypoints[j][i]
            ax, ay = a[0], a[1]
            bx, by = b[0], b[1]
            bax = bx - ax
            bay = by - ay
            norm = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9
            bax /= norm
            bay /= norm

            min_w = max(int(round(min(ax, bx) - theta)), 0)
            max_w = min(int(round(max(ax, bx) + theta)), width)
            min_h = max(int(round(min(ay, by) - theta)), 0)
            max_h = min(int(round(max(ay, by) + theta)), height)

            for h in range(min_h, max_h):
                for w in range(min_w, max_w):
                    px = w - ax
                    py = h - ay
                    dis = abs(bax * py - bay * px)
                    if dis <= theta:
                        vecmap[h, w, 0] = bax
                        vecmap[h, w, 1] = bay
                        _theta = math.degrees(math.atan2(bay, bax))
                        vecmap_angles[h, w] = (_theta + 360) % 360

    vecmap_angles = (vecmap_angles / bin_size).astype(int)
    return vecmap, vecmap_angles
