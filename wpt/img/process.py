"""
Image processing

This file conatins the processing of chain movement. for now, it only takes into
account the 2D movement, completely ignoring the 3D motion.
"""

import os
import cv2 as cv
import numpy as np
import networkx as nx
import json

from functools import partial
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from numpy.typing import ArrayLike
from cv2.typing import MatLike
from itertools import product
from collections.abc import Mapping
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

os.environ["QT_LOGGING_RULES"] = "*.warning=false"
os.environ["QT_QPA_PLATFORM"] = "xcb"


def extract_outline_values(img: MatLike, width: int = 1) -> MatLike:
    top = img[:width, :].flatten()
    bottom = img[-width:, :].flatten()
    left = img[:, :width].flatten()
    right = img[:, -width:].flatten()

    output = []
    for x in (top, bottom, left, right):
        output.extend(x)
    output = np.array(output)

    return cv.Mat(output)


def convert_to_binary(img: MatLike, thresh: int, invert: bool = True) -> MatLike:
    _, img_binary = cv.threshold(img, thresh, 255, cv.THRESH_BINARY_INV if invert else cv.THRESH_BINARY)
    return img_binary


def preprocess_and_skeletonize(img, thresh=None):
    """Complete pre-thinning pipeline with area-based filtering."""

    # Blurring for salt-and-pepper noise removal
    img_blurred = cv.medianBlur(img, 5)

    if thresh is None:
        thresh = np.mean(np.array(extract_outline_values(img_blurred, 3))) - 10

    # Binary conversion
    img_binary = convert_to_binary(img_blurred, thresh)

    # Morphological closing to fill gaps
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    img_binary_cleaned = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel)

    # Connected components with area-based filtering
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(img_binary_cleaned)

    if num_labels > 1:
        # Filter: Keep only components larger than 5% of the largest
        largest_area = np.max(stats[1:, cv.CC_STAT_AREA])
        min_area_threshold = largest_area * 0.05

        # Build mask with only valid components
        valid_labels = []
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv.CC_STAT_AREA]
            if area >= min_area_threshold:
                valid_labels.append(i)

        # Create final binary mask
        if valid_labels:
            binary_cleaned = np.zeros_like(img_binary_cleaned)
            for label in valid_labels:
                binary_cleaned[labels == label] = 255
        else:
            # Fallback: keep the largest component
            largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
            binary_cleaned = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    else:
        binary_cleaned = img_binary_cleaned

    # Generate skeleton
    img_skeleton = cv.ximgproc.thinning( # type: ignore
        binary_cleaned, thinningType=cv.ximgproc.THINNING_ZHANGSUEN # type: ignore
    )
    return img_skeleton


def extract_clean_ordered_path(skeleton_img, anchor_pt=None):
    """
    Converts skeleton pixels to a single ordered path from anchor to free end.

    Args:
        skeleton_img: Binary skeleton image (0 and 255)
        anchor_pt: (x, y) coordinates of the fixed anchor point (optional)
                   If None, automatically detects the endpoint with lowest variance

    Returns:
        path_array: Ordered list of (x, y) coordinates along the rope
    """
    # Step 1: Extract all skeleton pixel coordinates
    y_indices, x_indices = np.where(skeleton_img > 0)
    points = set(zip(x_indices, y_indices))

    if not points:
        # No points extracted.
        return np.array([])

    # Build graph connecting adjacent pixels.
    graph = nx.Graph()
    for point in points:
        graph.add_node(point)

        for dx, dy in product([-1, 0, 1], repeat=2):
            if (dx, dy) == (0, 0):
                continue
            neighbor = (point[0] + dx, point[1] + dy)
            if neighbor in points:
                weight = 1.414 if (dx != 0 and dy != 0) else 1.0
                graph.add_edge(point, neighbor, weight=weight)

    # Find longest path through the graph

    # this sperates the graph into components (curves) that are separated
    components = list(nx.connected_components(graph))

    if not components:
        return np.array([])

    # Take the longest one.
    main_component = max(components, key=len)

    subgraph = graph.subgraph(main_component)

    # Points thar are open-ended, that is are only connected on onse side (extremity).
    endpoints = [node for node, degree in subgraph.degree() if degree == 1]

    if not endpoints:
        endpoints = list(subgraph.nodes())

    max_path_len = 0
    longest_path = []

    for start_node in endpoints[:10]:
        distance, path = nx.single_source_dijkstra(
            subgraph, start_node, weight="weight"
        )

        assert isinstance(distance, Mapping) and isinstance(path, Mapping)

        for target_node, path_nodes in path.items():
            if distance[target_node] > max_path_len:
                max_path_len = distance[target_node]
                longest_path = path_nodes

    # Once the longest path is found, convert to NumPy array
    path_array = np.array(longest_path, dtype=np.float32)

    # Enforce anchor direction (with auto-detection fallback)
    if len(path_array) > 0:
        if anchor_pt is not None:
            # Use provided anchor point
            dist_start = np.sum((path_array[0] - anchor_pt) ** 2)
            dist_end = np.sum((path_array[-1] - anchor_pt) ** 2)
            if dist_end < dist_start:
                path_array = np.flip(path_array, axis=0)
        else:
            # Auto-detect anchor: endpoint with lower y-coordinate (usually top)
            # Or use the first endpoint as default.
            # This assumes the anchor is typically at the top of the image
            if (
                path_array[0][1] > path_array[-1][1]
            ):  # If first point is lower than last
                path_array = np.flip(path_array, axis=0)

    return path_array


def calculate_path_length(path: np.ndarray) -> float:
    """Calculate total length of the path in pixels.

    Args:
        path
    """
    total_length = 0.0

    # Simple sum by displacement between succesive nodes.
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        total_length += np.sqrt(dx**2 + dy**2)

    return total_length


def resample_path_to_n_nodes(path: np.ndarray, n_nodes: int) -> np.ndarray:
    """
    Resample a continuous path to exactly `n_nodes` evenly-spaced points.

    Args:
        path (np.ndarray): Array of (x, y) coordinates (continuous skeleton)
        n_nodes (int): Number of nodes to create (e.g., 20)

    Returns:
        resampled_path (np.ndarray): Array of n_nodes (x, y) coordinates
    """
    if len(path) < 2:
        return path  # Not enough points to resample

    # Calculate cumulative distance along the path
    distances = [0.0]
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]

        distances.append(distances[-1] + np.sqrt(dx**2 + dy**2))

    total_length = distances[-1]

    # Target spacing between nodes
    # spacing = total_length / (n_nodes - 1)

    # Generate target distances for each node (based on the curve length, not euclidian distance between nodes.)
    target_distances = np.linspace(0, total_length, n_nodes)

    # Interpolate to find coordinates at each target distance
    resampled_path = []

    for target_dist in target_distances:
        # Find which segment contains this distance
        idx = np.searchsorted(distances, target_dist)
        idx = min(idx, len(path) - 2)  # Clamp to valid range

        # Linear interpolation between path[idx] and path[idx+1] (The node is located between these two path points)
        seg_start_dist = distances[idx]
        seg_end_dist = distances[idx + 1]

        if seg_end_dist == seg_start_dist:
            # Avoid division by zero
            t = 0.0
        else:
            t = (target_dist - seg_start_dist) / (seg_end_dist - seg_start_dist)

        # Interpolate coordinates
        x = path[idx][0] + t * (path[idx + 1][0] - path[idx][0])
        y = path[idx][1] + t * (path[idx + 1][1] - path[idx][1])

        resampled_path.append([x, y])

    return np.array(resampled_path, dtype=np.float32)


def visualize_nodes(vis_img, resampled_path: np.ndarray, node_radius=4) -> None:
    """Draw resampled nodes on the visualization image."""
    for i, (x, y) in enumerate(resampled_path):
        # Draw node
        cv.circle(vis_img, (int(x), int(y)), node_radius, (255, 0, 0), -1)

        # Label node index
        cv.putText(
            vis_img,
            str(i),
            (int(x) - 10, int(y) - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

    # Draw lines connecting nodes
    for i in range(len(resampled_path) - 1):
        pt1 = tuple(resampled_path[i].astype(int))
        pt2 = tuple(resampled_path[i + 1].astype(int))
        cv.line(vis_img, pt1, pt2, (0, 255, 0), 2)


def get_minimum_distance_index_array(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Get minimum distance index array

    Outputs the index order of `b` based on `a` order. That is `a` order is
    unchanged and outputs the reordering of `b` so as to match `a` order.

    Args:
        a (ArrayLike): First array
        b (ArrayLike): Second array, which is to be ordered according to `a`
    """

    cost_matrix = cdist(a, b, metric="euclidean")

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assert isinstance(row_ind, np.ndarray) and isinstance(col_ind, np.ndarray)

    index_array = np.argsort(row_ind)

    return col_ind[index_array]


def extract_nodes_from_image(
    img: MatLike | str,
    n_nodes: int = 20,
    thresh: int | None = None,
    anchor_pt: np.ndarray | None = None,
) -> np.ndarray | None:
    if isinstance(img, str):
        out = cv.imread(img, cv.IMREAD_GRAYSCALE)

        if out is None:
            return

        img = out

    # --- Pre-thinning ---
    img_skeleton = preprocess_and_skeletonize(img, thresh)

    # --- Extract Path ---
    path = extract_clean_ordered_path(img_skeleton, anchor_pt)

    # --- Resample to N Nodes ---
    resampled_path = resample_path_to_n_nodes(path, n_nodes)

    return resampled_path


def extract_info_from_image(
    frame_timestamp_tuple: tuple,
    n_nodes: int = 20,
    thresh: int | None = None,
    anchor_pt: np.ndarray | None = None,
):
    frame, timestamp = frame_timestamp_tuple
    return extract_nodes_from_image(frame, n_nodes, thresh, anchor_pt), timestamp


def extract_nodes_from_video(
    vid: cv.VideoCapture | str,
    n_nodes: int = 20,
    max_workers: int | None = None,
    save_to_file: str | None = None,
    thresh: int | None = None,
    anchor_pt: np.ndarray | None = None,
) -> dict[str, dict[str, int] | list[dict[str, list[list[int | float]] | float | int]]]:
    if isinstance(vid, str):
        vid = cv.VideoCapture(vid)

    total_frames = int(vid.get(cv.CAP_PROP_FRAME_COUNT))

    def read_frame():
        ret, frame = vid.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            timestamp = vid.get(cv.CAP_PROP_POS_MSEC) / 1000
            return (frame, timestamp)
        return None

    def frame_generator():
        while True:
            data = read_frame()
            if data is None:
                break
            yield data

    output = []
    nodes_list = []
    timestamps = []

    with ProcessPoolExecutor(max_workers) as executor:
        # Use partial to pre-configure n_nodes and thresh for each parallel worker
        worker_func = partial(extract_info_from_image, n_nodes=n_nodes, thresh=thresh, anchor_pt=anchor_pt)
        results = executor.map(worker_func, frame_generator())

        for result in tqdm(results, total=total_frames, desc="Processing Frames"):
            if result is None:
                continue

            nodes_list.append(result[0])
            timestamps.append(result[1])

    dt_list = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

    for frame, (nodes, dt) in enumerate(zip(nodes_list[1:], dt_list)):
        output.append(
            {
                "nodes": nodes.tolist(),
                "velocity": ((nodes - nodes_list[frame]) / dt).tolist(),
                "dt": dt,
                "frame": frame,
            }
        )

    output = {
        "resolution": {
            "width": int(vid.get(cv.CAP_PROP_FRAME_WIDTH)),
            "height": int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)),
        },
        "frames": output[:],
    }

    if save_to_file is not None:
        with open(save_to_file, "w") as f:
            json.dump(output, f, indent=2)

    return output


def visualize_nodes_in_video(fp: str, original_video: str | None = None):
    with open(fp) as f:
        data = json.load(f)
    
    resolution = data["resolution"]
    target_w, target_h = resolution["width"], resolution["height"]
    
    cap = None
    if original_video is not None:
        cap = cv.VideoCapture(original_video)
        if not cap.isOpened():
            print(f"Warning: Could not open original video '{original_video}'.")
            cap = None

    img_black = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # -------------------------------------------------------------
    # 1. CONFIGURE WINDOW HERE
    # -------------------------------------------------------------
    # This enables manual dragging and programmatic resizing
    cv.namedWindow("Overlay Nodes", cv.WINDOW_NORMAL)
    
    # Optional: Force the window to start at a specific size (e.g., 960x540)
    # cv.resizeWindow("Overlay Nodes", 960, 540)
    # -------------------------------------------------------------

    try:
        for frame_data in data["frames"]:
            vis_img = None
            if cap is not None:
                ret, frame = cap.read()
                if ret:
                    if frame.shape[1] != target_w or frame.shape[0] != target_h:
                        vis_img = cv.resize(frame, (target_w, target_h))
                    else:
                        vis_img = frame.copy()
            
            if vis_img is None:
                vis_img = img_black.copy()

            visualize_nodes(vis_img, np.array(frame_data["nodes"]))

            # Show frame inside the pre-declared normal window
            cv.imshow("Overlay Nodes", vis_img)
            
            if cv.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        if cap is not None:
            cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    vid = cv.VideoCapture("media/vids/vid.mp4")
    # ret, frame = vid.read()
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow("bw_frame", frame)

    # thresh = 80
    # frame = preprocess_and_skeletonize(frame, thresh=thresh)

    # Blurring for salt-and-pepper noise removal
    # img_blurred = cv.medianBlur(frame, 5)

    # if thresh is None:
    #     thresh = np.mean(np.array(extract_outline_values(img_blurred, 3))) - 10

    # Binary conversion
    # img_binary = convert_to_binary(frame, thresh)

    # # nodes = extract_nodes_from_image(frame)

    # cv.namedWindow("Frame", cv.WINDOW_NORMAL)
    # # visualize_nodes(frame, nodes)
    # cv.imshow("Frame", img_binary)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    output = extract_nodes_from_video(vid, save_to_file="output.json", thresh=130, anchor_pt=np.array([500,0]))

    # with open("output.json", "w") as f:
    #     json.dump(output, f, indent=2)

    visualize_nodes_in_video("output.json", "media/vids/vid.mp4")

    # np.save("output.npy", output)
    # array: np.ndarray = np.load("output.npy", allow_pickle=True)
    # array = np.array(array, dtype=object)

    # for i in range(len(array)):
    #     array[i]["nodes"] = array[i]["nodes"].tolist()
    #     array[i]["velocity"] = array[i]["velocity"].tolist()

    # array = list(array)  # type: ignore

    # print(output)
    # # Check if camera opened successfully
    # if vid.isOpened() is False:
    #     print("Error opening video stream or file")

    # # Read until video is completed
    # while vid.isOpened():
    #     # Capture frame-by-frame
    #     ret, frame = vid.read()
    #     if ret == True:

    #         # Display the resulting frame
    #         cv.imshow("Frame", frame)

    #         # Press Q on keyboard to  exit
    #         if cv.waitKey(25) & 0xFF == ord("q"):
    #             break

    #     # Break the loop
    #     else:
    #         break

    # # When everything done, release the video capture object
    # vid.release()

    # img = cv.imread("media/imgs/rope.ppm", cv.IMREAD_GRAYSCALE)
    # assert img is not None

    # # --- Pre-thinning ---
    # img_skeleton = preprocess_and_skeletonize(img)

    # # --- Extract Path ---
    # path = extract_clean_ordered_path(img_skeleton)

    # # --- Resample to N Nodes ---
    # n_nodes = 20  # Adjust based on your needs
    # resampled_path = resample_path_to_n_nodes(path, n_nodes)

    # # --- Visualize ---
    # vis_img = cv.cvtColor(img_skeleton, cv.COLOR_GRAY2BGR)
    # visualize_nodes(vis_img, resampled_path)

    # cv.imshow("Resampled Nodes", vis_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
