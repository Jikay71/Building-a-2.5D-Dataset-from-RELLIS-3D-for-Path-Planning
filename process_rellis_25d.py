#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RELLIS-3D to 2.5D Semantic Elevation Costmap & Path Planning Pipeline
=====================================================================
Quy trình 3 bước hoàn chỉnh:
  1. Trích xuất & Tiền xử lý (ROI, Ego-vehicle & Driver Filter, Outlier Filter)
  2. Xây dựng Lưới độ cao 2.5D (Elevation, Slope, Roughness, Step-height)
  3. Tích hợp Ngữ nghĩa & Bản đồ Chi phí (Semantic Costmap & A* Path Planning Benchmark)

Được thiết kế tối ưu hóa tốc độ cao bằng NumPy & SciPy vectorization.
Tác giả: Nhóm Nghiên cứu RELLIS-3D 2.5D Path Planning
"""

import os
import sys
import io

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi encoding cp1252
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import time
import zipfile
import threading
import argparse
from pathlib import Path

# Quản lý file zip riêng cho từng worker thread để tối đa hóa tốc độ đọc đồng thời
_local = threading.local()

def _get_thread_zip(zip_path):
    if not hasattr(_local, 'zips'):
        _local.zips = {}
    if zip_path not in _local.zips:
        _local.zips[zip_path] = zipfile.ZipFile(zip_path, 'r')
    return _local.zips[zip_path]

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_dilation, generate_binary_structure
import heapq


# ==============================================================================
# 0. BẢNG CẤU HÌNH VÀ BẢNG MÀU CHUẨN RELLIS-3D
# ==============================================================================
RELLIS_NAMES = {
    0: "void",
    1: "dirt",
    3: "grass",
    4: "tree",
    5: "pole",
    6: "water",
    7: "sky",
    8: "vehicle",
    9: "object",
    10: "asphalt",
    12: "building",
    15: "log",
    17: "person",
    18: "fence",
    19: "bush",
    23: "concrete",
    27: "barrier",
    31: "puddle",
    33: "mud",
    34: "rubble"
}

RELLIS_COLORS = {
    0: (0, 0, 0),          # void: đen
    1: (108, 64, 20),      # dirt: nâu đất
    3: (0, 102, 0),        # grass: xanh lá đậm
    4: (0, 255, 0),        # tree: xanh lá tươi
    5: (0, 153, 153),      # pole: xanh lơ
    6: (0, 128, 255),      # water: xanh nước biển
    7: (0, 0, 255),        # sky: xanh dương
    8: (255, 255, 0),      # vehicle: vàng
    9: (255, 0, 127),      # object: hồng sen
    10: (64, 64, 64),      # asphalt: xám đậm
    12: (255, 0, 0),       # building: đỏ
    15: (102, 0, 0),       # log: nâu đỏ
    17: (204, 153, 255),   # person: tím nhạt
    18: (102, 0, 204),     # fence: tím thẫm
    19: (255, 153, 204),   # bush: hồng nhạt
    23: (170, 170, 170),   # concrete: xám sáng
    27: (41, 121, 255),    # barrier: xanh biển đậm
    31: (134, 255, 239),   # puddle: xanh nước sáng
    33: (99, 66, 34),      # mud: nâu bùn
    34: (110, 22, 138)     # rubble: tím than
}

# Chi phí vượt địa hình theo ngữ nghĩa (0: tối ưu nhất, 255: cấm đi)
SEMANTIC_COST_TABLE = {
    0: 100,    # void/unlabeled (vùng chưa rõ nhãn)
    1: 15,     # dirt (đường đất khô)
    3: 35,     # grass (bãi cỏ thấp)
    4: 255,    # tree (cây lớn - cấm đi)
    5: 255,    # pole (cột - cấm đi)
    6: 255,    # water (vùng nước sâu - cấm đi)
    7: 100,    # sky
    8: 255,    # vehicle (xe khác - cấm đi)
    9: 180,    # object (vật cản chưa rõ)
    10: 10,    # asphalt (đường nhựa - tối ưu)
    12: 255,   # building (công trình - cấm đi)
    15: 255,   # log (khúc gỗ đổ - cấm đi)
    17: 255,   # person (người ngoài xe - cấm đi)
    18: 255,   # fence (hàng rào - cấm đi)
    19: 120,   # bush (bụi cây rậm - cản trở mạnh)
    23: 10,    # concrete (đường bê tông - tối ưu)
    27: 255,   # barrier (dải phân cách - cấm đi)
    31: 75,    # puddle (vũng nước nông - thận trọng)
    33: 110,   # mud (bùn lầy - nguy cơ sa lầy cao)
    34: 90     # rubble (đất đá vụn gồ ghề)
}

# Nhóm nhãn vật cản tuyệt đối (Lethal Obstacles)
LETHAL_CLASSES = {4, 5, 6, 8, 9, 12, 15, 17, 18, 27}


class RellisConfig:
    """Tham số cấu hình hệ thống lưới 2.5D"""
    # Độ phân giải ô lưới (mét/ô)
    RES = 0.2

    # Vùng quan tâm (Region of Interest - ROI) theo hệ trục xe (Ouster OS1)
    # X: Hướng về phía trước (+), phía sau (-)
    # Y: Hướng sang trái (+), sang phải (-)
    X_MIN = -10.0
    X_MAX = 30.0   # Tầm nhìn 30m phía trước
    Y_MIN = -15.0  # 15m sang phải
    Y_MAX = 15.0   # 15m sang trái
    Z_MIN = -3.0   # Giới hạn cao độ dưới
    Z_MAX = 5.0    # Giới hạn cao độ trên

    # Kích thước ma trận lưới
    # H (dọc theo X: 40m / 0.2 = 200 ô)
    # W (ngang theo Y: 30m / 0.2 = 150 ô)
    GRID_H = int(round((X_MAX - X_MIN) / RES))  # 200 hàng
    GRID_W = int(round((Y_MAX - Y_MIN) / RES))  # 150 cột

    # Kích thước hình học thân xe Polaris W-MRZR (để lọc sạch thân xe và người lái)
    # Cảm biến Ouster đặt ở khung sau nhìn về phía trước xe
    VEHICLE_X_MIN = -1.2
    VEHICLE_X_MAX = 2.4
    VEHICLE_Y_MIN = -1.3
    VEHICLE_Y_MAX = 1.3
    VEHICLE_Z_MIN = -1.3

    # Giới hạn vật lý vượt địa hình của xe
    MAX_SLOPE_DEG = 30.0       # Độ dốc tối đa có thể leo (độ)
    MAX_STEP_HEIGHT = 0.35     # Bước cản đứng tối đa gầm xe vượt qua (m)
    MAX_ROUGHNESS = 0.25       # Độ gồ ghề tối đa (m)

    # Trọng số tính Costmap tổng thể
    W_SEM = 0.50
    W_SLOPE = 0.30
    W_ROUGH = 0.20

    # Bán kính mở rộng an toàn quanh vật cản (số ô lưới)
    INFLATION_CELLS = 2


# ==============================================================================
# BƯỚC 1: TRÍCH XUẤT VÀ TIỀN XỬ LÝ DỮ LIỆU
# ==============================================================================
class RellisDataLoader:
    """Quản lý nạp dữ liệu từ thư mục giải nén hoặc trực tiếp từ file nén .zip"""
    def __init__(self, workspace_dir="."):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.bin_source = None
        self.label_source = None
        self._find_data_sources()

    def _find_data_sources(self):
        # 1. Tìm nguồn BIN
        bin_zip_candidates = [
            self.workspace_dir / "Rellis_3D_os1_cloud_node_kitti_bin.zip",
            self.workspace_dir.parent / "Rellis_3D_os1_cloud_node_kitti_bin.zip",
            self.workspace_dir.parent / "RELLIS_3D" / "Rellis_3D_os1_cloud_node_kitti_bin.zip"
        ]
        for p in bin_zip_candidates:
            if p.is_file():
                self.bin_source = ("zip", p)
                break

        if not self.bin_source:
            for p in [self.workspace_dir, self.workspace_dir.parent]:
                for candidate in p.rglob("os1_cloud_node_kitti_bin"):
                    if candidate.is_dir() and any(candidate.glob("*.bin")):
                        self.bin_source = ("dir", candidate)
                        break
                if self.bin_source:
                    break

        # 2. Tìm nguồn LABEL
        label_zip_candidates = [
            self.workspace_dir / "Rellis_3D_os1_cloud_node_semantickitti_label_id_20210614.zip",
            self.workspace_dir.parent / "RELLIS_3D" / "Rellis_3D_os1_cloud_node_semantickitti_label_id_20210614.zip",
            self.workspace_dir.parent / "Rellis_3D_os1_cloud_node_semantickitti_label_id_20210614.zip"
        ]
        for p in label_zip_candidates:
            if p.is_file():
                self.label_source = ("zip", p)
                break

        if not self.label_source:
            for p in [self.workspace_dir, self.workspace_dir.parent]:
                for candidate in p.rglob("os1_cloud_node_semantickitti_label_id"):
                    if candidate.is_dir() and any(candidate.glob("*.label")):
                        self.label_source = ("dir", candidate)
                        break
                if self.label_source:
                    break

        print(f"[DataLoader] Nguồn LiDAR BIN: {self.bin_source}")
        print(f"[DataLoader] Nguồn Semantics LABEL: {self.label_source}")

    def get_all_frame_ids(self, sequence_id="00000"):
        """Lấy danh sách toàn bộ frame ID có trong sequence"""
        frame_ids = []
        if self.bin_source[0] == "zip":
            z = _get_thread_zip(self.bin_source[1])
            prefix = f"Rellis-3D/{sequence_id}/os1_cloud_node_kitti_bin/"
            for n in z.namelist():
                if n.startswith(prefix) and n.endswith(".bin"):
                    frame_ids.append(Path(n).stem)
        else:
            seq_dir = self.bin_source[1]
            for p in seq_dir.glob("*.bin"):
                frame_ids.append(p.stem)
        frame_ids.sort()
        return frame_ids

    def load_frame(self, sequence_id="00000", frame_id="000000"):
        """Nạp point cloud và nhãn cho một frame (hỗ trợ thread-safe I/O)"""
        pts = None
        lbls = None

        # Đọc BIN
        if self.bin_source[0] == "zip":
            z = _get_thread_zip(self.bin_source[1])
            internal_path = f"Rellis-3D/{sequence_id}/os1_cloud_node_kitti_bin/{frame_id}.bin"
            try:
                data = z.read(internal_path)
                pts = np.frombuffer(data, dtype=np.float32).reshape(-1, 4)
            except KeyError:
                matching = [n for n in z.namelist() if n.endswith(f"{frame_id}.bin") and sequence_id in n]
                if matching:
                    data = z.read(matching[0])
                    pts = np.frombuffer(data, dtype=np.float32).reshape(-1, 4)
        else:
            file_path = self.bin_source[1] / f"{frame_id}.bin"
            if file_path.is_file():
                pts = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, 4)

        # Đọc LABEL
        if self.label_source and self.label_source[0] == "zip":
            z = _get_thread_zip(self.label_source[1])
            internal_path = f"Rellis-3D/{sequence_id}/os1_cloud_node_semantickitti_label_id/{frame_id}.label"
            try:
                data = z.read(internal_path)
                lbls = np.frombuffer(data, dtype=np.uint32) & 0xFFFF
            except KeyError:
                matching = [n for n in z.namelist() if n.endswith(f"{frame_id}.label") and sequence_id in n]
                if matching:
                    data = z.read(matching[0])
                    lbls = np.frombuffer(data, dtype=np.uint32) & 0xFFFF
        elif self.label_source:
            file_path = self.label_source[1] / f"{frame_id}.label"
            if file_path.is_file():
                lbls = np.fromfile(str(file_path), dtype=np.uint32) & 0xFFFF

        if pts is None:
            raise FileNotFoundError(f"Không tìm thấy frame BIN {frame_id} của sequence {sequence_id}!")
        if lbls is None:
            lbls = np.zeros(len(pts), dtype=np.uint16)

        return pts, lbls


def preprocess_point_cloud(pts, lbls, cfg: RellisConfig):
    """
    Bước 1: Làm sạch và lọc dữ liệu:
      - Giới hạn ROI không gian quanh xe
      - Lọc bỏ thân xe tự hành và người lái ngồi trong xe (Ego-vehicle bounding box)
      - Lọc bỏ điểm nhiễu ngoại lai
    """
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    # 1. Mặt nạ lọc ROI
    roi_mask = (
        (x >= cfg.X_MIN) & (x < cfg.X_MAX) &
        (y >= cfg.Y_MIN) & (y < cfg.Y_MAX) &
        (z >= cfg.Z_MIN) & (z < cfg.Z_MAX)
    )

    # 2. Lọc bỏ thân xe và người lái (Vehicle & Driver 3D box)
    in_vehicle_box = (
        (x >= cfg.VEHICLE_X_MIN) & (x <= cfg.VEHICLE_X_MAX) &
        (y >= cfg.VEHICLE_Y_MIN) & (y <= cfg.VEHICLE_Y_MAX) &
        (z >= cfg.VEHICLE_Z_MIN)
    )

    valid_mask = roi_mask & (~in_vehicle_box)

    x_filt = x[valid_mask]
    y_filt = y[valid_mask]
    z_filt = z[valid_mask]
    intensity_filt = pts[valid_mask, 3]
    lbls_filt = lbls[valid_mask]

    return x_filt, y_filt, z_filt, intensity_filt, lbls_filt


# ==============================================================================
# BƯỚC 2: XÂY DỰNG LƯỚI ĐỘ CAO 2.5D (ELEVATION GRID)
# ==============================================================================
def build_elevation_grid(x, y, z, lbls, cfg: RellisConfig):
    """
    Bước 2: Vector hóa chuyển đổi từ 3D Point Cloud sang lưới 2.5D:
      - z_elevation: độ cao địa hình mặt đất (ground elevation)
      - roughness: độ gồ ghề (độ lệch chuẩn cao độ trong ô)
      - step_height: bước cản đứng cục bộ (z_max - z_min)
      - slope: độ dốc địa hình tính từ gradient không gian
      - semantic_grid: nhãn ngữ nghĩa kết hợp bầu chọn đa số và nội suy liên tục
    """
    H, W = cfg.GRID_H, cfg.GRID_W
    total_cells = H * W

    # Tính chỉ số hàng (r) và cột (c) trong ảnh BEV
    # Trục X (tiến): x = X_MAX -> hàng 0 (đỉnh ảnh), x = X_MIN -> hàng H-1 (đáy ảnh)
    # Trục Y (trái-phải): y = Y_MAX -> cột 0 (trái ảnh), y = Y_MIN -> cột W-1 (phải ảnh)
    row_idx = np.floor((cfg.X_MAX - x) / cfg.RES).astype(np.int32)
    col_idx = np.floor((cfg.Y_MAX - y) / cfg.RES).astype(np.int32)

    row_idx = np.clip(row_idx, 0, H - 1)
    col_idx = np.clip(col_idx, 0, W - 1)

    cell_flat_idx = row_idx * W + col_idx

    # 1. Thống kê số lượng điểm mỗi ô
    count = np.zeros(total_cells, dtype=np.int32)
    np.add.at(count, cell_flat_idx, 1)
    valid_cell_mask = count > 0

    # 2. Tính Z min, Z max, Z sum, Z sum bình phương
    z_min_flat = np.full(total_cells, np.inf, dtype=np.float32)
    z_max_flat = np.full(total_cells, -np.inf, dtype=np.float32)
    z_sum_flat = np.zeros(total_cells, dtype=np.float32)
    z_sq_sum_flat = np.zeros(total_cells, dtype=np.float32)

    np.minimum.at(z_min_flat, cell_flat_idx, z)
    np.maximum.at(z_max_flat, cell_flat_idx, z)
    np.add.at(z_sum_flat, cell_flat_idx, z)
    np.add.at(z_sq_sum_flat, cell_flat_idx, z**2)

    # 3. Tính độ cao đại diện, độ gồ ghề và bước cản đứng
    z_mean_flat = np.zeros(total_cells, dtype=np.float32)
    roughness_flat = np.zeros(total_cells, dtype=np.float32)
    step_height_flat = np.zeros(total_cells, dtype=np.float32)

    z_mean_flat[valid_cell_mask] = z_sum_flat[valid_cell_mask] / count[valid_cell_mask]
    var_z = (z_sq_sum_flat[valid_cell_mask] / count[valid_cell_mask]) - (z_mean_flat[valid_cell_mask]**2)
    roughness_flat[valid_cell_mask] = np.sqrt(np.maximum(0.0, var_z))
    step_height_flat[valid_cell_mask] = z_max_flat[valid_cell_mask] - z_min_flat[valid_cell_mask]

    # Mặt đất z: chọn z_min để bắt đúng bề mặt địa hình dưới các bụi cây/ngọn cỏ
    elevation_flat = np.zeros(total_cells, dtype=np.float32)
    elevation_flat[valid_cell_mask] = z_min_flat[valid_cell_mask]

    elevation = elevation_flat.reshape(H, W)
    roughness = roughness_flat.reshape(H, W)
    step_height = step_height_flat.reshape(H, W)
    observed_mask = valid_cell_mask.reshape(H, W)

    # ==========================================================================
    # BẦU CHỌN NGỮ NGHĨA CHO CÁC Ô QUAN SÁT ĐƯỢC
    # ==========================================================================
    is_lethal_pt = np.isin(lbls, list(LETHAL_CLASSES))

    # Bầu chọn nhãn đa số cho mỗi ô có điểm
    packed = (cell_flat_idx.astype(np.int64) << 6) | (lbls & 63)
    u_packed, u_counts = np.unique(packed, return_counts=True)
    u_cell = u_packed >> 6
    u_lbl = (u_packed & 63).astype(np.uint16)

    order = np.lexsort((-u_counts, u_cell))
    u_cell_sorted = u_cell[order]
    u_lbl_sorted = u_lbl[order]
    _, first_idx = np.unique(u_cell_sorted, return_index=True)

    majority_label_flat = np.zeros(total_cells, dtype=np.uint16)
    majority_label_flat[u_cell_sorted[first_idx]] = u_lbl_sorted[first_idx]

    # Ưu tiên vật cản tuyệt đối: ô có điểm cản lập tức nhận nhãn cản
    if np.any(is_lethal_pt):
        lethal_pts_idx = cell_flat_idx[is_lethal_pt]
        lethal_lbls = lbls[is_lethal_pt]
        majority_label_flat[lethal_pts_idx] = lethal_lbls

    semantic_grid = majority_label_flat.reshape(H, W)

    # 4. Xử lý vùng mù cảm biến dưới và sát thân xe (Sensor Blind Spot: R < 3.5m)
    # Vì cảm biến Ouster đặt trên cao có góc quét tối thiểu -16.6 độ nên sát xe không có tia LiDAR
    Y_grid, X_grid = np.ogrid[:H, :W]
    r_center = int(round((cfg.X_MAX - 0.0) / cfg.RES))
    c_center = int(round((cfg.Y_MAX - 0.0) / cfg.RES))
    dist_from_sensor = np.sqrt(((Y_grid - r_center) * cfg.RES)**2 + ((X_grid - c_center) * cfg.RES)**2)

    # Ước lượng cao độ mặt đường tiếp xúc bánh xe từ các điểm quét gần nhất (bán kính 3m - 6m)
    near_ground_mask = observed_mask & (dist_from_sensor >= 3.0) & (dist_from_sensor <= 6.5)
    if np.any(near_ground_mask):
        estimated_ground_z = np.median(elevation[near_ground_mask])
        ground_class_candidates = semantic_grid[near_ground_mask]
        ground_class = np.bincount(ground_class_candidates).argmax()
    else:
        estimated_ground_z = 0.0
        ground_class = 23  # Concrete / Road

    # Gán cao độ và nhãn đường đi cho vùng mù sát xe
    blind_spot = (~observed_mask) & (dist_from_sensor < 3.5)
    elevation[blind_spot] = estimated_ground_z

    # 5. Nội suy lấp đầy các ô trống giữa các vòng quét LiDAR (Inpainting)
    elevation_filled = elevation.copy()
    valid_elev_mask = observed_mask | blind_spot
    if np.any(~valid_elev_mask) and np.any(valid_elev_mask):
        indices = distance_transform_edt(~valid_elev_mask, return_distances=False, return_indices=True)
        elevation_filled = elevation[indices[0], indices[1]]

    # 6. Làm mịn bề mặt độ cao để loại bỏ nhiễu bậc thang trước khi tính đạo hàm
    elevation_smooth = gaussian_filter(elevation_filled, sigma=1.2)

    # Tính độ dốc (Slope) từ Gradient độ cao
    grad_r, grad_c = np.gradient(elevation_smooth, cfg.RES, cfg.RES)
    slope_rad = np.arctan(np.sqrt(grad_r**2 + grad_c**2))

    # Nội suy ngữ nghĩa mặt đất cho các ô trống giữa các tia quét
    semantic_filled = semantic_grid.copy()
    semantic_filled[blind_spot] = ground_class
    if np.any(~valid_elev_mask) and np.any(valid_elev_mask):
        indices = distance_transform_edt(~valid_elev_mask, return_distances=False, return_indices=True)
        semantic_filled = semantic_grid[indices[0], indices[1]]

    return elevation_filled, slope_rad, roughness, step_height, semantic_filled, observed_mask


# ==============================================================================
# BƯỚC 3: TÍCH HỢP NGỮ NGHĨA VÀ TÍNH TOÁN COSTMAP
# ==============================================================================
def compute_costmap(elevation, slope_rad, roughness, step_height, semantic_grid, observed_mask, cfg: RellisConfig):
    """
    Bước 3: Hợp nhất Hình học + Ngữ nghĩa tạo Bản đồ Chi phí hoàn chỉnh:
      - Chi phí ngữ nghĩa: dựa trên độ bám và độ nguy hiểm (dirt, grass, puddle, mud, bush)
      - Chi phí hình học: độ dốc (slope), độ gồ ghề (roughness)
      - Nhận diện vật cản tuyệt đối: tree, pole, vehicle, person ngoài xe, slope > 30 độ, step > 0.35m
      - Mở rộng vùng cản an toàn (Obstacle Inflation)
      - Đảm bảo vùng thân xe xuất phát luôn khả hành
    """
    H, W = cfg.GRID_H, cfg.GRID_W

    # 1. Chi phí Ngữ nghĩa (Semantic Cost: 0 - 255)
    semantic_cost = np.zeros((H, W), dtype=np.float32)
    for cls_id, base_cost in SEMANTIC_COST_TABLE.items():
        semantic_cost[semantic_grid == cls_id] = base_cost

    # 2. Chi phí Độ dốc (Slope Cost: 0 - 255)
    max_slope_rad = np.radians(cfg.MAX_SLOPE_DEG)
    slope_norm = np.clip(slope_rad / max_slope_rad, 0.0, 1.0)
    slope_cost = slope_norm * 255.0

    # 3. Chi phí Độ gồ ghề (Roughness Cost: 0 - 255)
    rough_norm = np.clip(roughness / cfg.MAX_ROUGHNESS, 0.0, 1.0)
    roughness_cost = rough_norm * 255.0

    # 4. Hợp nhất Chi phí tổng thể
    costmap = (
        cfg.W_SEM * semantic_cost +
        cfg.W_SLOPE * slope_cost +
        cfg.W_ROUGH * roughness_cost
    )

    # 5. Nhận diện vật cản tuyệt đối (Lethal Mask)
    lethal_mask = np.isin(semantic_grid, list(LETHAL_CLASSES))
    # Chỉ áp dụng kiểm tra dốc/bước cản đứng trên các ô thực sự có dữ liệu LiDAR
    lethal_mask |= (observed_mask & (slope_rad >= max_slope_rad))
    lethal_mask |= (observed_mask & (step_height >= cfg.MAX_STEP_HEIGHT))

    # 6. Đảm bảo vùng thân xe hiện tại (Vehicle Footprint) không bị coi là vật cản
    Y, X = np.ogrid[:H, :W]
    r_center = int(round((cfg.X_MAX - 0.5) / cfg.RES))
    c_center = int(round((cfg.Y_MAX - 0.0) / cfg.RES))
    vehicle_footprint = (
        (np.abs((Y - r_center) * cfg.RES) <= 2.2) &
        (np.abs((X - c_center) * cfg.RES) <= 1.4)
    )
    lethal_mask[vehicle_footprint] = False

    # 7. Mở rộng an toàn quanh vật cản (Inflation Layer)
    if cfg.INFLATION_CELLS > 0:
        struct = generate_binary_structure(2, 1)
        inflated_lethal = binary_dilation(lethal_mask, structure=struct, iterations=cfg.INFLATION_CELLS)
        inflated_lethal[vehicle_footprint] = False
        near_obstacle = inflated_lethal & (~lethal_mask)
        costmap[near_obstacle] = np.maximum(costmap[near_obstacle], 220.0)

    costmap[lethal_mask] = 255.0
    costmap[vehicle_footprint] = np.minimum(costmap[vehicle_footprint], 25.0)
    costmap_uint8 = np.clip(costmap, 0, 255).astype(np.uint8)

    # Tạo bản đồ chỉ hình học (Geometry Only) để so sánh đối chứng
    geom_cost = (0.6 * slope_cost + 0.4 * roughness_cost)
    geom_lethal = (observed_mask & (slope_rad >= max_slope_rad)) | (observed_mask & (step_height >= cfg.MAX_STEP_HEIGHT))
    geom_lethal[vehicle_footprint] = False
    if cfg.INFLATION_CELLS > 0:
        inflated_geom = binary_dilation(geom_lethal, structure=struct, iterations=cfg.INFLATION_CELLS)
        inflated_geom[vehicle_footprint] = False
        geom_cost[inflated_geom & (~geom_lethal)] = np.maximum(geom_cost[inflated_geom & (~geom_lethal)], 220.0)
    geom_cost[geom_lethal] = 255.0
    geom_cost[vehicle_footprint] = np.minimum(geom_cost[vehicle_footprint], 25.0)
    costmap_geom_only = np.clip(geom_cost, 0, 255).astype(np.uint8)

    return costmap_uint8, costmap_geom_only


# ==============================================================================
# THỬ NGHIỆM TÌM ĐƯỜNG A* (PATH PLANNING BENCHMARK)
# ==============================================================================
def find_free_cell(costmap, r, c, max_r=25, thresh=240):
    """Tìm ô di chuyển được gần vị trí (r, c) nhất"""
    H, W = costmap.shape
    if costmap[r, c] < thresh:
        return (r, c)
    for rad in range(1, max_r + 1):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and costmap[nr, nc] < thresh:
                    return (nr, nc)
    return (r, c)


def astar_search(costmap, start_rc, goal_rc, max_traversable_cost=240):
    """
    Thuật toán A* tối ưu tìm đường trên ma trận chi phí 2.5D:
      - 8 hướng di chuyển (ngang, dọc, chéo)
      - Chi phí bước = khoảng cách hình học * (1.0 + cost/255 * 8.0)
    """
    H, W = costmap.shape
    sr, sc = find_free_cell(costmap, start_rc[0], start_rc[1], thresh=max_traversable_cost)
    gr, gc = find_free_cell(costmap, goal_rc[0], goal_rc[1], thresh=max_traversable_cost)

    def heuristic(r1, c1, r2, c2):
        dr = abs(r1 - r2)
        dc = abs(c1 - c2)
        return 1.0 * (dr + dc) + (1.414 - 2.0) * min(dr, dc)

    moves = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
    ]

    open_set = []
    heapq.heappush(open_set, (0.0, 0, sr, sc))

    came_from = {}
    g_score = {(sr, sc): 0.0}
    visited = set()
    counter = 1

    while open_set:
        _, _, curr_r, curr_c = heapq.heappop(open_set)

        if (curr_r, curr_c) == (gr, gc):
            path = []
            cur = (gr, gc)
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append((sr, sc))
            path.reverse()
            return path

        if (curr_r, curr_c) in visited:
            continue
        visited.add((curr_r, curr_c))

        for dr, dc, base_dist in moves:
            nr = curr_r + dr
            nc = curr_c + dc

            if not (0 <= nr < H and 0 <= nc < W):
                continue

            cell_cost = costmap[nr, nc]
            if cell_cost >= max_traversable_cost:
                continue

            terrain_penalty = 1.0 + (float(cell_cost) / 255.0) * 8.0
            step_cost = base_dist * terrain_penalty
            tentative_g = g_score[(curr_r, curr_c)] + step_cost

            if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                came_from[(nr, nc)] = (curr_r, curr_c)
                g_score[(nr, nc)] = tentative_g
                f_score = tentative_g + heuristic(nr, nc, gr, gc)
                heapq.heappush(open_set, (f_score, counter, nr, nc))
                counter += 1

    return None


# ==============================================================================
# BƯỚC 4: TRỰC QUAN HÓA TOÀN BỘ KẾT QUẢ (DASHBOARD 6 ĐỒ THỊ)
# ==============================================================================
def create_semantic_rgb_image(semantic_grid):
    """Chuyển đổi ma trận class ID sang ảnh màu RGB chuẩn RELLIS-3D"""
    H, W = semantic_grid.shape
    rgb_img = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, color in RELLIS_COLORS.items():
        mask = (semantic_grid == cls_id)
        rgb_img[mask] = color
    return rgb_img


_plot_lock = threading.Lock()

def visualize_dashboard(frame_id, pts_bev, elevation, slope_rad, roughness, 
                        semantic_grid, costmap, path_semantic, path_geom, 
                        start_rc, goal_rc, out_fig_path, cfg: RellisConfig):
    """Tạo Dashboard tổng hợp 6 đồ thị trực quan hóa phục vụ báo cáo khoa học (Thread-safe)"""
    with _plot_lock:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
        plt.subplots_adjust(wspace=0.25, hspace=0.30)
        fig.suptitle(f"RELLIS-3D to 2.5D Semantic Elevation Costmap & Path Planning - Frame [{frame_id}]", 
                     fontsize=17, fontweight='bold', y=0.98)

        extent = [cfg.Y_MAX, cfg.Y_MIN, cfg.X_MIN, cfg.X_MAX]

        # --- 1. Point Cloud BEV ---
        ax = axes[0, 0]
        sc = ax.scatter(pts_bev['y'], pts_bev['x'], c=pts_bev['z'], s=0.3, cmap='terrain', alpha=0.7)
        ax.set_title("1. Raw LiDAR Point Cloud (BEV)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Y (Trái -> Phải, m)")
        ax.set_ylabel("X (Sau -> Trước, m)")
        ax.set_xlim(cfg.Y_MAX, cfg.Y_MIN)
        ax.set_ylim(cfg.X_MIN, cfg.X_MAX)
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.colorbar(sc, ax=ax, label="Cao độ Z (m)", fraction=0.046, pad=0.04)

        # --- 2. Elevation Grid ---
        ax = axes[0, 1]
        im = ax.imshow(elevation, extent=extent, cmap='gist_earth', origin='upper')
        ax.set_title(f"2. Elevation Grid (Res: {cfg.RES}m)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("X (m)")
        plt.colorbar(im, ax=ax, label="Mặt đất z (m)", fraction=0.046, pad=0.04)

        # --- 3. Slope Map ---
        ax = axes[0, 2]
        slope_deg = np.degrees(slope_rad)
        im = ax.imshow(slope_deg, extent=extent, cmap='hot', vmin=0, vmax=35, origin='upper')
        ax.set_title("3. Terrain Slope Map", fontsize=13, fontweight='bold')
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("X (m)")
        plt.colorbar(im, ax=ax, label="Độ dốc (Độ)", fraction=0.046, pad=0.04)

        # --- 4. Roughness Map ---
        ax = axes[1, 0]
        im = ax.imshow(roughness, extent=extent, cmap='plasma', vmin=0, vmax=0.3, origin='upper')
        ax.set_title("4. Surface Roughness (Std Z)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("X (m)")
        plt.colorbar(im, ax=ax, label="Độ nhấp nhô (m)", fraction=0.046, pad=0.04)

        # --- 5. Semantic Grid ---
        ax = axes[1, 1]
        sem_rgb = create_semantic_rgb_image(semantic_grid)
        ax.imshow(sem_rgb, extent=extent, origin='upper')
        ax.set_title("5. 2.5D Semantic Grid", fontsize=13, fontweight='bold')
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("X (m)")
        
        present_classes = np.unique(semantic_grid)
        legend_patches = [
            mpatches.Patch(color=np.array(RELLIS_COLORS.get(c, (0,0,0)))/255.0, 
                           label=RELLIS_NAMES.get(c, f"ID {c}"))
            for c in present_classes if c in RELLIS_NAMES
        ]
        ax.legend(handles=legend_patches[:8], loc='lower right', fontsize=8, framealpha=0.7)

        # --- 6. Costmap & Path Planning ---
        ax = axes[1, 2]
        im = ax.imshow(costmap, extent=extent, cmap='YlOrRd', vmin=0, vmax=255, origin='upper')
        ax.set_title("6. Costmap & A* Path Planning", fontsize=13, fontweight='bold')
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("X (m)")
        plt.colorbar(im, ax=ax, label="Chi phí vượt địa hình (0-255)", fraction=0.046, pad=0.04)

        def grid_to_world(rc_path):
            if not rc_path:
                return [], []
            r = np.array([p[0] for p in rc_path])
            c = np.array([p[1] for p in rc_path])
            x_w = cfg.X_MAX - (r + 0.5) * cfg.RES
            y_w = cfg.Y_MAX - (c + 0.5) * cfg.RES
            return x_w, y_w

        if path_geom:
            gx_w, gy_w = grid_to_world(path_geom)
            ax.plot(gy_w, gx_w, color='blue', linestyle='--', linewidth=2.0, label='Chỉ Hình học (Geo-Only)')

        if path_semantic:
            sx_w, sy_w = grid_to_world(path_semantic)
            ax.plot(sy_w, sx_w, color='lime', linewidth=3.0, label='Hình học + Ngữ nghĩa (Full 2.5D)')

        sx0_w, sy0_w = grid_to_world([start_rc])
        gx0_w, gy0_w = grid_to_world([goal_rc])
        ax.scatter(sy0_w, sx0_w, color='cyan', s=130, edgecolors='black', zorder=5, label='Start (Vị trí xe)')
        ax.scatter(gy0_w, gx0_w, color='magenta', s=160, marker='*', edgecolors='black', zorder=5, label='Goal (Đích đến)')
        ax.legend(loc='lower left', fontsize=8, framealpha=0.85)

        plt.tight_layout()
        plt.savefig(out_fig_path)
        plt.close(fig)
    print(f"[Visualizer] Đã lưu Dashboard trực quan hóa tại: {out_fig_path}")


# ==============================================================================
# HÀM CHÍNH VẬN HÀNH TOÀN BỘ PIPELINE
# ==============================================================================
def process_single_frame(loader: RellisDataLoader, sequence_id: str, frame_id: str, 
                         out_dir: Path, cfg: RellisConfig, save_preview: bool = True,
                         skip_existing: bool = False, verbose: bool = True):
    """Xử lý trọn gói 1 frame: Đọc -> Lọc -> 2.5D Grid -> Costmap -> A* -> Lưu kết quả"""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_file = out_dir / f"{sequence_id}_{frame_id}_25d_dataset.npz"
    preview_file = out_dir / f"{sequence_id}_{frame_id}_dashboard.png"

    if skip_existing and npz_file.is_file():
        if not save_preview or preview_file.is_file():
            if verbose:
                print(f"[Bỏ qua] Frame {sequence_id}/{frame_id} đã tồn tại.")
            return True

    t_start = time.time()
    if verbose:
        print(f"\n========================================================")
        print(f">> BẮT ĐẦU XỬ LÝ FRAME: [{sequence_id} / {frame_id}]")
        print(f"========================================================")

    # 1. Nạp dữ liệu
    pts, lbls = loader.load_frame(sequence_id, frame_id)
    if verbose:
        print(f"[Bước 1] Nạp thành công: {len(pts):,} điểm LiDAR Ouster OS1 và {len(lbls):,} nhãn.")

    # Tiền xử lý
    x, y, z, intensity, lbls_filt = preprocess_point_cloud(pts, lbls, cfg)
    if verbose:
        print(f"[Bước 1] Sau khi lọc ROI & Thân xe: còn lại {len(x):,} điểm hợp lệ.")

    # 2. Xây dựng Lưới độ cao 2.5D
    t_grid0 = time.time()
    elevation, slope_rad, roughness, step_height, semantic_grid, observed_mask = build_elevation_grid(
        x, y, z, lbls_filt, cfg
    )
    t_grid1 = time.time()
    if verbose:
        print(f"[Bước 2] Dựng lưới 2.5D hoàn tất trong {(t_grid1 - t_grid0)*1000:.1f} ms. Kích thước: {elevation.shape}")

    # 3. Tích hợp Ngữ nghĩa & Tính toán Costmap
    t_cost0 = time.time()
    costmap, costmap_geom = compute_costmap(
        elevation, slope_rad, roughness, step_height, semantic_grid, observed_mask, cfg
    )
    t_cost1 = time.time()
    if verbose:
        print(f"[Bước 3] Tính Costmap đa tầng hoàn tất trong {(t_cost1 - t_cost0)*1000:.1f} ms.")

    # 4. Thử nghiệm Tìm đường A* (Benchmark Path Planning)
    start_r = int(round((cfg.X_MAX - 0.0) / cfg.RES))
    start_c = int(round((cfg.Y_MAX - 0.0) / cfg.RES))

    goal_r = int(round((cfg.X_MAX - 20.0) / cfg.RES))
    goal_c = int(round((cfg.Y_MAX - 2.0) / cfg.RES))

    path_semantic = astar_search(costmap, (start_r, start_c), (goal_r, goal_c))
    path_geom = astar_search(costmap_geom, (start_r, start_c), (goal_r, goal_c))

    if verbose:
        if path_semantic:
            cost_sum = sum(int(costmap[p[0], p[1]]) for p in path_semantic)
            print(f"[A* Planner] Tìm thấy đường đi Ngữ nghĩa + Hình học: {len(path_semantic)} bước (Tổng Cost: {cost_sum}).")
        else:
            print("[A* Planner] Không tìm thấy đường trên Costmap ngữ nghĩa.")

        if path_geom:
            cost_sum_geom = sum(int(costmap_geom[p[0], p[1]]) for p in path_geom)
            print(f"[A* Planner] Tìm thấy đường đi Chỉ Hình học: {len(path_geom)} bước (Tổng Cost: {cost_sum_geom}).")

    # 5. Lưu trữ tệp dữ liệu chuẩn .NPZ
    np.savez_compressed(
        npz_file,
        elevation=elevation,
        slope=slope_rad,
        roughness=roughness,
        step_height=step_height,
        semantic=semantic_grid,
        costmap=costmap,
        costmap_geom=costmap_geom,
        observed_mask=observed_mask,
        resolution=cfg.RES,
        x_min=cfg.X_MIN,
        x_max=cfg.X_MAX,
        y_min=cfg.Y_MIN,
        y_max=cfg.Y_MAX,
        path_semantic=np.array(path_semantic) if path_semantic else np.empty(0),
        path_geom=np.array(path_geom) if path_geom else np.empty(0)
    )
    if verbose:
        print(f"[Lưu trữ] Đã lưu bộ dữ liệu 2.5D chuẩn tại: {npz_file}")

    # 6. Xuất ảnh trực quan Dashboard nếu được yêu cầu
    if save_preview:
        pts_bev = {'x': x, 'y': y, 'z': z}
        visualize_dashboard(
            frame_id=f"{sequence_id}-{frame_id}",
            pts_bev=pts_bev,
            elevation=elevation,
            slope_rad=slope_rad,
            roughness=roughness,
            semantic_grid=semantic_grid,
            costmap=costmap,
            path_semantic=path_semantic,
            path_geom=path_geom,
            start_rc=(start_r, start_c),
            goal_rc=(goal_r, goal_c),
            out_fig_path=preview_file,
            cfg=cfg
        )

    t_total = time.time() - t_start
    if verbose:
        print(f">> HOÀN THÀNH FRAME {frame_id} TRONG {t_total:.2f} GIÂY!\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="RELLIS-3D to 2.5D Semantic Costmap Generator")
    parser.add_argument("--seq", type=str, default="00000", help="Sequence ID (mặc định: 00000)")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ tất cả các frame trong sequence")
    parser.add_argument("--frame", type=str, default="000000", help="Frame ID bắt đầu nếu không dùng --all (mặc định: 000000)")
    parser.add_argument("--num_frames", type=int, default=1, help="Số lượng frame cần xử lý nếu không dùng --all")
    parser.add_argument("--workers", type=int, default=6, help="Số luồng xử lý song song (mặc định: 6)")
    parser.add_argument("--preview_every", type=int, default=50, 
                        help="Khoảng cách frame xuất ảnh Dashboard (50: cứ 50 frame xuất 1 ảnh; 1: xuất toàn bộ; 0: tắt xuất ảnh để tăng tốc tối đa)")
    parser.add_argument("--skip_existing", action="store_true", help="Bỏ qua các frame đã có file .npz hoàn tất")
    parser.add_argument("--out", type=str, default="output_25d", help="Thư mục xuất kết quả (mặc định: output_25d)")
    args = parser.parse_args()

    cfg = RellisConfig()
    loader = RellisDataLoader(workspace_dir=".")
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    # Xác định danh sách frame cần chạy
    all_available_frames = loader.get_all_frame_ids(args.seq)
    if not all_available_frames:
        print(f"[LỖI] Không tìm thấy frame nào cho sequence {args.seq}!")
        return

    if args.all:
        frame_list = all_available_frames
    else:
        start_frame_int = int(args.frame)
        frame_list = [
            f"{start_frame_int + i:06d}"
            for i in range(args.num_frames)
            if f"{start_frame_int + i:06d}" in all_available_frames
        ]

    total_frames = len(frame_list)
    print(f"\n============================================================")
    print(f" BẮT ĐẦU XỬ LÝ {total_frames:,} FRAME TỪ SEQUENCE [{args.seq}]")
    print(f"============================================================")
    print(f"- Số luồng xử lý song song (Workers): {args.workers}")
    print(f"- Chu kỳ xuất ảnh Dashboard: mỗi {args.preview_every} frame (0 = tắt)")
    print(f"- Chế độ bỏ qua file đã có (--skip_existing): {args.skip_existing}")
    print(f"- Thư mục lưu kết quả: {out_path.resolve()}\n")

    if total_frames == 1:
        # Nếu chỉ chạy 1 frame thì in chi tiết
        process_single_frame(loader, args.seq, frame_list[0], out_path, cfg,
                             save_preview=(args.preview_every > 0),
                             skip_existing=args.skip_existing, verbose=True)
        return

    # Chạy song song nhiều frame
    from concurrent.futures import ThreadPoolExecutor, as_completed

    t0_total = time.time()
    success_count = 0
    fail_count = 0

    def task_exec(f_id):
        f_int = int(f_id)
        save_prev = (args.preview_every > 0) and (f_int % args.preview_every == 0)
        ok = process_single_frame(
            loader, args.seq, f_id, out_path, cfg,
            save_preview=save_prev,
            skip_existing=args.skip_existing,
            verbose=False
        )
        return ok, f_id

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(task_exec, f_id): f_id for f_id in frame_list}
        for future in as_completed(futures):
            f_id = futures[future]
            try:
                ok, _ = future.result()
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as exc:
                fail_count += 1
                print(f"[LỖI] Frame {f_id}: {exc}")

            done = success_count + fail_count
            elapsed = time.time() - t0_total
            fps = done / elapsed if elapsed > 0 else 0.0
            eta_sec = (total_frames - done) / fps if fps > 0 else 0.0

            if done % 25 == 0 or done == total_frames or done <= 5:
                pct = (done / total_frames) * 100
                eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s" if eta_sec < 3600 else f"{eta_sec/3600:.1f}h"
                print(f"[{done:4d}/{total_frames:4d} - {pct:5.1f}%] Frame {f_id} | Tốc độ: {fps:4.1f} FPS | Ước tính còn: {eta_str}")

    total_time = time.time() - t0_total
    print(f"\n============================================================")
    print(f" HOÀN TẤT XỬ LÝ TOÀN BỘ SEQUENCE {args.seq}!")
    print(f"============================================================")
    print(f"- Tổng số frame thành công: {success_count:,} / {total_frames:,}")
    if fail_count > 0:
        print(f"- Số frame lỗi: {fail_count}")
    print(f"- Tổng thời gian chạy: {int(total_time // 60)} phút {int(total_time % 60)} giây")
    print(f"- Tốc độ trung bình: {total_frames / total_time:.1f} FPS (~{total_time / total_frames * 1000:.1f} ms/frame)")
    print(f"- Toàn bộ dữ liệu 2.5D được lưu tại: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
