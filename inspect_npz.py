#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script đọc và kiểm tra nhanh 1 file dữ liệu 2.5D .NPZ đã tạo
Cách dùng:
    python inspect_npz.py --file output_25d/00000_000000_25d_dataset.npz
"""

import sys
import argparse
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Kiểm tra thông tin file 2.5D NPZ")
    parser.add_argument("--file", type=str, default="output_25d/00000_000000_25d_dataset.npz", 
                        help="Đường dẫn đến file .npz")
    args = parser.parse_args()

    data = np.load(args.file)
    print("=" * 60)
    print(f">> THÔNG TIN FILE: {args.file}")
    print("=" * 60)
    
    print("\n[1. Danh sách các Layer trong tệp]:")
    for k in sorted(data.files):
        val = data[k]
        if isinstance(val, np.ndarray):
            print(f"  - {k:<15}: shape = {str(val.shape):<15} dtype = {str(val.dtype):<10} range = [{val.min():.2f}, {val.max():.2f}]")
        else:
            print(f"  - {k:<15}: {val}")

    print("\n[2. Thống kê địa hình]:")
    slope_deg = np.degrees(data['slope'])
    print(f"  - Độ dốc trung bình: {slope_deg.mean():.2f}° (max: {slope_deg.max():.2f}°)")
    print(f"  - Độ gồ ghề trung bình: {data['roughness'].mean():.3f} m (max: {data['roughness'].max():.3f} m)")
    print(f"  - Cao độ mặt đất z: [{data['elevation'].min():.2f} m -> {data['elevation'].max():.2f} m]")

    print("\n[3. Thống kê nhãn ngữ nghĩa (RELLIS-3D)]: ")
    unique_sem, counts = np.unique(data['semantic'], return_counts=True)
    total_cells = len(data['semantic'].ravel())
    for s_id, cnt in zip(unique_sem, counts):
        pct = (cnt / total_cells) * 100
        print(f"  - Class ID {s_id:2d}: {cnt:5d} ô ({pct:5.1f}%)")

    print("\n[4. Đường đi tìm kiếm (A*)]: ")
    p_sem = data['path_semantic']
    p_geo = data['path_geom']
    print(f"  - Chiều dài đường Semantic + Geometry: {len(p_sem)} bước")
    print(f"  - Chiều dài đường Geometry-Only     : {len(p_geo)} bước")
    print("=" * 60)

if __name__ == "__main__":
    main()
