"""
Hướng dẫn tải và truy xuất bộ dữ liệu RELLIS-3D 2.5D Semantic Costmap.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Đảm bảo in tiếng Việt trên console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def load_frame(npz_path: str):
    """
    Đọc một frame dữ liệu 2.5D (.npz) và trả về dictionary các ma trận.
    """
    data = np.load(npz_path)
    print(f"=== ĐÃ TẢI DỮ LIỆU: {Path(npz_path).name} ===")
    print(f"- Kích thước lưới: {data['elevation'].shape} (Cao x Rộng)")
    print(f"- Độ phân giải không gian: {float(data['resolution']):.2f} mét / ô")
    print(f"- Tọa độ X: [{float(data['x_min']):.1f}m đến {float(data['x_max']):.1f}m]")
    print(f"- Tọa độ Y: [{float(data['y_min']):.1f}m đến {float(data['y_max']):.1f}m]\n")
    
    print("Danh sách các trường dữ liệu:")
    for key in data.files:
        val = data[key]
        if isinstance(val, np.ndarray) and val.ndim > 0:
            print(f"  + {key:<15}: shape={str(val.shape):<10} dtype={str(val.dtype):<8} [min={val.min():.2f}, max={val.max():.2f}]")
        else:
            print(f"  + {key:<15}: giá trị = {val}")
    return data

def quick_plot(data, save_path="demo_output.png"):
    """Vẽ minh họa nhanh 4 kênh cốt lõi: Độ cao, Độ dốc, Ngữ nghĩa, Bản đồ chi phí"""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    # 1. Elevation
    im0 = axes[0].imshow(data['elevation'], cmap='terrain', origin='lower')
    axes[0].set_title("1. Elevation (Mặt đất z)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Slope (đổi từ radian sang độ)
    slope_deg = np.rad2deg(data['slope'])
    im1 = axes[1].imshow(slope_deg, cmap='hot', origin='lower')
    axes[1].set_title("2. Slope (Độ dốc °)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Semantic Grid
    im2 = axes[2].imshow(data['semantic'], cmap='tab20', origin='lower')
    axes[2].set_title("3. Semantic Grid")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Costmap + A* Path
    im3 = axes[3].imshow(data['costmap'], cmap='YlOrRd', origin='lower')
    path_sem = data['path_semantic']
    if len(path_sem) > 0:
        axes[3].plot(path_sem[:, 1], path_sem[:, 0], 'lime', linewidth=2, label="A* Path")
        axes[3].legend(loc="lower right")
    axes[3].set_title("4. Costmap & Path Planning")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n[Visualizer] Đã lưu ảnh minh họa nhanh tại: {save_path}")

if __name__ == "__main__":
    candidates = [
        "samples/00000_000500_25d_dataset.npz",
        "out_00000/00000_000500_25d_dataset.npz"
    ]
    sample_path = None
    for cand in candidates:
        if Path(cand).exists():
            sample_path = cand
            break

    if sample_path:
        data = load_frame(sample_path)
        quick_plot(data, save_path="demo_quickview.png")
    else:
        print(f"Không tìm thấy file mẫu tại các đường dẫn: {candidates}")

