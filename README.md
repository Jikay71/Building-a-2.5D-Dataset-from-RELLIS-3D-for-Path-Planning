# RELLIS-3D to 2.5D Semantic Elevation Costmap & Path Planning Benchmark

Dự án nghiên cứu và xây dựng bộ dữ liệu **2.5D Semantic Elevation Costmap** từ tập dữ liệu off-road thực tế **RELLIS-3D** phục vụ bài toán quy hoạch quỹ đạo và tự hành xe bánh xích/bánh lốp (UGV).

---

## 📌 1. Kết Quả Nghiệm Thu Đạt Được

- **Quy mô dữ liệu đã sinh:** **13,556 frames** dữ liệu chuẩn `.npz` nén trên 5 Sequence (`00000` đến `00004`).
- **Dung lượng tổng thể:** ~3.26 GB.
- **Trực quan hóa:** **275 ảnh Dashboard** trực quan chi tiết 6 góc nhìn (Raw LiDAR, Elevation, Slope, Roughness, Semantic Grid, Costmap & A* Path).
- **Thực nghiệm A* Planning:**
  - Tỉ lệ thành công: **100%**.
  - Mức giảm thiểu rủi ro địa hình so với chỉ dùng độ cao hình học: **29.24%**.
  - Né tránh được hơn **9,000 điểm rủi ro cao** (hố bùn, bụi rậm, đá lổn nhổn).

---

## 📁 2. Cấu Trúc Thư Mục

```text
Kiem Thu/
│
├── out_00000/                # 2,847 frame .npz + 59 ảnh preview .png (Sequence 00000)
├── out_00001/                # 2,319 frame .npz + 47 ảnh preview .png (Sequence 00001)
├── out_00002/                # 4,147 frame .npz + 83 ảnh preview .png (Sequence 00002)
├── out_00003/                # 2,184 frame .npz + 44 ảnh preview .png (Sequence 00003)
├── out_00004/                # 2,059 frame .npz + 42 ảnh preview .png (Sequence 00004)
│
├── demo_sequence_00000.gif   # Clip ảnh động demo xe chạy thời gian thực cho Slide
├── demo_quickview.png        # Ảnh kết quả demo 4 kênh
├── demo_load_dataset.py      # Script Python mẫu hướng dẫn load và truy xuất file .npz
├── process_rellis_25d.py     # Script pipeline cốt lõi (3 giai đoạn)
├── run_batch_2_3_4.py        # Script batch runner xử lý hàng loạt
├── evaluate_metrics.py       # Script tính toán các chỉ số định lượng đánh giá thuật toán
└── README.md                 # Tài liệu hướng dẫn đồ án
```

---

## 🚀 3. Hướng Dẫn Sử Dụng Nhanh

### 3.1. Đọc và sử dụng 1 file dữ liệu 2.5D (.npz)
Chỉ cần chạy lệnh sau:
```bash
python demo_load_dataset.py
```
Mỗi file `.npz` bao gồm:
* `elevation`: Ma trận độ cao mặt đất $(200 \times 150)$ (mét).
* `slope`: Ma trận độ dốc địa hình (radian).
* `roughness`: Độ gồ ghề bề mặt địa hình ($\sigma_z$).
* `step_height`: Bước chênh độ cao thẳng đứng giữa các ô lân cận.
* `semantic`: Ma trận nhãn lớp vật thể/địa hình.
* `costmap`: Ma trận chi phí di chuyển tích hợp ($0 \to 255$).
* `costmap_geom`: Ma trận chi phí chỉ dựa trên hình học.
* `path_semantic` & `path_geom`: Tọa độ đường đi tìm được bằng thuật toán A*.

### 3.2. Chạy tái hiện tính toán các chỉ số định lượng:
```bash
python evaluate_metrics.py
```

### 3.3. Xử lý thêm một Sequence mới bất kỳ:
```bash
python process_rellis_25d.py --seq 00005 --all --workers 8 --preview_every 50 --out out_00005
```
