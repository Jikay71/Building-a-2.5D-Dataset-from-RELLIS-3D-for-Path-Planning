"""
Tạo ảnh GIF động từ chuỗi Dashboard của sequence 00000 để thuyết trình demo.
"""
import sys
import glob
from pathlib import Path
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def create_gif(png_dir="out_00000", output_gif="demo_sequence_00000.gif", max_width=1200, fps=5):
    png_files = sorted(glob.glob(f"{png_dir}/*_dashboard.png"))
    if not png_files:
        print(f"Không tìm thấy ảnh dashboard trong {png_dir}")
        return

    print(f"Tìm thấy {len(png_files)} ảnh dashboard. Đang xử lý và ghép GIF...")
    frames = []
    
    for i, p in enumerate(png_files):
        with Image.open(p) as img:
            # Resize để giảm dung lượng file GIF và mượt mà khi trình chiếu
            w, h = img.size
            if w > max_width:
                new_h = int(h * (max_width / w))
                img_resized = img.resize((max_width, new_h), Image.Resampling.LANCZOS)
            else:
                img_resized = img.copy()
            # Convert to adaptive 256 color palette for compact GIF
            img_p = img_resized.convert("P", palette=Image.Palette.ADAPTIVE)
            frames.append(img_p)

    duration_ms = int(1000 / fps)
    print(f"Đang xuất file GIF: {output_gif} (Thời gian mỗi frame: {duration_ms}ms)...")
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True
    )
    file_size_mb = Path(output_gif).stat().st_size / (1024 * 1024)
    print(f"HOÀN TẤT! Đã tạo file GIF tại: {output_gif} (Dung lượng: {file_size_mb:.2f} MB)")

if __name__ == "__main__":
    create_gif(png_dir="out_00000", output_gif="demo_sequence_00000.gif", max_width=1200, fps=4)
