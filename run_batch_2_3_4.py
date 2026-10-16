"""
Batch runner to process sequences 00002, 00003, 00004 sequentially.
"""
import sys
import os
import subprocess
import time
from pathlib import Path
from PIL import Image

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
os.environ["PYTHONIOENCODING"] = "utf-8"

SEQUENCES = ["00002", "00003", "00004"]
WORKERS = 8
PREVIEW_EVERY = 50

def verify_output(out_dir: Path):
    npz_files = list(out_dir.glob("*.npz"))
    png_files = list(out_dir.glob("*.png"))
    corrupted_png = []
    for p in png_files:
        if p.stat().st_size < 100 * 1024:
            corrupted_png.append(str(p))
        else:
            try:
                with Image.open(p) as img:
                    img.verify()
            except Exception:
                corrupted_png.append(str(p))
    return len(npz_files), len(png_files), corrupted_png

def main():
    t_global_start = time.time()
    results = {}

    print(f"\n============================================================")
    print(f" BAT DAU BATCH PROCESS CHO CAC SEQUENCES: {', '.join(SEQUENCES)}")
    print(f" So luong (Workers): {WORKERS} | Preview every: {PREVIEW_EVERY}")
    print(f"============================================================\n", flush=True)

    for idx, seq in enumerate(SEQUENCES, 1):
        out_folder = Path(f"out_{seq}")
        out_folder.mkdir(parents=True, exist_ok=True)
        t_seq_start = time.time()

        print(f"\n>>> [{idx}/{len(SEQUENCES)}] DANG CHAY SEQUENCE {seq} VAO THU MUC {out_folder}...", flush=True)
        cmd = [
            sys.executable,
            "-u",
            "process_rellis_25d.py",
            "--seq", seq,
            "--all",
            "--workers", str(WORKERS),
            "--preview_every", str(PREVIEW_EVERY),
            "--out", str(out_folder)
        ]
        
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"[CANH BAO] Sequence {seq} ket thuc voi exit code {proc.returncode}", flush=True)

        t_seq_elapsed = time.time() - t_seq_start
        npz_cnt, png_cnt, corrupted = verify_output(out_folder)
        results[seq] = {
            "elapsed_sec": t_seq_elapsed,
            "npz": npz_cnt,
            "png": png_cnt,
            "corrupted": len(corrupted)
        }
        print(f">>> [HOAN THANH SEQUENCE {seq}] NPZ: {npz_cnt} | PNG: {png_cnt} | Loi anh: {len(corrupted)} | Thoi gian: {int(t_seq_elapsed//60)}m {int(t_seq_elapsed%60):02d}s\n", flush=True)

    t_global_elapsed = time.time() - t_global_start
    print("\n" + "=" * 60)
    print(" BAO CAO TONG KET TOAN BO 3 SEQUENCES (00002, 00003, 00004)")
    print("=" * 60)
    for seq, r in results.items():
        print(f"Sequence {seq} -> out_{seq}: {r['npz']} .npz, {r['png']} .png, {r['corrupted']} corrupted | {int(r['elapsed_sec']//60)}m {int(r['elapsed_sec']%60):02d}s")
    print(f"Tong thoi gian toan bo: {int(t_global_elapsed//60)} phut {int(t_global_elapsed%60):02d} giay")
    print("=" * 60 + "\n", flush=True)

if __name__ == "__main__":
    main()
