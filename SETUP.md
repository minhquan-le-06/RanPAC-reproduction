# Setup – chạy local bằng venv (Windows, RTX 4060)

Môi trường ảo (virtualenv) đã có sẵn tại: `D:\Tai\ML\RanPAC\.venv` với:
- **Python**: 3.11
- **PyTorch**: 2.6.0+cu124 (CUDA khả dụng trên GPU RTX 4060 Laptop 8GB)
- **torchvision**, **timm** (1.0.31), **wandb** (0.31.0), **Pillow**, **numpy** đã cài đặt đầy đủ.

---

## 1. Đường dẫn Python venv

Mỗi khi chạy lệnh, bạn dùng trực tiếp Python của venv:
```powershell
$py = "D:\Tai\ML\RanPAC\.venv\Scripts\python.exe"
```
*(Hoặc kích hoạt venv bằng lệnh: `& D:\Tai\ML\RanPAC\.venv\Scripts\Activate.ps1`)*

---

## 2. Set W&B API Key

```powershell
# PowerShell (cho session hiện tại)
$env:WANDB_API_KEY = "your_wandb_api_key_here"

# Hoặc lưu cố định cho tài khoản Windows
[System.Environment]::SetEnvironmentVariable("WANDB_API_KEY", "your_wandb_api_key_here", "User")
```
Lấy API key tại: https://wandb.ai/settings → API keys

---

## 3. Chạy thực nghiệm RanPAC trên máy (tự đẩy lên W&B)

Mở PowerShell tại thư mục `d:\Tai\ML\RanPAC-reproduction\`:

```powershell
cd d:\Tai\ML\RanPAC-reproduction
$py = "D:\Tai\ML\RanPAC\.venv\Scripts\python.exe"

# 1. Preset rp-table-a5 (M=2500, λ=100) — [Table A5: mốc 87.7%] (Phần còn thiếu)
& $py run_local.py --preset rp-table-a5 --wandb_entity pnp123 --wandb_project ML --run_label local

# 2. Preset ncm (NCM Cosine) — [Table 1: mốc 83.4%]
& $py run_local.py --preset ncm --wandb_entity pnp123 --wandb_project ML --run_label local

# 3. Preset ridge (Ridge không RP) — [Table 1: mốc 87.0%]
& $py run_local.py --preset ridge --wandb_entity pnp123 --wandb_project ML --run_label local

# 4. Preset rp-table1 (Ridge + RP M=10000) — [Table 1: mốc 89.0%]
& $py run_local.py --preset rp-table1 --wandb_entity pnp123 --wandb_project ML --run_label local
```

> **Lưu ý**:
> - Trên Windows nếu DataLoader báo lỗi đa luồng, thêm `--workers 0`.
> - Tham số `--stats_device cuda` có thể tăng tốc tính ma trận cho `rp-table1`.

---

## Bảng đối chiếu kết quả Paper

| Preset | Phương pháp | Kết quả thực tế | Mốc Paper | Chênh lệch (Δ) | Trạng thái |
|--------|------------|-----------------|-----------|----------------|------------|
| `ncm` | NCM cosine | **83.38%** | 83.4% | −0.02 pp | Đã chạy (Colab) |
| `ridge` | Ridge (no RP) | **87.02%** | 87.0% | +0.02 pp | Đã chạy (Colab) |
| `rp-table1` | Ridge + RP (M=10000) | **88.75%** | 89.0% | −0.25 pp | Đã chạy (Colab) |
| `rp-table-a5` | Ridge + RP (M=2500) | **87.68%** | 87.7% | −0.02 pp | Đã chạy (Local RTX 4060) ✅ |
