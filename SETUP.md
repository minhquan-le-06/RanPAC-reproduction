# Setup RanPAC trên Colab

Chạy ablation Phase 2 trên Split CIFAR-100 với seed `1993` (10 task, mỗi task 10 lớp).

Cả ba notebook đã được chạy sẵn. Kết quả có trong project W&B `pnp123/ML`.

## Cài đặt

1. Copy `ranpac_core.py`, `ranpac_data.py`, `ranpac_colab.py` và `test_ranpac_core.py` từ thư mục gốc vào `MyDrive/RanPAC_implement`. Nếu đặt ở chỗ khác, sửa `SOURCE_DIR` trong notebook.
2. Mở notebook bằng Google Colab và chọn **Runtime → Change runtime type → GPU**. Hai cell setup cài `timm==1.0.22`, `wandb==0.31.0` và mount Drive. PyTorch và torchvision lấy từ runtime Colab. Khi cần trích embedding, chương trình tự tải CIFAR-100 và trọng số ViT.
3. Thêm `WANDB_API_KEY` vào **Colab Secrets** và bật quyền truy cập cho notebook. Cả ba notebook log online vào `pnp123/ML` (`WANDB_ENTITY = 'pnp123'`, `WANDB_PROJECT = 'ML'`). Dùng API key của tài khoản có quyền ghi vào project này. Nếu chạy mới trên project riêng, sửa `WANDB_ENTITY` và `WANDB_PROJECT` cho phù hợp.

## Cấu hình trong notebook

| Notebook | `PRESET` | `RESUME` |
| --- | --- | --- |
| [01_ncm_cosine.ipynb](01_ncm_cosine.ipynb) | `ncm` · NCM cosine | `False` |
| [02_ridge_no_rp.ipynb](02_ridge_no_rp.ipynb) | `ridge` · Ridge không random projection | `False` |
| [03_ridge_rp_relu.ipynb](03_ridge_rp_relu.ipynb) | `rp-table1` · RP + ReLU, `M=10000`, chọn λ theo task | `True` |

Cả ba đặt `STOP_AFTER_TASK = 10`, `RUN_LABEL = 'main'`, `BATCH_SIZE = 48`, `NUM_WORKERS = 2` và `STATS_DEVICE = 'cpu'`. GPU dùng để trích embedding. `RUN_WANDB_SMOKE = False` mặc định.

## Chạy mới hoặc resume

- **Chạy lại từ đầu** · Đặt `RESUME = False` và đổi `RUN_LABEL` từ `main` sang tên chưa dùng cho preset đó, ví dụ `rerun-01`, để tránh trùng thư mục checkpoint và tên run trên W&B.
- **Resume** · Giữ `RESUME = True` và dùng đúng `WORK_DIR`, `RUN_LABEL`, preset, mã nguồn, cấu hình và project W&B của run cũ. Với cấu hình hiện tại của notebook `03`, checkpoint cần có ở `MyDrive/ranpac-runs/runs/rp-table1-s1993-main/checkpoint.pt`.
- Chạy các cell theo thứ tự. Để kiểm tra W&B, đặt `RUN_WANDB_SMOKE = True` rồi xác nhận `smoke/ok = 1` ở URL in ra. Nếu chỉ muốn thử một task khi chạy mới, đặt `STOP_AFTER_TASK = 1`. Sau đó bật `RESUME = True` và tăng `STOP_AFTER_TASK` để chạy tiếp.

## Kết quả

Mặc định Drive lưu run tại `MyDrive/ranpac-runs/runs/<preset>-s1993-<run_label>/`. Trong đó có `checkpoint.pt` để resume, `config.json`, `metrics.csv` với `A[t]` và `accuracy_matrix.csv` với `R[t,i]`. Cache embedding dùng chung nằm ở `MyDrive/ranpac-runs/embeddings/`.

## Algorithm 1 đầy đủ

Mở [04_ranpac_full.ipynb](04_ranpac_full.ipynb) và thêm `ranpac_full.py` vào `SOURCE_DIR` cùng các module trên. Notebook train AdaptFormer trên task 1, sau đó đóng băng backbone đã thích nghi và chạy Phase 2 từ task 1. Cấu hình mặc định train 20 epoch và chạy đủ 10 task với `M=10000`. Đặt `STOP_AFTER_TASK = 2` để chạy pilot Phase 2, rồi tăng mốc dừng lên 10 và bật `RESUME` để tiếp tục.

Run đầy đủ lưu ở `MyDrive/ranpac-runs/runs/full-s1993-<run_label>/` và có thêm `phase1_metrics.csv`. Notebook ghi cách tiếp tục từ checkpoint theo epoch, tùy chọn giảm VRAM và các khác biệt so với cấu hình của tác giả.

## Sweep M trên CIFAR-100

Mở [05_ranpac_m_sweep.ipynb](05_ranpac_m_sweep.ipynb) và chép ba file `ranpac_core.py`, `ranpac_data.py` và `ranpac_colab.py` từ thư mục gốc vào `SOURCE_DIR`. Nếu đã tạo cache từ notebook 03, dùng cùng `WORK_DIR` và phiên bản thư viện để tái sử dụng embedding. Notebook 05 chạy lần lượt sáu mức `M = 300, 500, 1000, 1250, 2500, 10000` và xuất bảng so sánh accuracy, thời gian và RAM.

## Algorithm 1 trên ImageNet-R

Mở [06_ranpac_full_imagenetr.ipynb](06_ranpac_full_imagenetr.ipynb) trong Colab với GPU runtime. Đặt các tệp theo bảng dưới đây hoặc sửa đường dẫn tương ứng trong cell mount Drive. Thư mục mã nguồn dùng chung với notebook 04.

| Đường dẫn mặc định | Biến trong notebook |
| --- | --- |
| `MyDrive/RanPAC_implement/RanPAC-reimplement-imagenetr.zip` | `SOURCE_ZIP` |
| `MyDrive/imagenet-r.zip` với cùng split `train/` và `test/` như notebook source gốc. Notebook tự tải nếu thiếu. | `DATASET_INPUT` |
| `MyDrive/ranpac-runs/` cho checkpoint và kết quả | `WORK_DIR` |

W&B mặc định log online vào `pnp123/ML` như notebook 04. Bật quyền đọc Colab Secret `WANDB_API_KEY` cho notebook. Để thử một epoch, đặt `STOP_AFTER_EPOCH = 1` và giữ `PHASE1_EPOCHS = 20`. Sau đó đặt `RESUME = True`, `STOP_AFTER_EPOCH = None` và giữ nguyên nhãn run, chế độ W&B cùng cấu hình huấn luyện.

Cell cuối đọc kết quả preset 7 của notebook source gốc trong `MyDrive/RanPAC-original-imagenetr-results/` để tạo bảng và biểu đồ so sánh. Chạy cell này sau khi cả hai run đã hoàn tất đủ 10 task.
