# RanPAC Reproduction

Tái hiện thực nghiệm bài báo **"RanPAC: Random Projections and Pre-trained Models for Continual Learning"** (NeurIPS 2023) trên benchmark chuẩn Split CIFAR-100.

- Paper gốc: [arXiv:2307.02251](https://arxiv.org/abs/2307.02251) (NeurIPS 2023)

---

## 1. Giới thiệu phương pháp RanPAC

RanPAC giải quyết bài toán Class-Incremental Learning (CIL) với mô hình backbone được đóng băng hoàn toàn (frozen pre-trained ViT-B/16):
1. **Trích xuất đặc trưng (Feature Extraction)**: Ảnh đầu vào đi qua backbone ViT-B/16 (ImageNet-21K) cố định để lấy vector đặc trưng $F \in \mathbb{R}^{D}$ ($D=768$).
2. **Chiếu ngẫu nhiên & Kích hoạt phi tuyến (Random Projection + ReLU)**:
   $$H = \text{ReLU}(F \cdot W)$$
   với ma trận chiếu ngẫu nhiên cố định $W \in \mathbb{R}^{D \times M}$ sinh từ phân phối Gaussian $\mathcal{N}(0, 1)$, mở rộng số chiều lên $M$ ($M=2500$ hoặc $M=10000$).
3. **Cộng dồn Ridge Regression**: Khi nhận dữ liệu các task theo thời gian, mô hình tích lũy ma trận Gram $G$ và ma trận tương quan nhãn $C$:
   $$G \leftarrow G + H^T H, \quad C \leftarrow C + H^T Y$$
   Trọng số phân loại tối ưu được giải dưới dạng nghiệm đóng:
   $$W_{\text{out}} = (G + \lambda I)^{-1} C$$
   Phương pháp này không cần fine-tune backbone, không lưu lại ảnh cũ (exemplar-free), tính toán cực nhanh và hạn chế tối đa hiện tượng quên kiến thức cũ (Catastrophic Forgetting).

---

## 2. Kết quả tái hiện (Reproduced Results)

Benchmark Split CIFAR-100 gồm **10 task liên tiếp**, mỗi task 10 lớp ngẫu nhiên (seed `1993`). Đánh giá độ chính xác trung bình trên toàn bộ các lớp đã học $A[t]$:

| # | Phương pháp | Thiết lập | File chạy | Kết quả tái hiện ($A_{10}$) | Mốc trong Paper ($A_{10}$) | Chênh lệch ($\Delta$) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| 1 | **NCM Cosine** | Frozen ViT + Nearest Class Mean Cosine | `01_ncm_cosine.ipynb` | **83.38%** | 83.4% (Table 1) | −0.02 pp |
| 2 | **Ridge (no RP)** | Frozen ViT + Ridge Regression không qua RP | `02_ridge_no_rp.ipynb` | **87.02%** | 87.0% (Table 1) | +0.02 pp |
| 3 | **RanPAC (Table 1)** | $M=10000$ + ReLU + Task CV $\lambda$ | `03_ridge_rp_relu.ipynb` | **88.75%** | 89.0% (Table 1) | −0.25 pp |
| 4 | **RanPAC (Table A5)** | $M=2500$ + ReLU + Cố định $\lambda=100$ | `run_local.py` (`rp-table-a5`) | **87.68%** | 87.7% (Table A5) | −0.02 pp |

---

## 3. Biểu đồ so sánh & Đánh giá Forgetting

### 3.1. So sánh đường cong độ chính xác qua 10 Task
Độ chính xác trung bình trên các lớp đã học qua từng task của 4 phương pháp:

![Accuracy Comparison](accuracy_comparison.png)

### 3.2. Ma trận Forgetting $R[t, i]$ (Heatmap)
Độ chính xác kiểm tra trên task $i$ sau khi đã học xong task $t$ (với cấu hình RanPAC Table A5):

![Forgetting Heatmap](forgetting_heatmap.png)

> **Nhận xét**: Sau khi học tới Task 10, độ chính xác trên Task 1 vẫn giữ ở mức **90.4%** (so với 96.8% lúc mới học xong Task 1). Điều này chứng minh RanPAC duy trì trí nhớ đối với các lớp đã học rất tốt mà không cần lưu lại dữ liệu mẫu (rehearsal-free).

---

## 4. Hướng dẫn cài đặt & Chạy thực nghiệm

### 4.1. Cài đặt môi trường

Yêu cầu Python 3.10+ và GPU hỗ trợ CUDA.

```powershell
# 1. Cài đặt PyTorch với CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 4.2. Chạy trên Google Colab
Chạy tuần tự các notebook trong repo:
- `01_ncm_cosine.ipynb` (NCM Cosine)
- `02_ridge_no_rp.ipynb` (Ridge không RP)
- `03_ridge_rp_relu.ipynb` (RanPAC Table 1)

### 4.3. Chạy Local (Windows / Linux)
Chạy trực tiếp thông qua script `run_local.py`:

```powershell
# Cấu hình RanPAC Table A5 (M=2500, λ=100)
python run_local.py --preset rp-table-a5 --wandb_entity <entity> --wandb_project <project> --wandb_mode offline

# Cấu hình NCM
python run_local.py --preset ncm --wandb_entity <entity> --wandb_project <project> --wandb_mode offline

# Cấu hình Ridge không RP
python run_local.py --preset ridge --wandb_entity <entity> --wandb_project <project> --wandb_mode offline

# Cấu hình RanPAC Table 1 (M=10000)
python run_local.py --preset rp-table1 --wandb_entity <entity> --wandb_project <project> --wandb_mode offline
```

### 4.4. Xuất đồ thị phân tích
```powershell
# Vẽ biểu đồ so sánh accuracy 4 phương pháp
python plot_comparison.py

# Vẽ heatmap ma trận forgetting
python plot_heatmap.py
```

---

## 5. Dataset & Pretrained Backbone

- **Dataset**: CIFAR-100 (tự động tải và phân chia thành 10 task với seed `1993`).
- **Backbone**: `vit_base_patch16_224.augreg_in21k` (tải tự động qua thư viện `timm`).

---

## 6. Trích dẫn

```bibtex
@inproceedings{mcdonnell2023ranpac,
  title={RanPAC: Random Projections and Pre-trained Models for Continual Learning},
  author={McDonnell, Mark D and Gong, Dong and Parvaneh, Amin and Ehsan, Mohammad Reza and Abbasnejad, Ehsan and van den Hengel, Anton},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```
