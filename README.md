# 🏥 Medical Named Entity Recognition — ViMedNER

Đồ án khóa luận tốt nghiệp — Nhận diện thực thể y tế tiếng Việt.

## Mô hình

**PhoBERT-Large + GATv2 (Window Graph) + Transformer Decoder + CRF**

- PhoBERT-Large làm encoder trích xuất đặc trưng ngữ nghĩa
- Window Graph + GATv2 mô hình hóa quan hệ cục bộ giữa các token
- Transformer Decoder tăng cường ngữ cảnh hai chiều
- CRF đảm bảo chuỗi nhãn BIO hợp lệ khi decode
- Class-weighted CE Loss để tăng độ nhạy với nhãn hiếm (CAUSE, ...)

## Nhãn

`DISEASE` · `SYMPTOM` · `DRUG` · `CAUSE` · `TREATMENT` · ...

## Môi trường

Toàn bộ pipeline được phát triển và huấn luyện trên **Google Colab** (GPU T4/A100).  
Dữ liệu và checkpoint lưu trên **Google Drive**, không cần cài đặt local.

```python
# Cài dependencies
!pip install torch-geometric pytorch-crf transformers seqeval \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')
```

## Cấu trúc
