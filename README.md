# Chú thích ảnh tiếng Việt (Vietnamese Image Captioning)

Dự án sinh **một câu mô tả tiếng Việt** cho ảnh RGB đầu vào, kết hợp kiến thức thị giác máy tính (CNN, CLIP) và xử lý ngôn ngữ tự nhiên (LSTM, attention, Transformer). Dự án so sánh hai kiến trúc decoder và mở rộng thêm chức năng **tìm ảnh bằng câu truy vấn tiếng Việt**.


## Mục lục

- [Tính năng](#tính-năng)
- [Dữ liệu](#dữ-liệu)
- [Kiến trúc mô hình](#kiến-trúc-mô-hình)
- [Kết quả](#kết-quả)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt và chạy](#cài-đặt-và-chạy)
- [Hạn chế và hướng phát triển](#hạn-chế-và-hướng-phát-triển)

## Tính năng

- **Tiền xử lý tiếng Việt:** chuẩn hóa Unicode (NFC), chuyển chữ thường, tách từ bằng PyVi, xây từ điển với các token đặc biệt `<pad>`, `<unk>`, `<start>`, `<end>`.
- **Mô hình 1 – CNN + LSTM có Attention:** encoder ResNet-50, decoder LSTM với additive attention.
- **Mô hình 2 – CNN + Transformer:** encoder ResNet-50, Transformer encoder–decoder.
- **Đánh giá:** BLEU-1 đến BLEU-4 và CIDEr trên tập test.
- **Mở rộng – Tìm ảnh bằng tiếng Việt:** dùng CLIP đa ngôn ngữ, đánh giá bằng Recall@K và MRR.

## Dữ liệu

Dự án dùng bộ **UIT-ViIC** (caption tiếng Việt cho ảnh theo định dạng MS-COCO). Ảnh được tải từ `coco_url` có trong file annotation.

| Tập | Số ảnh | Số caption |
|---|---|---|
| Train | 2.695 | 13.481 |
| Validation | 924 | 4.620 |
| Test | 231 | 1.155 |
| **Tổng** | **3.850** | **19.256** |

Mỗi ảnh có đúng 5 caption tham chiếu. Từ điển xây từ tập train có **1.317 token** (`min_freq=1`).

## Kiến trúc mô hình

### Tiền xử lý ảnh

Resize về 224×224, chuẩn hóa theo thống kê ImageNet.

### CNN Encoder (dùng chung)

ResNet-50 pretrained trên ImageNet, bỏ lớp phân loại và lớp pooling cuối, **đóng băng** trọng số. Mỗi ảnh được biểu diễn thành `49 vùng × 2048 chiều` (lưới 7×7).

### Mô hình 1: LSTM + Attention

| Thành phần | Cấu hình |
|---|---|
| Attention | Additive attention, `attention_dim = 256` |
| Decoder | `LSTMCell`, `decoder_dim = 512`, `embed_dim = 256` |
| Khởi tạo trạng thái | `h0`, `c0` từ trung bình đặc trưng ảnh |
| Dropout | 0.5 |
| Optimizer | Adam, `lr = 1e-3` |

Tại mỗi bước, attention tính context vector từ 49 vùng ảnh dựa trên hidden state hiện tại, rồi ghép với embedding của từ trước đó để đưa vào LSTM.

### Mô hình 2: Transformer

| Thành phần | Cấu hình |
|---|---|
| Projection | Linear 2048 → 512 |
| Positional Encoding | Sinusoidal, cộng vào đặc trưng ảnh và embedding từ |
| Encoder | 3 lớp, 8 head, `d_model = 512`, `ffn = 2048` |
| Decoder | 3 lớp, 8 head, causal mask, `d_model = 512` |
| Dropout | 0.1 |
| Optimizer | Adam, `lr = 1e-4` |

### Cấu hình huấn luyện chung

- Loss: `CrossEntropyLoss` (bỏ qua vị trí `<pad>`)
- Batch size 32, 20 epoch
- Lưu checkpoint có **validation loss thấp nhất**
- Sinh caption bằng **greedy decoding**, tối đa 30 từ

### Tìm ảnh bằng tiếng Việt (CLIP)

- Encode ảnh: `clip-ViT-B-32`
- Encode câu tiếng Việt: `clip-ViT-B-32-multilingual-v1` (text encoder đa ngôn ngữ được distill để khớp không gian vector của CLIP gốc)
- Vector đã chuẩn hóa, xếp hạng theo cosine similarity
- Dùng **zero-shot**, không huấn luyện lại trên UIT-ViIC

## Kết quả

### Sinh caption (tập test, 231 ảnh)

| Mô hình | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr |
|---|---|---|---|---|---|
| CNN + LSTM-Attention | **0.6841** | 0.5165 | 0.4076 | 0.3167 | 0.7944 |
| CNN + Transformer | 0.6824 | **0.5353** | **0.4311** | **0.3439** | **0.8885** |

Transformer cao hơn LSTM-Attention ở BLEU-2/3/4 và CIDEr (CIDEr tăng 0.094, tương đương khoảng 12%), còn BLEU-1 hai mô hình gần như bằng nhau.

Checkpoint tốt nhất của LSTM-Attention ở epoch 2 (val loss 1.6454), của Transformer ở epoch 4 (val loss 1.4210).


## Cấu trúc dự án

```
.
├── image-caption.ipynb   # Toàn bộ pipeline: dữ liệu, huấn luyện, đánh giá, tìm ảnh
└── README.md
```

Các file sinh ra khi chạy notebook: `vocab.json`, `train_pairs.json`, `val_pairs.json`, `test_pairs.json`, `best_model.pth` (LSTM-Attention), `best_transformer_model.pth` (Transformer), `image_embeddings.pt`, `image_paths.json`.

## Cài đặt và chạy

Notebook được viết để chạy trên **Kaggle** (GPU). Môi trường cần:

```bash
pip install underthesea pyvi nltk pycocoevalcap sentence-transformers torch torchvision pillow tqdm requests matplotlib
```

Các bước:

1. Tải bộ dữ liệu UIT-ViIC (3 file `uitviic_captions_{train,val,test}2017.json`) và chỉnh biến `DATA_DIR` ở đầu notebook cho đúng đường dẫn.
2. Chạy lần lượt các phần trong `image-caption.ipynb`:
   1. Xử lý dữ liệu và tokenizer tiếng Việt (có bước tải 3.850 ảnh, khoảng 8 phút)
   2. Huấn luyện và đánh giá CNN + LSTM-Attention
   3. Huấn luyện và đánh giá CNN + Transformer
   4. Trích xuất vector CLIP và tìm ảnh bằng tiếng Việt
3. Ví dụ tìm ảnh:

```python
results = search_images("Một người đàn ông đang chơi bóng đá trên sân cỏ", top_k=5)
```

Ví dụ sinh caption cho một ảnh bằng mô hình Transformer:

```python
image = Image.open("anh.jpg").convert("RGB")
caption = generate_caption_transformer(model, image_transform(image), tokenizer, device)
print(caption)
```

## Hạn chế và hướng phát triển

- **Dữ liệu nhỏ:** chỉ 2.695 ảnh train nên cả hai mô hình overfit nhanh (val loss tăng sau vài epoch). Có thể thử data augmentation, early stopping, hoặc dùng thêm dữ liệu.
- **Encoder bị đóng băng:** chưa fine-tune ResNet-50. Có thể thử mở khóa các block cuối.
- **Greedy decoding:** chưa dùng beam search nên caption đôi khi lặp hoặc cụt.
- **Chủ đề hẹp:** dữ liệu tập trung vào cảnh thể thao (tennis, bóng chày, bóng đá), nên mô hình khó tổng quát sang cảnh khác.
- **Tìm ảnh:** có thể fine-tune text encoder trên cặp ảnh–caption tiếng Việt để tăng Recall, hoặc thử các mô hình CLIP đa ngôn ngữ lớn hơn.


## Tham khảo

- Xu et al., *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*, 2015
- Vaswani et al., *Attention Is All You Need*, 2017
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*, 2021
- UIT-ViIC: bộ dữ liệu caption tiếng Việt cho ảnh
