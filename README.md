# Dự đoán Bệnh Tiểu đường bằng Machine Learning

## Giới thiệu

Trong lĩnh vực chăm sóc sức khỏe, việc phát hiện và phòng ngừa sớm đóng vai trò then chốt trong việc kiểm soát các tình trạng mãn tính và cải thiện kết quả của bệnh nhân. Đái tháo đường hay tiểu đường[^1] là một nhóm các rối loạn chuyển hóa đặc trưng là tình trạng đường huyết cao kéo dài, đây là một bệnh phổ biến gây những rủi ro đáng kể nếu không được điều trị kịp thời.

Dự án này tập trung vào việc xây dựng một hệ thống dự đoán bệnh tiểu đường dựa trên bộ dữ liệu sức khỏe. Mục tiêu chính là tạo ra các mô hình máy học hiệu quả để dự đoán bệnh tiểu đường dựa trên các yếu tố sức khỏe và lối sống. Ngoài ra, dự án cũng bao gồm một website trực quan hóa dữ liệu và mô phỏng quá trình phân tích dữ liệu thăm dò (EDA).

---

## Bộ dữ liệu

- **Tên tệp**: `diabetes_012_health_indicators_BRFSS2021.csv`
- **Số lượng mẫu**: 236,378
- **Số lượng cột**: 22
- **Mục tiêu**: Cột `Diabetes_012` (0: Không bị tiểu đường, 1: Tiền tiểu đường, 2: Tiểu đường).
- **Các cột đặc trưng**: Bao gồm chỉ số BMI, huyết áp cao, cholesterol, hoạt động thể chất, và các yếu tố nhân khẩu học như tuổi, giới tính, thu nhập, v.v.

|

---

## Mục tiêu dự án

1. **Tiền xử lý dữ liệu**:
   - Làm sạch và chuẩn hóa dữ liệu.
   - Xử lý mất cân bằng dữ liệu bằng nhiều thuật toán khác nhau.
   - Tăng cường dữ liệu (Data Augmentation).

2. **Phân tích dữ liệu thăm dò (EDA)**:
   - Khám phá mối quan hệ giữa các biến đặc trưng.
   - Trực quan hóa dữ liệu bằng các biểu đồ.

3. **Xây dựng và huấn luyện mô hình**:
   - Chọn 3 mô hình máy học phù hợp dựa trên EDA.
   - Huấn luyện mô hình trên các tập dữ liệu đã xử lý.
   - Áp dụng thuật toán **AdaBoosting** để cải thiện hiệu suất.

4. **Xây dựng website**:
   - Dùng Spark và Plotly để trực quan hóa dữ liệu và kết quả EDA.
   - Cung cấp giao diện mô phỏng quá trình phân tích và dự đoán.

---

## Công cụ và thư viện sử dụng

- **Ngôn ngữ lập trình**: Python
- **Thư viện chính**:
  - `sklearn`, `pandas`, `numpy`, `matplotlib`, `seaborn` (Xử lý dữ liệu và xây dựng mô hình).
  - `spark`, `plotly` (Xây dựng website và trực quan hóa dữ liệu).

---

## Cách sử dụng

### 1. Cài đặt

```bash
# Tạo môi trường ảo
python -m venv env
source env/bin/activate  # Trên Linux/Mac
env\Scripts\activate   # Trên Windows

# Cài đặt các thư viện
pip install -r requirements.txt
```

### 2. Chạy dự án

- **Huấn luyện mô hình**:

```bash
python train_models.py
```

- **Khởi chạy website**:

```bash
python app.py
```

---

## Kỳ vọng kết quả

- **Hiệu suất mô hình**:
  - Đạt độ chính xác trên 85%.
  - Các chỉ số quan trọng: F1-score, ROC-AUC.
- **Giao diện website**:
  - Cung cấp trực quan hóa dễ hiểu.
  - Mô phỏng chi tiết từng bước phân tích dữ liệu.

---

## Thách thức và giải pháp

1. **Mất cân bằng dữ liệu**:
   - Sử dụng các kỹ thuật như SMOTE, under-sampling, over-sampling.
2. **Hiệu suất mô hình**:
   - Tối ưu hóa hyperparameters.
   - Áp dụng AdaBoosting để cải thiện kết quả.

---

## Mở rộng trong tương lai

- Tích hợp thêm dữ liệu từ các nguồn khác.
- Áp dụng các mô hình học sâu (Deep Learning).
- Xây dựng API cho dự án.

---

## Bản quyền

Dự án này được phát triển cho mục đích học tập và nghiên cứu. Mọi đóng góp hoặc sử dụng lại vui lòng ghi nguồn.

---

Đây là tất cả!

---

[^1]: Bệnh viện Đa khoa Tâm Anh, "Đái tháo đường: Nguyên nhân, dấu hiệu, chẩn đoán, cách phân loại", Tam Anh Hospital, 07/06/2021, <https://tamanhhospital.vn/dai-thao-duong/>
