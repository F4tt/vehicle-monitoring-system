################ Cách dùng ##################
1. giải nén file
2. chạy lệnh: "pip install - r requirements.txt"
3. thêm các video cần xử lý vào thư mục /videos
4. thay đổi các tham số trong file config.yaml để phù hợp với nhu cầu (mặc định đang dùng detector yolov8n với cpu và mobilenet làm reid model cho tracker deepsort)
5. Chạy file main.py để hiển thị GUI, kết quả được lưu trong thư mục /logs


############### Các tính năng đã thực hiện ################
### Phát hiện đối tượng

- Phát hiện đối tượng phương tiện (ví dụ: YOLO, v.v.)
- Phân loại loại phương tiện (ví dụ: ô tô con, xe buýt, xe tải, v.v.)
- Tính toán số lượng phương tiện và số lượng người đi bộ trên đường


### Theo dõi đối tượng và quản lý ID

- Theo dõi đa đối tượng
- Gán và duy trì ID duy nhất cho mỗi đối tượng
- Nhận diện lại (Re-ID) các phương tiện bị che khuất hoặc trùng lặp


### Phân tích hướng di chuyển và đi ngược chiều

- Xác định hướng dựa trên sự thay đổi tọa độ giữa các khung hình
- Phân tích vào/ra khu vực dựa trên vùng định trước (Line/ROI)
- Hiển thị quỹ đạo di chuyển của phương tiện
- Phát hiện và đánh dấu xe đi ngược chiều

### Xử lý video và xuất kết quả

- Phân tích đồng thời nhiều nguồn video CCTV
- Ghi chồng Overlay kết quả thời gian thực lên video (ID phương tiện, hướng, trạng thái, v.v.)
- Cân bằng khung hình (ổn định FPS)
- Lưu kết quả phân tích (CSV, JSON, v.v.)

### Tối ưu hóa và kiến trúc hệ thống

- Tối ưu hiệu năng xử lý thời gian thực: Skip frame, Multithreading



################# Lý do lựa chọn #######################
-Yolov8n thay vì các mô hình mạnh mẽ hơn như Yolov8l, Yolov9, Transformer-based vì hệ thống triển khai trên edge device chưa rõ cấu hình nên chọn mô hình cân bằng giữa hiệu năng và tốc độ.?
-DeepSORT thay vì ByteTrack hay StrongSORT bởi yêu cầu cần Re-ID nên cần các mô hình có Ri-ID module như DeepSORT nhưng vẫn cần cân bằng với tốc độ.
-Bởi mục tiêu là xử lý nhiều luồng camera nên việc xác định một ROI chung dùng cho tất cả là không khả quan, do đó sử dụng LINE để đếm; mặc dù nếu có thể dùng ROI để giới hạn phạm vi xử lý sẽ gia tăng hiệu suất đáng kể
-Laptop hiện tại không có GPU nên việc implement thử các tính năng với GPU là hạn chế
<img width="1919" height="1078" alt="Ảnh chụp màn hình 2025-10-04 160944" src="https://github.com/user-attachments/assets/b25d701e-01d6-480e-bcbd-0f8f12b4aa19" />

