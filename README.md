1. Giới thiệu dự án: 

Trong bối cảnh trí tuệ nhân tạo ngày càng được ứng dụng rộng rãi vào đời sống, các hệ thống nhận diện vật thể và định giá sản phẩm đã xuất hiện tại nhiều cửa hàng tiện lợi như Circle K, Bách Hóa Xanh, GS25, CoopMart,... Tuy nhiên, các giải pháp này phần lớn vẫn dựa trên công nghệ quét mã vạch hoặc nhận diện sản phẩm có bao bì đóng gói sẵn. Trong khi đó, tại các môi trường như căn tin trường học, bệnh viện, hay khu công nghiệp, việc áp dụng phương thức quét mã vạch lại gặp nhiều hạn chế. Món ăn trong căn tin thường được chế biến tại chỗ, không có bao bì tiêu chuẩn và thay đổi linh hoạt theo ngày, khiến việc áp dụng mã vạch gần như không khả thi. Dự án của chúng tôi mang đến một hệ thống hoàn toàn tự động trong việc nhận diện món ăn trên khay cơm và thực hiện thanh toán một cách nhanh chóng và chính xác.

Trái tim của hệ thống là sự kết hợp mạnh mẽ giữa công nghệ YOLO tiên tiến để phát hiện chính xác vị trí từng món ăn và mô hình mạng nơ-ron tích chập (CNN) được huấn luyện chuyên biệt để phân loại đa dạng các món ăn một cách thông minh. Ngay sau khi nhận diện, hệ thống sẽ tự động tính toán tổng chi phí dựa trên bảng giá được thiết lập sẵn và tạo mã QR tiện lợi, dẫn đến trang thanh toán giả lập trực quan.

Điểm nổi bật của giải pháp:

Tiết kiệm thời gian: Loại bỏ hoàn toàn quy trình nhận diện và tính tiền thủ công, giảm thiểu thời gian chờ đợi cho khách hàng.
Giảm thiểu nhân lực: Tối ưu hóa nguồn lực, cho phép nhân viên tập trung vào các nhiệm vụ quan trọng khác.
Nâng cao trải nghiệm người dùng: Mang đến sự tiện lợi, nhanh chóng và hiện đại trong quá trình thanh toán.
Độ chính xác cao: Công nghệ AI tiên tiến đảm bảo khả năng nhận diện và phân loại món ăn một cách đáng tin cậy.
Khả năng mở rộng linh hoạt: Dễ dàng tích hợp thêm các món ăn mới, cập nhật bảng giá và mở rộng sang việc sử dụng camera trực tiếp.
2. Hướng dẫn cài đặt nhanh chóng:

Để bắt đầu khám phá sức mạnh của hệ thống, bạn có thể thực hiện theo các bước đơn giản sau:

Bước 1: Đảm bảo bạn đã cài đặt Python phiên bản 3.8 trở lên (tải từ https://www.python.org/downloads/).

Bước 2: Cài đặt Visual Studio Code (VSCode) để có môi trường phát triển tốt nhất.

Bước 3: Tải mã nguồn dự án từ GitHub hoặc sao chép vào máy tính của bạn.

Bước 4: Mở thư mục dự án bằng VSCode.

Bước 5: Tạo và kích hoạt môi trường ảo (virtual environment) để quản lý các thư viện một cách độc lập:
Mở Terminal (trong VSCode).
Chạy lệnh: python -m venv venv
Kích hoạt môi trường:
Windows: .\venv\Scripts\activate
macOS/Linux: source venv/bin/activate

Bước 6: Cài đặt tất cả các thư viện cần thiết một cách dễ dàng bằng lệnh: pip install -r requirements.txt
3. Khám phá cách hệ thống hoạt động:

Chỉ với vài bước đơn giản, bạn có thể trải nghiệm quy trình nhận diện và thanh toán tự động:

Bước 1: Đảm bảo rằng mô hình CNN đã được huấn luyện và lưu trữ với tên model.h5 trong thư mục dự án.

Bước 2: Khởi chạy ứng dụng bằng lệnh: python app.py

Bước 3: Giao diện trực quan sẽ xuất hiện, cho phép bạn chọn ảnh khay cơm từ thiết bị của mình.

Bước 4: Hệ thống sẽ tự động thực hiện các tác vụ phức tạp:
Phát hiện vị trí các món ăn trên khay cơm bằng công nghệ YOLO.
Phân loại từng món ăn một cách chính xác bằng mô hình CNN.

4. Nền tảng công nghệ vững chắc:

Dự án được xây dựng dựa trên các thư viện mạnh mẽ và phổ biến trong lĩnh vực xử lý ảnh và học máy, được liệt kê chi tiết trong file requirements.txt:

opencv-python: Thư viện hàng đầu cho các tác vụ xử lý ảnh và video.
numpy: Nền tảng cho các phép toán số học và xử lý mảng đa chiều hiệu quả.
ultralytics: Thư viện mạnh mẽ hỗ trợ mô hình phát hiện đối tượng YOLO.
tensorflow: Framework mã nguồn mở cho việc xây dựng và huấn luyện các mô hình CNN.
Pillow: Thư viện Python Imaging Library, hỗ trợ đa dạng các định dạng ảnh.
Ngoài ra, dự án còn tận dụng các thư viện tích hợp sẵn của Python:

Gradio: Bộ công cụ mạnh mẽ để xây dựng giao diện người dùng đồ họa trực quan.
threading: Hỗ trợ thực hiện các tác vụ đồng thời, đảm bảo giao diện mượt mà trong quá trình xử lý ảnh.
5. Kiến trúc chương trình thông minh và linh hoạt:

Dự án được tổ chức một cách khoa học và dễ bảo trì với các thư mục riêng biệt cho từng loại món ăn (ví dụ: Com Trang/, Thit kho/, Trung chien/,...). Điều này không chỉ giúp quản lý dữ liệu huấn luyện hiệu quả mà còn tạo tiền đề cho việc mở rộng và cập nhật hệ thống một cách dễ dàng.

Giao diện người dùng được xây dựng bằng Gradio mang đến trải nghiệm thân thiện và trực quan. Việc ứng dụng đa luồng giúp chương trình hoạt động mượt mà, không bị gián đoạn trong quá trình xử lý ảnh phức tạp.

Với cấu trúc linh hoạt, dự án sẵn sàng cho việc bổ sung các món ăn mới, cập nhật bảng giá một cách nhanh chóng hoặc thậm chí tích hợp trực tiếp với camera để xử lý hình ảnh theo thời gian thực.
