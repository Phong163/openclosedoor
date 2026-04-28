import cv2
import numpy as np

# Danh sách để lưu tọa độ các điểm
points = []
sl_point = 20
# Hàm xử lý sự kiện chuột
def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:  # Nhấp chuột trái để lấy tọa độ
        points.append([round(x/720, 2), round(y/640, 2)])  # Sửa lỗi ở đây
        print(f"Điểm {len(points)}: ({x}, {y})")
        # Vẽ vòng tròn tại điểm nhấp chuột
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", image)

        # Khi có 6 điểm, tự động dừng
        if len(points) == sl_point:
            print("Đã lấy đủ 6 điểm:", points)
            cv2.destroyAllWindows()

# Đọc ảnh
image = cv2.imread(r"C:\Users\admin\Downloads\Screenshot 2026-04-16 233917.png")
image = cv2.resize(image, (720, 640))
if image is None:
    print("Không thể tải ảnh!")
    exit()

# Hiển thị ảnh
cv2.imshow("Image", image)

# Thiết lập sự kiện chuột
cv2.setMouseCallback("Image", mouse_callback)

# Chờ người dùng nhấn 'q' để thoát hoặc tự động thoát khi có 6 điểm
while True:
    if cv2.waitKey(1) & 0xFF == ord('q') or len(points) == sl_point:
        break

# Hiển thị tọa độ cuối cùng
print("Tọa độ 6 điểm:", points)

# Giải phóng cửa sổ
cv2.destroyAllWindows()