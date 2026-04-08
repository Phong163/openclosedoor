import cv2
import time
import datetime

# RTSP camera URL
rtsp_url = "rtsp://admin:eidox.ai2026@192.168.100.112:5552/live"

# Mở camera
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Không kết nối được camera!")
    exit()

# Lấy thông số video
fps = 20.0  # nếu camera trả về 0 thì dùng mặc định
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Hàm tạo tên file theo giờ
def new_filename():
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"camera_{now}.mp4"

# Tạo file đầu tiên
filename = new_filename()
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

start_time = time.time()
max_duration = 60 * 60  # 1 giờ

print("Bắt đầu ghi video...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Mất kết nối camera, thử lại...")
        time.sleep(1)
        continue

    out.write(frame)
    f = cv2.resize(frame,(1280,640))
    cv2.imshow("frame",f)
    # # Nếu đủ 1 giờ thì tạo file mới
    # if time.time() - start_time >= max_duration:
    #     out.release()
    #     filename = new_filename()
    #     out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    #     start_time = time.time()
    #     print(f"Tạo file mới: {filename}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()