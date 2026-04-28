import cv2
import os

# Đường dẫn video và thư mục lưu frame
video_path = r'C:\Users\admin\Videos\vlc-record-2026-04-20-20h39m02s-rtsp___293pvdcam3.cameraddns.net_5556_live-.mp4'
save_folder = r'C:\Users\admin\Desktop\VTT_opendoor\data'
os.makedirs(save_folder, exist_ok=True)

# Load video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không thể mở video!")
    exit()

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Kết thúc video

    if frame_count % 50 == 0 :#and saved_count < 501
        saved_count += 1
        height, width, _ = frame.shape
        zone = [[0.84, 0.01], [0.07, 0.0], [0.18, 0.74], [0.82, 0.7]]
        pts = [(int(x * width), int(y * height)) for x, y in zone]
        x_min = min([p[0] for p in pts])
        y_min = min([p[1] for p in pts])
        x_max = max([p[0] for p in pts])
        y_max = max([p[1] for p in pts])

        crop = frame[y_min:y_max, x_min:x_max]
        # roi = [450, 0, 1700, 300]
        # frame_roi = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        save_path = os.path.join(save_folder, f"opendoor11_VTT{saved_count:04d}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"Đã lưu {save_path}")
        
        # Thoát bằng phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Hoàn thành!")