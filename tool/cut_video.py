from moviepy.video.io.VideoFileClip import VideoFileClip

video = VideoFileClip(r"C:\Users\admin\Videos\vlc-record-2026-04-01-07h32m41s-rtsp___293pvdcam1.cameraddns.net_5554_live-.mp4")

# Cắt từ giây 10 đến 20
cut = video.subclipped(1347, 1440)

cut.write_videofile("output.mp4")