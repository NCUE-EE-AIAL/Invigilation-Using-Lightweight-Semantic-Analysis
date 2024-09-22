import cv2
import os
import sys

def extract_frames(video_path, output_dir, frame_rate=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}, FPS: {fps}")

    frame_indices = [int(fps * i) for i in range(int(total_frames / fps)) if i % frame_rate == 0]

    frame_id = 1
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        #print(f"Trying to grab frame at index {idx}")
        ret, frame = cap.read()
        if ret:
            filename = os.path.join(output_dir, f"{os.path.basename(output_dir)}_{frame_id:02}.png")
            cv2.imwrite(filename, frame)
            frame_id += 1
        else:
            print(f"Failed to grab frame at index {idx}")
    cap.release()

def process_all_videos(root_dir, output_root):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(subdir, file)
                output_dir = os.path.join(output_root, os.path.splitext(file)[0])
                print(f"Processing {video_path}...")
                extract_frames(video_path, output_dir)

# 使用範例
root_directory = r'D:\NCUE_lab\Data\Classificaiton\Students_Video'
output_directory = r'D:\NCUE_lab\Data\Classificaiton\Students_Picture'
process_all_videos(root_directory, output_directory)
