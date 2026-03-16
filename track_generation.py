import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort
import csv
import json
from enum import Enum

video_ID = '1000EXT'

# Setup
input_size = [800, 1440]
device = torch.device('cpu')
model = YOLO("/home/edisonz/maneuver_heuristics25/yolov8x.pt")
tracker = BotSort(reid_weights=Path('clip_vehicleid.pt'), device=device, half=False)
vid = cv2.VideoCapture(f'/home/edisonz/maneuver_heuristics25/clips/Parking-clip{video_ID}.mp4')
FPS = 30
print(vid.get(cv2.CAP_PROP_FPS))

class ManeuverType(Enum):
    ENT = 0
    EXT = 1

maneuver_type = ManeuverType.ENT if (video_ID[-3:].lower()=='ent') else ManeuverType.EXT

# Parameters
TIME_RESOLUTION = 0.1
SKIP = int(TIME_RESOLUTION * FPS)

selected_track = None

clicked_point = None
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global clicked_point 
        clicked_point = (x, y)

frame_count = 0
cv2.namedWindow("Tracking")
cv2.setMouseCallback("Tracking", click_event)
while True:
    ret, full_frame = vid.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % SKIP != 0:
        continue

    height = full_frame.shape[0]
    cropped_frame = full_frame[height//3:, :]

    results = model(cropped_frame)[0]

    dets = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().item()
        cls_id = int(box.cls[0].cpu().item())
        if cls_id != 2 and cls_id != 7:
            continue
        dets.append([x1, y1, x2, y2, conf, cls_id])

    dets = np.array(dets)
    tracks = tracker.update(dets, cropped_frame)

    for track in tracks:
        x1, y1, x2, y2, track_id = track[:5]

        if(clicked_point is not None and selected_track is None
        and x1 < clicked_point[0] < x2
           and y1 < clicked_point[1] < y2):
            print(f'track {track_id} selected')
            selected_track = int(track_id)
            vid.release()
            cv2.destroyAllWindows()
    
    if(selected_track is not None):
        break

    tracker.plot_results(cropped_frame, show_trajectories=True)
    cv2.imshow("Tracking", cropped_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        vid.release()
        cv2.destroyAllWindows()
        break

# Setup
input_size = [800, 1440]
device = torch.device('cpu')
model = YOLO("/home/edisonz/maneuver_heuristics25/yolov8x.pt")
tracker = BotSort(reid_weights=Path('clip_vehicleid.pt'), device=device, half=False)
vid = cv2.VideoCapture(f'/home/edisonz/maneuver_heuristics25/clips/Parking-clip{video_ID}.mp4')
FPS = 30

# Parameters
TIME_RESOLUTION = 0.1
SKIP = int(TIME_RESOLUTION * FPS)

clicked_point = None
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global clicked_point 
        clicked_point = (x, y)

with open('/home/edisonz/maneuver_heuristics25/mappings.json', 'r') as f:
    LR_ID = json.load(f).get(f'{video_ID}')
    print(LR_ID)

with open(f'/home/edisonz/maneuver_heuristics25/regions{LR_ID}.json', 'r') as f:
    data = json.load(f)

parking_areas = [
    np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
    for poly in data.get("parking_areas", [])
]

def in_parking_area(point):
    return any(cv2.pointPolygonTest(polygon, point, False) >= 0 for polygon in parking_areas)

# CSV output setup
csv_file = open(f'clips/Parking-clip{video_ID}.csv', 'w', newline='')
writer = csv.DictWriter(csv_file, fieldnames=["frame", "track_id", "in_parking_zone", "cx", "cy", "x1", "x2", "y1", "y2", "height", "width"])
writer.writeheader()

frame_count = 0
while True:
    ret, full_frame = vid.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % SKIP != 0:
        continue

    height = full_frame.shape[0]
    cropped_frame = full_frame[height//3:, :]

    results = model(cropped_frame)[0]

    dets = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().item()
        cls_id = int(box.cls[0].cpu().item())
        if cls_id != 2 and cls_id != 7:
            continue
        dets.append([x1, y1, x2, y2, conf, cls_id])

    dets = np.array(dets)
    tracks = tracker.update(dets, cropped_frame)

    for track in tracks:
        x1, y1, x2, y2, track_id = track[:5]
        if(track_id!=selected_track):
            continue
        
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bottom = (int(cx), int(y2))  # bottom-center

        in_parking = None
        if(maneuver_type==ManeuverType.ENT):
            if(in_parking_area((int(x1),int(y2))) or in_parking_area((int(x2),int(y2)))):
                in_parking = 1
            else: 
                in_parking = 0
        if(maneuver_type==ManeuverType.EXT):
            if(in_parking_area((int(x1),int(y2))) or in_parking_area((int(x2),int(y2)))):
                in_parking = 1
            else:
                in_parking = 0

        height = y2 - y1
        width = x2 - x1

        writer.writerow({
            "frame": frame_count,
            "track_id": int(track_id),
            "in_parking_zone": in_parking,
            "cx": cx,
            "cy": cy,
            "x1": x1,
            "x2": x2,
            "y1": y1,
            "y2": y2,
            "height": height,
            "width": width
        })

# Cleanup
vid.release()
csv_file.close()
cv2.destroyAllWindows() 
