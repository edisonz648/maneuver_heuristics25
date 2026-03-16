import cv2
import csv
import time
import numpy as np  # For median calculation

video_ID = '1002ENT'

# === Config ===
video_path = f'clips/Parking-clip{video_ID}.mp4'
output_csv = f'clip-annotations/maneuver{video_ID}.csv'

# === Setup ===
FPS = 30  # Use FPS value directly
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video.")
    exit()

cv2.namedWindow("Manual Timing")

# Prepare CSV with header if not exists
try:
    with open(output_csv, 'x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "start_frame", "end_frame"])
except FileExistsError:
    pass  # File already exists

# Variables to store multiple start and end frames
start_frames = []
end_frames = []
frame_idx = 0
last_space_time = 0
space_cooldown = 1.0  # seconds
maneuver_recorded = False  # Flag to ensure only one maneuver is recorded

# Counter to track how many maneuvers the user has completed
maneuver_count = 0

print("[INFO] Press SPACE to start/end timing. Press 'q' to quit.")

while maneuver_count < 3:
    # Reset the video back to the start for each new attempt
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    start_frame = None
    frame_idx = 0
    maneuver_recorded = False  # Reset for each new maneuver

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        label = f"Start Frame: {start_frames[-1] if start_frames else 'Not set'}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display current frame number in blue on bottom-left
        height = frame.shape[0]
        cv2.putText(frame, f"Frame: {frame_idx}", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


        if maneuver_recorded:
            # Display end frame info after maneuver has been completed
            end_label = f"End Frame: {frame_idx}"
            cv2.putText(frame, end_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Manual Timing", frame)
        key = cv2.waitKey(int(1000 / FPS)) & 0xFF

        current_time = time.time()

        if key == ord(' ') and (current_time - last_space_time) > space_cooldown:
            last_space_time = current_time
            if not maneuver_recorded:  # Start timing a maneuver
                start_frames.append(frame_idx)
                print(f"[START] Maneuver {maneuver_count + 1} started at frame {frame_idx}")
                maneuver_recorded = True
            else:  # End timing the maneuver
                end_frames.append(frame_idx)
                print(f"[END] Maneuver {maneuver_count + 1} ended at frame {frame_idx}")
                with open(output_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["", start_frames[maneuver_count], end_frames[maneuver_count]])
                print(f"[SAVED] ({start_frames[maneuver_count]} → {end_frames[maneuver_count]}) to {output_csv}")
                maneuver_recorded = False  # Reset for next maneuver
                maneuver_count += 1  # Increment maneuver count

                # Display end time briefly before resetting the footage
                cv2.putText(frame, f"End Frame: {end_frames[maneuver_count-1]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Manual Timing", frame)
                cv2.waitKey(2000)  # Show end frame for 2 seconds before resetting the video
                
                # Reset the video to the beginning for the next maneuver
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                break  # Exit the loop to reset the video and start a new maneuver

            if maneuver_count == 3:
                # After 3 maneuvers, calculate the median of the start and end frames
                median_start = int(np.median(start_frames))
                median_end = int(np.median(end_frames))
                print(f"[INFO] Median Start Frame: {median_start}")
                print(f"[INFO] Median End Frame: {median_end}")

                # Display median on the screen
                cv2.putText(frame, f"Median Start: {median_start}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(frame, f"Median End: {median_end}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.imshow("Manual Timing", frame)
                cv2.waitKey(2000)  # Show the median results for 2 seconds
                break  # Exit the loop after 3 maneuvers

        elif key == ord('q'):
            print("[QUIT] Exiting.")
            break

cap.release()
cv2.destroyAllWindows()
