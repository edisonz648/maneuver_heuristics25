import cv2
import numpy as np
import pandas as pd
from enum import Enum
import json
import matplotlib.pyplot as plt

def analyze_maneuversv2(video_ID):
    """"Returns front_parking, rear_parking, apex, zone-based"""
    # Enter hyperparameters (adjustable)
    start_entry_ratio = 0 # 0 = zone based | 1 = peak based 
    end_entry_ratio = 1.0 # 0 = front parking zone | 1 = rear parking zone

    # Exit hyperparameters (adjustable)
    start_exit_ratio = 0 # 0 = rear parking zone | 1 = front parking zone
    end_exit_ratio = 1.0 # 0 = peak based | 1 = zone based

    # low motion hyperparameters 
    movement_threshold = 1.0

    # Peak hyperparameters 
    tA = 5
    aA = np.pi/6

    class ManeuverType(Enum):
        ENT = 0
        EXT = 1

    maneuver_type = ManeuverType.ENT if video_ID[-3:].lower()=='ent' else ManeuverType.EXT

    vid = cv2.VideoCapture(f'/home/edisonz/maneuver_heuristics25/Parking-clip{video_ID}.mp4')
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # Max x-coordinate
    frame_height = 2*int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))//3  # Max y-coordinate
    # ----------------------------------------------------------------------------------------------------------------------------

    def load_lines(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        line_endpoints = [list(map(tuple, line)) for line in data['lines']]
        line_params = []
        for line in line_endpoints: 
            m = (line[1][1] - line[0][1])/(line[1][0] - line[0][0])
            c = line[0][1] - m * line[0][0]
            line_params.append((m, c))
        return line_params, line_endpoints

    # Load data
    vehicle_df = pd.read_csv(f'clip_trajectory_csvs/Parking-clip{video_ID}.csv')
    maneuvers = pd.read_csv(f'clip-annotations/maneuver{video_ID}.csv')
    ann_start_frame, ann_end_frame = np.median(maneuvers["start_frame"]), np.median(maneuvers["end_frame"])
    driving_df = vehicle_df[vehicle_df["in_parking_zone"]==0]
    driving_coords = driving_df[['cx', 'cy']].values
    driving_coords_ground = driving_df[['cx', 'y2']].values
    parking_df = vehicle_df[vehicle_df["in_parking_zone"]==1]
    parking_coords = parking_df[['cx', 'cy']].values
    parking_coords_ground = parking_df[['cx', 'y2']].values

    region_code = None
    with open("mappings.json", 'r') as f:
        data = json.load(f)
        region_code = data.get(f'{video_ID}')
    parking_lines = f'scene_annotation/lines/lines{region_code}.json'
    all_line_params, all_line_endpoints = load_lines(parking_lines)

    class ManeuverType(Enum):
        ENT = 0
        EXT = 1

    maneuver_type = ManeuverType.ENT if video_ID[-3:].lower()=='ent' else ManeuverType.EXT

    # --------------------------------------------------------------------------------------------------------------------------

    def plot_maneuver_analysis(video_ID, peak_idx, zone_based_idx):
        # 1. Run your analysis to get the front_end_point and data
        # (Assuming these variables are accessible from your function scope)
        # For this example, we re-extract the necessary bits:

        # ... [Your existing loading logic for vehicle_df and all_line_params] ...

        # 2. Find the two nearest parking lines
        # Distance from point (x0, y0) to line mx - y + c = 0 is:
        # d = |m*x0 - y0 + c| / sqrt(m^2 + 1)

        line_distances = []
        for i, (m, c) in enumerate(all_line_params):
            dist = abs(m * front_end_point[0] - front_end_point[1] + c) / np.sqrt(m**2 + 1)
            line_distances.append((dist, i))

        # Sort by distance and pick top 2
        line_distances.sort()
        # Takes the top 2 indices if they exist, or just 1 (or 0) if they don't
        nearest_indices = [item[1] for item in line_distances[:2]]

        # 3. Plotting
        plt.figure(figsize=(12, 8))

        # Plot the two nearest lines
        for idx in nearest_indices:
            endpoints = all_line_endpoints[idx]
            px, py = zip(*endpoints)
            plt.plot(px, py, 'r-', linewidth=3, label=f'Parking Line {idx}')

        # Plot vehicle corner trajectories
        # (x1, y2) = Bottom Left | (x2, y2) = Bottom Right
        plt.scatter(vehicle_df['x1'], vehicle_df['y2'], s=1, c='blue', alpha=0.5, label='Bottom-Left (x1, y2)')
        plt.scatter(vehicle_df['x2'], vehicle_df['y2'], s=1, c='green', alpha=0.5, label='Bottom-Right (x2, y2)')

        # Mark the front_end_point
        plt.scatter(driving_df.iloc[peak_idx]['x1'], driving_df.iloc[peak_idx]['y2'], c='yellow', s=100, edgecolors='black', label='Front End Point', zorder=5)
        plt.scatter(driving_df.iloc[peak_idx]['x2'], driving_df.iloc[peak_idx]['y2'], c='yellow', s=100, edgecolors='black', label='Front End Point', zorder=5)
        plt.scatter(driving_df.iloc[zone_based_idx]['x1'], driving_df.iloc[zone_based_idx]['y2'], c='cyan', s=100, edgecolors='black', label='Zone-Based Point', zorder=5)
        plt.scatter(driving_df.iloc[zone_based_idx]['x2'], driving_df.iloc[zone_based_idx]['y2'], c='cyan', s=100, edgecolors='black', label='Zone-Based Point', zorder=5)

        plt.title(f"Maneuver Analysis Visualization: {video_ID}")
        plt.xlabel("X Coordinate (pixels)")
        plt.ylabel("Y Coordinate (pixels)")
        plt.gca().invert_yaxis() # Standard image coordinate system
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        save_path = f"plot_{video_ID}.png"
        plt.savefig(save_path)
        print(f"Visualization saved to: {save_path}")
        plt.close()

    # Determine if direction reversal in x or y occured
    def is_reversal(r1, r2):
        angle_diff = np.abs(np.arctan2(r1[1], r1[0]) - np.arctan2(r2[1], r2[0]))
        if angle_diff < aA:
            return False  # No significant angle change
        return (r1[0] * r2[0] < 0) or (r1[1] * r2[1] < 0)  # Opposite signs in x or y

    # Determine if vehicle is hugging the border of the frame
    def on_border(df_row):
        height = df_row["height"]
        width = df_row["width"]
        x = df_row["cx"]
        y = df_row["cy"]
        left_x = int(x-width//2)
        right_x = int(x+width//2)
        top_y = int(y-height//2)
        bottom_y = int(y+height//2)
        return (left_x==0 or top_y==0 or right_x==frame_width or bottom_y==frame_height)

    # Motion logic BEGIN

    # Motion logic END
    # Peak logic BEGIN

    # Peak logic END
    # Region logic BEGIN

    def check_crossing(x1, y1, x2, y2, line_params, line_endpoints):
        """"only one set of line_params and line_endpoints"""
        if line_params is None or line_endpoints is None:
            return False
        m, c = line_params
            # [x_component, y_component] 
        r_normal = [m, -1]

        r1 = [x1 - line_endpoints[1][0], y1 - line_endpoints[1][1]]
        r2 = [x2 - line_endpoints[1][0], y2 - line_endpoints[1][1]]

        if r1[0]*r_normal[0] + r1[1]*r_normal[1] > 0 and r2[0]*r_normal[0] + r2[1]*r_normal[1] <= 0:
            return True 
        if r1[0]*r_normal[0] + r1[1]*r_normal[1] < 0 and r2[0]*r_normal[0] + r2[1]*r_normal[1] >= 0:
            return True
        return False

    def check_departure(driving_df, start_idx, end_idx, all_line_params, all_line_endpoints):
        departure_idx = None

        # 1. Get the reference point (using front_end_point as defined in your ENT logic)
        x_ref, y_ref = driving_df.iloc[start_idx]["cx"], driving_df.iloc[start_idx]["y2"]

        left_lines = []  # Stores (distance, index)
        right_lines = [] # Stores (distance, index)

        for i, (m, c) in enumerate(all_line_params):
            if m == 0: continue

            line_x_at_y_ref = (y_ref - c) / m
            dist = abs(m * x_ref - y_ref + c) / np.sqrt(m**2 + 1)

            if line_x_at_y_ref < x_ref:
                left_lines.append((dist, i))
            else:
                right_lines.append((dist, i))

        left_lines.sort()
        right_lines.sort()

        # Distance to the nearest line on each side (None if no line on that side)
        dL = left_lines[0][0] if left_lines else None
        dR = right_lines[0][0] if right_lines else None

        nearest_indices = []
        if left_lines:
            nearest_indices.append(left_lines[0][1])
        if right_lines:
            nearest_indices.append(right_lines[0][1])

        selected_line_params = [all_line_params[i] for i in nearest_indices]
        selected_line_endpoints = [all_line_endpoints[i] for i in nearest_indices]

        step = 1 if end_idx > start_idx else -1
        for idx in range(start_idx + 1, end_idx, step):
            x1 = driving_df.iloc[idx - 1]['x1']
            x2 = driving_df.iloc[idx - 1]['x2']
            y2 = driving_df.iloc[idx - 1]['y2']

            x1_prime = driving_df.iloc[idx]['x1']
            x2_prime = driving_df.iloc[idx]['x2']
            y2_prime = driving_df.iloc[idx]['y2']

            pL = (x1, y2)
            pR = (x2, y2)

            pL_prime = (x1_prime, y2_prime)
            pR_prime = (x2_prime, y2_prime)

            crossed_line_params = []
            crossed_line_endpoints = []
            initial_hit_code = None
            for slp, sle in zip(selected_line_params, selected_line_endpoints):
                if check_crossing(pL[0], pL[1], pL_prime[0], pL_prime[1], slp, sle):
                    initial_hit_code = 'pL'
                    crossed_line_params.append(slp)
                    crossed_line_endpoints.append(sle)
                if check_crossing(pR[0], pR[1], pR_prime[0], pR_prime[1], slp, sle):
                    initial_hit_code = 'pR'
                    crossed_line_params.append(slp)
                    crossed_line_endpoints.append(sle)

            for line_params, line_endpoints in zip(crossed_line_params, crossed_line_endpoints):
                if initial_hit_code == 'pL':
                    if check_crossing(pR[0], pR[1], pR_prime[0], pR_prime[1], line_params, line_endpoints):
                        departure_idx = idx
                        break
                elif initial_hit_code == 'pR':
                    if check_crossing(pL[0], pL[1], pL_prime[0], pL_prime[1], line_params, line_endpoints):
                        departure_idx = idx
                        break

            if departure_idx is not None:
                return departure_idx

        zone_distance = dL + dR if dL is not None and dR is not None else (dL if dL is not None else (dR if dR is not None else float('inf')))   

        for idx in range(start_idx, end_idx, step):
            if np.linalg.norm([driving_df.iloc[idx]['cx'] - x_ref, driving_df.iloc[idx]['y2'] - y_ref]) > zone_distance:
                return idx
        return end_idx

    def in_adjacent_zones(p1, p2, line_params):
        """This function calculates whether two points are in the same or adjacent zones"""
        x_point1 = p1[0]
        y_point1 = p1[1]
        x_point2 = p2[0]
        y_point2 = p2[1]

        intersections1 = []
        intersections2 = []
        for m, c in line_params:
            x_inter1 = (y_point1 - c) / m
            intersections1.append(x_inter1)
            x_inter2 = (y_point2 - c) / m
            intersections2.append(x_inter2)

        left1 = [(x, params) for x, params in zip(intersections1, line_params) if x < x_point1]
        right1 = [(x, params) for x, params in zip(intersections1, line_params) if x > x_point1]

        left2 = [(x, params) for x, params in zip(intersections2, line_params) if x < x_point2]
        right2 = [(x, params) for x, params in zip(intersections2, line_params) if x > x_point2]

        if len(left1) > 0: 
            __, left_params1 = max(left1, key = lambda x: x[0])  
        else: 
            left_params1 = None

        if len(right1) > 0:
            __, right_params1 = min(right1, key = lambda x: x[0]) 
        else:
            right_params1 = None

        if len(left2) > 0: 
            __, left_params2 = max(left2, key = lambda x: x[0])  
        else: 
            left_params2 = None

        if len(right2) > 0:
            __, right_params2 = min(right2, key = lambda x: x[0]) 
        else:
            right_params2 = None

        return (left_params1 == left_params2 or left_params1 == right_params2 or 
                right_params1 == right_params2 or right_params1 == left_params2)

    # Region logic END

    if maneuver_type == ManeuverType.ENT:
        # --- End-of-Maneuver Detection (STOP) ---
        front_end_point = parking_coords[0]
        dists_from_front = np.linalg.norm(parking_coords - front_end_point, axis=1)

        rear_end_idx = np.argmax(dists_from_front)
        rear_end_point = parking_coords[rear_end_idx]
        dists_from_rear = np.linalg.norm(parking_coords - rear_end_point, axis=1)
        close_indices = np.where(dists_from_rear <= movement_threshold)[0]
        candidate_frames = parking_df.iloc[close_indices]['frame']
        adjusted_rear_idx = close_indices[np.argmin(candidate_frames)]
        rear_end_point = parking_coords[adjusted_rear_idx]
        rear_end_frame = parking_df.iloc[adjusted_rear_idx]['frame']

        interpolated_point = front_end_point*(1-end_entry_ratio) + rear_end_point*end_entry_ratio

        dists_interpolated = np.linalg.norm(parking_coords - interpolated_point, axis=1)
        final_end_idx = np.argmin(dists_interpolated)
        final_end_point = parking_coords[final_end_idx]
        final_end_frame = parking_df.iloc[final_end_idx]['frame']

        # --- Start-of-Maneuver Detection ---
        peak_idx = None
        consecutive_post_peak_frames=0
        for idx in range(tA, len(driving_coords)-tA):
            r1 = driving_coords[idx] - driving_coords[idx-tA]
            r2 = driving_coords[idx+tA] - driving_coords[idx]
            if is_reversal(r1, r2):
                true_reversal = True
                for t in range(1, tA):
                    rA = driving_coords[idx] - driving_coords[max(0, idx-tA-t)]
                    rB = driving_coords[min(len(driving_coords)-1, idx+tA+t)] - driving_coords[idx]
                    if not is_reversal(rA, rB):
                        true_reversal = False
                        break
                if true_reversal and in_adjacent_zones(parking_coords_ground[0], driving_coords_ground[idx], all_line_params):      
                    peak_idx = idx    
            if peak_idx is not None and not is_reversal(r1, r2):
                consecutive_post_peak_frames+=1
            else:
                consecutive_post_peak_frames=0
            if consecutive_post_peak_frames>=tA:
                break

        if(peak_idx is None):
            consecutive_turn_frames = 0
            for idx in range(len(driving_coords)-1, 0, -1):
                aspect_ratio1 = driving_df.iloc[idx-1]['height']/driving_df.iloc[idx-1]['width']
                aspect_ratio2 = driving_df.iloc[idx]['height']/driving_df.iloc[idx]['width']
                if(round(aspect_ratio1, 2)!=round(aspect_ratio2, 2)):
                    consecutive_turn_frames+=1
                if(consecutive_turn_frames==tA):
                    peak_idx = idx

        if on_border(driving_df.iloc[peak_idx]):
            for idx in range(peak_idx, len(driving_df)):
                if not on_border(driving_df.iloc[idx]):
                    peak_idx = idx
                    break
                
        peak_point = driving_coords[peak_idx]
        peak_frame = driving_df.iloc[peak_idx]['frame']

        zone_based_start_idx = check_departure(driving_df, peak_idx, 0, all_line_params, all_line_endpoints)

        zone_based_start_frame = driving_df.iloc[zone_based_start_idx]["frame"]

        front_end_idx = len(driving_df)
        front_end_frame = vehicle_df.iloc[front_end_idx]['frame']

        plot_maneuver_analysis(video_ID, peak_idx, zone_based_start_idx)

        return front_end_frame, rear_end_frame, peak_frame, zone_based_start_frame, ann_start_frame, ann_end_frame

    if(maneuver_type==ManeuverType.EXT):
        # detect start of exit maneuver 
        front_start_point = driving_coords[0]
        front_start_frame = driving_df.iloc[0]['frame']
        dists_from_front = np.linalg.norm(parking_coords - front_start_point, axis=1)

        rear_start_idx = np.argmax(dists_from_front)
        rear_start_point = parking_coords[rear_start_idx]

        dists_from_rear = np.linalg.norm(parking_coords - rear_start_point, axis=1)
        close_indices = np.where(dists_from_rear <= movement_threshold)[0]
        candidate_frames = parking_df.iloc[close_indices]['frame']
        adjusted_rear_idx = close_indices[np.argmax(candidate_frames)]
        rear_start_point = parking_coords[adjusted_rear_idx]
        rear_start_frame = parking_df.iloc[adjusted_rear_idx]['frame']

        interpolated = front_start_point*(start_exit_ratio) + rear_start_point*(1-start_exit_ratio)

        dists_interpolated = np.linalg.norm(parking_coords - interpolated, axis=1)
        final_start_idx = np.argmin(dists_interpolated)

        # detect end of exit maneuver
        peak_idx = None
        consecutive_post_peak_frames=0
        for idx in range(tA, len(driving_coords)-tA):
            r1 = driving_coords[idx] - driving_coords[idx-tA]
            r2 = driving_coords[idx+tA] - driving_coords[idx]
            if is_reversal(r1, r2):
                true_reversal = True
                for t in range(1, tA):
                    rA = driving_coords[idx] - driving_coords[max(0, idx-tA-t)]
                    rB = driving_coords[min(len(driving_coords)-1, idx+tA+t)] - driving_coords[idx]
                    if not is_reversal(rA, rB):
                        true_reversal = False
                        break
                if true_reversal and in_adjacent_zones(parking_coords_ground[0], driving_coords_ground[idx], all_line_params):      
                    peak_idx = idx    
            if peak_idx is not None and not is_reversal(r1, r2):
                consecutive_post_peak_frames+=1
            else:
                consecutive_post_peak_frames=0
            if consecutive_post_peak_frames>=tA:
                break

        if(peak_idx is None):
            peak_idx = 0
            consecutive_turn_frames = 0
            for idx in range(len(driving_coords)-1):
                aspect_ratio1 = driving_df.iloc[idx]['height']/driving_df.iloc[idx]['width']
                aspect_ratio2 = driving_df.iloc[idx+1]['height']/driving_df.iloc[idx+1]['width']
                if(round(aspect_ratio1, 2)!=round(aspect_ratio2, 2)):
                    consecutive_turn_frames+=1
                if(consecutive_turn_frames==tA):
                    peak_idx = idx

        if on_border(driving_df.iloc[peak_idx]):
            for idx in range(peak_idx-1, -1, -1):
                if not on_border(driving_df.iloc[idx]):
                    peak_idx = idx
                    break
                
        peak_point = driving_coords[peak_idx]
        peak_frame = driving_df.iloc[peak_idx]['frame']

        bottom_peak_point = peak_point.copy()
        bottom_peak_point[1]+=driving_df.iloc[peak_idx]['height']//2

        zone_based_end_idx = check_departure(driving_df, peak_idx, len(driving_df)-1, all_line_params, all_line_endpoints)
        
        zone_based_end_frame = driving_df.iloc[zone_based_end_idx]["frame"]

        return front_start_frame, rear_start_frame, peak_frame, zone_based_end_frame, ann_start_frame, ann_end_frame

video_IDs = [
    '1ENT', '84EXT', '1002ENT', '2EXT', '96ENT', '11ENT', '83EXT', 
    '97ENT', '98ENT', '81EXT', '9FENT', '92FENT', '291ENT', '86EXT', 
    '80EXT', '85EXT', '92ENT', '21ENT', '90FENT', '290ENT', '4EXT', 
    '94ENT', '93ENT', '82EXT', '3FENT', '292ENT', '91ENT', '1003EXT', 
    '90ENT', '0EXT', '95ENT', '11FENT', '99ENT', '93FENT', '9ENT', 
    '3ENT', '1000EXT', '1001ENT'
]

for id in video_IDs:
    print(f"Processing {id}...")
    results = analyze_maneuversv2(id)
    print(f"Results for {id}: {results}")
