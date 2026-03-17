# maneuver_heuristics25

identify_parking_locations.ipynb: Take in a video path and annotate where the parking region and parking lines are on the screen for some frame. Store the annotation information in CSV files.

manual_timing.py: a tool that will help you gather manual start and end frames and store them. 

track_generation.py: take a clip of footage and click on screen to indicate the vehicle you'd like to track. Store trajectory data in a CSV file. 

analyze_maneuvers.py: use the annotation information for the parking region and parking lines and the trajectory data and generate predictions for each video you have access to. It's easy to take the printed output and store it in a CSV file to be used easily.

demo_algorithm.py: this tool assumes that you have all the predictions and annotations across videos stored in their own CSV files. It simply visualizes the tracking dot of a vehicle on a plot moving concurrently with the vehicle on the video. 
