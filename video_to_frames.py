import cv2

def video_to_frames(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Loop through each frame and save it as an image
    for frame_number in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {frame_number}")
            break

        # Save the frame as an image
        frame_filename = f"{output_path}/frame_{frame_number:04d}.jpg"
        cv2.imwrite(frame_filename, frame)

    # Release the video capture object
    cap.release()

# Example usage
video_path = "videos/trymefirst.mp4"
output_path = "processed_videos/trymefirst/frames"
video_to_frames(video_path, output_path)