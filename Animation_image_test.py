import cv2
import numpy as np
import time

enable_avarage = 1
# Set the desired frame rate (e.g., 10 frames per second)
desired_frame_rate = 10
frame_delay = int(1000 / desired_frame_rate)  # Calculate delay in milliseconds

# Initialize variables for FPS calculation
start_time = time.time()
frame_count = [0]  # Use a list to store frame_count as a mutable object
fps = 0

# Create a function to generate and return random 100x100 RGB images
def generate_random_image():
    return np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    # return np.full((100, 100, 3), 10, dtype=int)

def average_image(current_image, new_image, num_of_frames_till_now):
    # print(current_image*num_of_frames_till_now + new_image)
    return ((current_image*num_of_frames_till_now + new_image) / (num_of_frames_till_now + 1))

image = generate_random_image()

while True:

    # Generate a random RGB image
    new_image = generate_random_image()   

    if enable_avarage and frame_count[0]!=0:
        image = average_image(image, new_image, frame_count[0])
    else:
        image = new_image

    # Add FPS text overlay on the image
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(image, fps_text, (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)  # Adjust the position, font, and size

    # Display the image
    cv2.namedWindow("Random test Image", cv2.WINDOW_NORMAL)  # Create a resizable window
    print(image)
    cv2.imshow("Random test Image", image.astype(np.uint8))

   

    # Calculate FPS
    frame_count[0] += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time != 0:
        fps = frame_count[0] / elapsed_time
    print(elapsed_time)

    

    # Introduce a delay to achieve the desired frame rate
    if elapsed_time < (frame_count[0]/desired_frame_rate):
        time.sleep((frame_count[0]/desired_frame_rate) - elapsed_time)
    
    

    # Press 'q' to quit the stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all OpenCV windows
cv2.destroyAllWindows()
