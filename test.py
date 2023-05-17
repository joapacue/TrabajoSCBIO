import cv2
import os
import time

# create a VideoCapture object to capture frames from the webcam
cap = cv2.VideoCapture(0)

# check if the webcam is opened successfully
if not cap.isOpened():
    print("Error opening webcam")
else:
    # create the output directory if it doesn't exist
    output_dir = 'captured_frames'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # set the initial timestamp and frame counter
    timestamp = int(time.time())
    frame_counter = 0

    # loop over frames from the webcam
    while True:
        # read a frame from the webcam
        ret, frame = cap.read()

        # check if the frame was successfully read
        if not ret:
            print("Error reading frame")
            break

        # check if it's time to save a new frame
        current_timestamp = int(time.time())
        if current_timestamp > timestamp:
            # construct the output filename
            output_filename = os.path.join(output_dir, f'frame_{frame_counter:04}.jpg')

            # save the frame to disk
            cv2.imwrite(output_filename, frame)

            # update the timestamp and frame counter
            timestamp = current_timestamp
            frame_counter += 0.5

        # display the frame in a window
        cv2.imshow("Webcam", frame)

        # check if the user has pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
