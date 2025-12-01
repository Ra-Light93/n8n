import cv2
import numpy as np
import mss
import time

def select_screen_region():
    """
    Displays the screen and allows the user to select a region of interest (ROI).
    Returns the BBOX dictionary for the selected region.
    """
    print("Taking a screenshot for region selection...")
    with mss.mss() as sct:
        # Get a screenshot of the primary monitor
        sct_img = sct.grab(sct.monitors[1])
        # Convert to an OpenCV image
        selection_img = np.array(sct_img)
        selection_img = cv2.cvtColor(selection_img, cv2.COLOR_BGRA2BGR)

    print("\nA window has appeared with a screenshot of your screen.")
    print("Click and drag your mouse to draw a rectangle around the video you want to record.")
    print("Once you are happy with the selection, press ENTER or SPACE.")
    print("Press 'c' to cancel the selection.")

    # Allow user to select ROI
    roi = cv2.selectROI("Select Recording Area - Press ENTER to confirm", selection_img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if not any(roi):
        print("No region selected. Exiting.")
        return None

    # roi is a tuple (x, y, w, h)
    # Convert to the BBOX format used by mss
    bbox = {"top": roi[1], "left": roi[0], "width": roi[2], "height": roi[3]}
    return bbox

# --- Configuration ---
BBOX = select_screen_region()

if BBOX:
    # Video output settings
    OUTPUT_FILENAME = "output.mp4"
    FPS = 30.0  # Desired frames per second

    # --- Initialization ---
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, FPS, (BBOX['width'], BBOX['height']))

    # Create an MSS instance
    sct = mss.mss()

    print(f"\nRecording selected screen region: {BBOX}")
    print("A live preview window will be shown.")
    print("Press 'q' on the preview window to stop recording.")

    # --- Recording Loop ---
    try:
        last_time = time.time()
        while True:
            # Grab screen frame using MSS
            img = sct.grab(BBOX)

            # Convert the MSS BGRA image to a NumPy array
            frame = np.array(img)

            # Convert BGRA to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Write the frame to the video file
            writer.write(frame_bgr)

            # Display the recording preview
            cv2.imshow('Recording Preview - Press "q" to quit', frame_bgr)

            # Check for 'q' key press to quit (waits 1 ms)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # --- Frame rate limiting ---
            sleep_time = (1.0 / FPS) - (time.time() - last_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()

    except KeyboardInterrupt:
        # This block is kept as a fallback
        print("\nRecording stopped by user (Ctrl+C).")

    finally:
        # --- Cleanup ---
        writer.release()
        cv2.destroyAllWindows()
        print(f"Video saved as {OUTPUT_FILENAME}")
