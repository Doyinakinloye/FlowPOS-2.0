import cv2
import time
import numpy as np

class CameraStream:
    def __init__(self, stream_url=None):
        """
        Initialize camera stream
        stream_url: DroidCam URL or None for webcam
        """
        self.stream_url = stream_url
        self.cap = None
        self.connected = False

    def connect(self, retries=3):
        """
        Connect to the camera stream
        retries: number of connection attempts
        """
        print(f"Connecting to camera...")

        for attempt in range(retries):
            try:
                # If no URL use laptop webcam
                if not self.stream_url:
                    self.cap = cv2.VideoCapture(0)
                else:
                    self.cap = cv2.VideoCapture(self.stream_url)

                # Optimize stream performance
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)    

                # Give it 2 seconds to connect
                time.sleep(1)

                if self.cap.isOpened():
                    # Try reading a test frame
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.connected = True
                        print("Camera connected successfully!")
                        return True
                    else:
                        print(f"Connected but no frame received! Attempt {attempt + 1}/{retries}")
                        self.cap.release()
                else:
                    print(f"Failed to open camera! Attempt {attempt + 1}/{retries}")

            except Exception as e:
                print(f"Connection error: {e} | Attempt {attempt + 1}/{retries}")

            # Wait before retrying
            if attempt < retries - 1:
                print("Retrying in 3 seconds...")
                time.sleep(3)

        # All attempts failed
        self.connected = False
        print("\n" + "=" * 50)
        print("CAMERA CONNECTION FAILED!")
        print("Please check the following:")
        print("1. Is DroidCam app open on your phone?")
        print("2. Is your laptop connected to phone hotspot?")
        print("3. Is the IP address correct?")
        print("4. Try restarting DroidCam app")
        print("=" * 50)
        return False

    def reconnect(self, retries=5, delay=3):
        """
        Reconnect if stream drops during session
        retries: number of reconnection attempts
        delay: seconds between attempts
        """
        print("\nStream dropped! Attempting to reconnect...")

        for attempt in range(retries):
            print(f"Reconnecting... Attempt {attempt + 1}/{retries}")

            try:
                if self.cap:
                    self.cap.release()

                time.sleep(delay)
                self.cap = cv2.VideoCapture(self.stream_url)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.connected = True
                        print("Reconnected successfully!")
                        return True

            except Exception as e:
                print(f"Reconnection error: {e}")

        print("Failed to reconnect after all attempts!")
        self.connected = False
        return False

    def read_frame(self):
            """
            Read latest frame from stream
            """

            if not self.connected:
                return None

            try:

                # Skip old buffered frames
                if self.stream_url:
                    self.cap.grab()

                ret, frame = self.cap.read()

                if not ret or frame is None:

                    if self.stream_url:
                        print("Frame read failed. Reconnecting...")

                        if self.reconnect():
                            ret, frame = self.cap.read()

                            if ret:
                                return frame

                    return None

                return frame

            except Exception as e:
                print(f"Frame read error: {e}")
                return None

    def release(self):
        """Release the camera stream"""
        try:
            if self.cap:
                self.cap.release()
            self.connected = False
            print("Camera stream released!")
        except Exception as e:
            print(f"Release error: {e}")

    def test_stream(self):
        """
        Test the camera stream by showing
        live feed for 10 seconds
        """
        print("Testing camera stream for 10 seconds...")
        print("Press 'q' to quit early")

        if not self.connect():
            print("\nTip: Make sure DroidCam is open on your phone")
            print("Tip: Check that laptop is connected to phone hotspot")
            return False

        start_time = time.time()
        frame_count = 0
        failed_frames = 0

        while True:
            frame = self.read_frame()

            if frame is None:
                failed_frames += 1
                if failed_frames > 10:
                    print("Too many failed frames! Stopping test.")
                    break
                time.sleep(0.1)
                continue

            # Reset failed frame counter on success
            failed_frames = 0
            frame_count += 1

            # Resize frame for display
            frame = cv2.resize(frame, (640, 480))

            # Show info on frame
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            remaining = max(0, 10 - int(elapsed))

            cv2.putText(frame, f"FPS: {fps:.1f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Time remaining: {remaining}s",
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 255), 2)

            cv2.putText(frame, "Press Q to quit",
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255, 255, 255), 2)

            cv2.putText(frame, "Smart Inventory System",
                       (10, 460), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 200, 255), 2)

            # Show the frame
            cv2.imshow("Smart Inventory - Camera Test", frame)

            # Quit after 10 seconds or if Q pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if time.time() - start_time > 10:
                break

        print(f"\nCamera Test Results:")
        print(f"Total frames received: {frame_count}")
        print(f"Failed frames: {failed_frames}")
        print(f"Average FPS: {frame_count / max(1, time.time() - start_time):.1f}")

        self.release()
        cv2.destroyAllWindows()
        print("Camera test complete!")
        return True


# ============================================
# HELPER FUNCTION — GET STREAM URL
# ============================================
def get_stream_url():
    """
    Get DroidCam stream URL from user.
    Press Enter with nothing typed to use the laptop''s built-in webcam instead.
    Returns the full stream URL, or None for webcam.
    """
    print("\n" + "=" * 50)
    print("   CAMERA CONNECTION SETUP")
    print("=" * 50)
    print("Steps:")
    print("1. To use DroidCam: open the app on your phone, connect")
    print("   your laptop to the same network, and enter its IP.")
    print("2. To use your laptop''s built-in webcam instead, just")
    print("   press Enter without typing anything.")
    print("=" * 50)

    ip = input("\nEnter IP Address (or press Enter for webcam): ").strip()

    if not ip:
        print("\nUsing laptop webcam.")
        return None

    port = "8080"  # IP Webcam default port
    url = f"http://{ip}:{port}/video"
    print(f"\nStream URL: {url}")
    return url


# ============================================
# TEST THE CAMERA STREAM
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("   SMART INVENTORY - CAMERA STREAM TEST")
    print("=" * 50)

    # Get IP address from user
    STREAM_URL = get_stream_url()

    # Create and test camera
    camera = CameraStream(stream_url=STREAM_URL)
    camera.test_stream()
