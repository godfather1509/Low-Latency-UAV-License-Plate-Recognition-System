# this is not part of website 


import cv2                      # OpenCV for capturing video frames
import base64                   # To encode frames as base64 strings for transmission
import threading                # To run the capture loop in a separate thread
import asyncio                  # For async event loop and scheduling async tasks


class FrameBroadcaster:
    def __init__(self):
        self.clients = set()             # Set to hold all connected clients
        self.lock = threading.Lock()    # Lock to synchronize access to shared data across threads
        self.running = False             # Flag to control the capture thread’s running state
        self.capture_thread = None      # Background thread that captures video frames

    def add_client(self, client):
        # Add a new client and start the capture thread if not running
        with self.lock:
            self.clients.add(client)     # Add client to set
            if not self.running:
                self.running = True
                # Start the capture loop thread as a daemon (exits with main program)
                self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.capture_thread.start()

    def remove_client(self, client):
        # Remove a client and stop capture thread if no clients remain
        with self.lock:
            self.clients.discard(client)  # Remove client from set (safe even if not present)
            if not self.clients:
                self.running = False       # Signal capture loop to stop when no clients left

    def _capture_loop(self):
        # This is the background thread’s main loop to capture frames and send to clients

        # Open video capture using MSMF backend for Windows (can switch to other backends if needed)
        cap = cv2.VideoCapture(1, cv2.CAP_MSMF)

        while self.running:
            ret, frame = cap.read()          # Capture one frame from the camera
            if not ret:
                continue                    # Skip if frame not captured successfully

            # Encode the frame as JPEG image bytes in memory
            _, buffer = cv2.imencode('.jpg', frame)

            # Convert encoded bytes to base64 string for safe transmission over WebSocket
            encoded = base64.b64encode(buffer).decode('utf-8')

            # Lock to safely access clients set and schedule sending frame to each
            with self.lock:
                # Iterate over a copy of clients to avoid issues if clients change mid-iteration
                for client in list(self.clients):
                    try:
                        # Schedule coroutine to put the frame in the client’s async queue
                        # run_coroutine_threadsafe lets us run async code safely from this thread
                        asyncio.run_coroutine_threadsafe(client.send_queue.put(encoded), client.loop)
                    except Exception:
                        # Silently ignore any errors (e.g. client disconnected)
                        pass

            # Wait roughly 1/30th of a second (~30 FPS) before capturing next frame
            threading.Event().wait(1 / 30)

        cap.release()  # Release the camera resource when done
