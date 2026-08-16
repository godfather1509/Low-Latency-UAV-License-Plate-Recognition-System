import asyncio                  # For async event loops and scheduling coroutines
import base64                   # For encoding frames to base64 strings
import cv2                      # OpenCV to capture and encode video frames
import threading                # To run capture loop in a separate thread

from channels.generic.websocket import AsyncWebsocketConsumer  # WebSocket consumer base class


class FrameBroadcaster:
    def __init__(self):
        self.clients = set()               # Set to keep track of all connected clients
        self.lock = threading.Lock()      # Thread lock to protect shared data access
        self.running = False               # Flag to control capture thread lifecycle
        self.capture_thread = None        # Background thread that captures frames

    def add_client(self, client):
        # Add a client to the clients set
        with self.lock:                   # Acquire lock for thread safety
            self.clients.add(client)      # Add the new client

            # If capture thread not running, start it
            if not self.running:
                self.running = True
                # Create and start the background capture thread as daemon
                self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.capture_thread.start()

    def remove_client(self, client):
        # Remove a client from the clients set
        with self.lock:
            self.clients.discard(client)  # Remove client safely (no error if missing)

            # If no clients left, signal the capture thread to stop
            if not self.clients:
                self.running = False

    def _capture_loop(self):
        # Background thread method that captures frames continuously
        cap = cv2.VideoCapture(0)         # Open default camera (index 0)

        while self.running:
            ret, frame = cap.read()       # Read one frame from the camera
            if not ret:
                continue                  # If capture failed, skip and try again

            # Encode frame as JPEG image in memory
            _, buffer = cv2.imencode('.jpg', frame)

            # Convert JPEG bytes to base64 encoded string for transmission
            encoded = base64.b64encode(buffer).decode('utf-8')

            # Copy clients list while holding lock, to avoid race conditions
            with self.lock:
                clients_copy = list(self.clients)

            # Send the encoded frame to each connected client asynchronously
            for client in clients_copy:
                try:
                    # Run the async send_frame coroutine in the client's event loop thread safely
                    asyncio.run_coroutine_threadsafe(client.send_frame(encoded), client.loop)
                except Exception:
                    pass                    # Ignore exceptions (e.g. client disconnected)

            # Small wait to roughly limit to ~30 frames per second
            cv2.waitKey(33)

        cap.release()                     # Release camera resource when stopping


# Create a global singleton broadcaster instance
broadcaster = FrameBroadcaster()


class VideoStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Called when a client opens a WebSocket connection
        await self.accept()              # Accept the connection (necessary to start receiving/sending)

        self.send_queue = asyncio.Queue()       # Async queue for frames to send to client
        self.loop = asyncio.get_running_loop() # Store current event loop to schedule tasks from thread

        broadcaster.add_client(self)             # Register this client with the FrameBroadcaster

        # Start a background task that will send frames from queue to the client WebSocket
        self.send_task = asyncio.create_task(self._send_frames())

    async def disconnect(self, close_code):
        # Called when the client disconnects or connection closes
        broadcaster.remove_client(self)           # Remove client from broadcaster's list

        self.send_task.cancel()                    # Cancel the background frame sending task
        try:
            await self.send_task                   # Await task cancellation gracefully
        except asyncio.CancelledError:
            pass                                  # Ignore cancellation error

    async def send_frame(self, encoded_frame):
        # Called by broadcaster from capture thread to enqueue a new frame for sending
        await self.send_queue.put(encoded_frame)  # Put the base64 frame string in the queue

    async def _send_frames(self):
        # Background coroutine that sends frames from the queue over the WebSocket
        try:
            while True:
                frame = await self.send_queue.get()  # Wait for the next frame to send
                await self.send(text_data=frame)     # Send frame as WebSocket text message
                # No sleep here, rely on capture FPS for pacing to reduce latency
        except asyncio.CancelledError:
            pass                                  # Handle graceful shutdown on disconnect
