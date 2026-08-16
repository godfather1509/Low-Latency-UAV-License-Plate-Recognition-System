import React, { useEffect, useRef } from "react";

const VideoStream = () => {
  const imgRef = useRef(null);

  useEffect(() => {
    const socket = new WebSocket("ws://127.0.0.1:8000/ws/video/");

    socket.onopen = () => {
      console.log("✅ WebSocket connection opened");
    };

    socket.onerror = (error) => {
      console.error("❌ WebSocket error:", error);
    };

    socket.onmessage = (event) => {
      try {
        // If backend sends plain base64 string:
        if (!event.data.startsWith("{")) {
          if (imgRef.current) {
            imgRef.current.src = `data:image/jpeg;base64,${event.data}`;
          }
          return;
        }

        // If backend sends JSON: {"image": "base64string"}
        const data = JSON.parse(event.data);
        if (data.image && imgRef.current) {
          imgRef.current.src = `data:image/jpeg;base64,${data.image}`;
        }
      } catch (err) {
        console.error("⚠️ Error parsing message", err);
      }
    };

    socket.onclose = (event) => {
      console.log("❌ WebSocket closed:", event);
    };

    return () => {
      socket.close();
    };
  }, []);

  return (
    <div>
      <h2>Live Stream</h2>
      <img ref={imgRef} alt="Live Feed" width="640" height="480" />
    </div>
  );
};

export default VideoStream;
