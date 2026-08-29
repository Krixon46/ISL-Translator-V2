import { useEffect, useRef, useState, useCallback } from "react";
import {
  initializeMediaPipe,
  getHandLandmarker,
} from "./mediapipe";
import "./App.css";

const SUPPORTED_SIGNS = [
  "HELLO",
  "HOW_ARE_YOU",
  "ALRIGHT",
  "GOOD_MORNING",
  "GOOD_AFTERNOON",
  "GOOD_EVENING",
  "GOOD_NIGHT",
  "THANK_YOU",
  "PLEASED",
];

// MediaPipe 21 Landmark Connections
const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring finger
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm base
  [5, 9], [9, 13], [13, 17], [0, 5], [0, 17]
];

function getBackendWsUrl() {
  let rawUrl = (import.meta.env.VITE_BACKEND_URL || "").trim();

  if (!rawUrl) {
    if (typeof window !== "undefined" && (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")) {
      rawUrl = "ws://127.0.0.1:10000";
    } else {
      rawUrl = "wss://isl-translator-v2.onrender.com";
    }
  }

  let url = rawUrl;

  if (url.startsWith("http://")) {
    url = "ws://" + url.slice(7);
  } else if (url.startsWith("https://")) {
    url = "wss://" + url.slice(8);
  } else if (!url.startsWith("ws://") && !url.startsWith("wss://")) {
    url = "wss://" + url;
  }

  if (url.endsWith("/")) {
    url = url.slice(0, -1);
  }

  return `${url}/ws/predict`;
}

function drawHandSkeleton(ctx, landmarks, width, height, handIndex) {
  const isPrimary = handIndex === 0;
  const strokeColor = isPrimary ? "rgba(56, 189, 248, 0.85)" : "rgba(244, 114, 182, 0.85)";
  const jointColor = isPrimary ? "#38bdf8" : "#f472b6";
  const tipColor = "#facc15";

  // Draw bone connections with subtle glow
  ctx.save();
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 3.5;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.shadowColor = strokeColor;
  ctx.shadowBlur = 8;

  for (const [start, end] of HAND_CONNECTIONS) {
    const p1 = landmarks[start];
    const p2 = landmarks[end];
    if (!p1 || !p2) continue;

    ctx.beginPath();
    ctx.moveTo(p1.x * width, p1.y * height);
    ctx.lineTo(p2.x * width, p2.y * height);
    ctx.stroke();
  }
  ctx.restore();

  // Draw 21 keypoints
  for (let i = 0; i < landmarks.length; i++) {
    const landmark = landmarks[i];
    const x = landmark.x * width;
    const y = landmark.y * height;
    const isTip = [4, 8, 12, 16, 20].includes(i);
    const isWrist = i === 0;

    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, isTip ? 5.5 : isWrist ? 6 : 3.5, 0, Math.PI * 2);
    ctx.fillStyle = isTip ? tipColor : isWrist ? "#ffffff" : jointColor;
    ctx.shadowColor = isTip ? tipColor : jointColor;
    ctx.shadowBlur = 6;
    ctx.fill();

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
    ctx.restore();
  }
}

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const animationRef = useRef(null);
  const wsRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const isRunningRef = useRef(false);
  const reconnectTimerRef = useRef(null);

  // States
  const [isRunning, setIsRunning] = useState(false);
  const [mediaPipeReady, setMediaPipeReady] = useState(false);
  const [mediaPipeError, setMediaPipeError] = useState("");
  const [wsConnected, setWsConnected] = useState(false);
  const [handsDetected, setHandsDetected] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [requiredFrames, setRequiredFrames] = useState(20);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [statusText, setStatusText] = useState("Standby");

  // Initialize MediaPipe once on mount
  useEffect(() => {
    let isMounted = true;

    async function init() {
      try {
        await initializeMediaPipe();
        if (isMounted) {
          setMediaPipeReady(true);
        }
      } catch (err) {
        console.error("MediaPipe initialization error:", err);
        if (isMounted) {
          setMediaPipeError("Could not load local MediaPipe models.");
        }
      }
    }

    init();

    return () => {
      isMounted = false;
    };
  }, []);

  // Frame processing loop
  const startProcessingLoop = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d");

    function processFrame() {
      if (!video || video.paused || video.ended || !isRunningRef.current) return;

      const landmarker = getHandLandmarker();

      if (landmarker && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        // Adjust canvas dimensions to match video stream
        if (
          canvas.width !== video.videoWidth ||
          canvas.height !== video.videoHeight
        ) {
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Monotonically increasing timestamp for MediaPipe VIDEO mode
        let now = performance.now();
        if (now <= lastVideoTimeRef.current) {
          now = lastVideoTimeRef.current + 1;
        }
        lastVideoTimeRef.current = now;

        const results = landmarker.detectForVideo(video, now);
        const detected = results?.landmarks || [];
        const detectedCount = detected.length;

        setHandsDetected(detectedCount);

        // Draw skeleton overlay
        for (let i = 0; i < detected.length; i++) {
          drawHandSkeleton(ctx, detected[i], canvas.width, canvas.height, i);
        }

        // Send landmarks over WebSocket
        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN) {
          if (detectedCount > 0) {
            const payload = {
              type: "landmarks",
              timestamp: Math.round(now),
              hands: detected.map((hand) =>
                hand.map((pt) => ({
                  x: pt.x,
                  y: pt.y,
                  z: pt.z ?? 0,
                }))
              ),
            };
            ws.send(JSON.stringify(payload));
          } else {
            const payload = {
              type: "no_hand",
              timestamp: Math.round(now),
              hands: [],
            };
            ws.send(JSON.stringify(payload));
          }
        }
      }

      animationRef.current = requestAnimationFrame(processFrame);
    }

    animationRef.current = requestAnimationFrame(processFrame);
  }, []);

  const connectWsRef = useRef(null);

  // Connect WebSocket helper
  const connectWebSocket = useCallback(() => {
    if (!isRunningRef.current) return;

    if (
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    const wsUrl = getBackendWsUrl();
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
        setStatusText("Ready for signs");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (typeof data.frames === "number") {
            setFrameCount(data.frames);
          }
          if (typeof data.required === "number") {
            setRequiredFrames(data.required);
          }

          // Handle server statuses
          if (data.status === "ready" && data.frames === 0) {
            setPrediction("");
            setConfidence(0);
            setStatusText("Ready for signs");
          } else if (data.status === "no_hand") {
            setStatusText("No hands present");
          } else if (data.status === "collecting") {
            setStatusText(`Buffering: ${data.frames}/${data.required}`);
          } else if (data.status === "prediction" || data.status === "tracking") {
            if (data.text) {
              setPrediction(data.text);
            }
            if (typeof data.confidence === "number") {
              setConfidence(data.confidence);
            }
            setStatusText(data.status === "prediction" ? "Sign Detected" : "Holding Sign");
          } else if (data.status === "stabilizing") {
            setStatusText("Stabilizing prediction");
          } else if (data.status === "uncertain") {
            setStatusText("Hold sign steady");
          }
        } catch (err) {
          console.error("Error parsing backend message:", err);
        }
      };

      ws.onerror = () => {
        setWsConnected(false);
      };

      ws.onclose = () => {
        setWsConnected(false);
        if (isRunningRef.current) {
          reconnectTimerRef.current = setTimeout(() => {
            if (connectWsRef.current) {
              connectWsRef.current();
            }
          }, 2000);
        }
      };
    } catch (e) {
      console.warn("WebSocket connection error:", e);
      setWsConnected(false);
      if (isRunningRef.current) {
        reconnectTimerRef.current = setTimeout(() => {
          if (connectWsRef.current) {
            connectWsRef.current();
          }
        }, 2000);
      }
    }
  }, []);

  useEffect(() => {
    connectWsRef.current = connectWebSocket;
  }, [connectWebSocket]);

  // Stop camera & cleanup
  const stopCamera = useCallback(() => {
    isRunningRef.current = false;

    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    // 1. Cancel animation loop
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    // 2. Stop webcam tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    // 3. Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }

    // 4. Close WebSocket once
    if (wsRef.current) {
      if (
        wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING
      ) {
        wsRef.current.close();
      }
      wsRef.current = null;
    }

    // 5. Reset UI state
    setIsRunning(false);
    setWsConnected(false);
    setHandsDetected(0);
    setFrameCount(0);
    setPrediction("");
    setConfidence(0);
    setStatusText("Standby");
    lastVideoTimeRef.current = -1;
  }, []);

  // Start Camera
  const startCamera = async () => {
    try {
      setStatusText("Starting camera...");
      isRunningRef.current = true;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }

      setIsRunning(true);
      connectWebSocket();
      startProcessingLoop();
    } catch (err) {
      console.error("Camera access error:", err);
      setStatusText("Camera access denied");
      stopCamera();
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  const bufferPercentage = Math.min(100, Math.round((frameCount / requiredFrames) * 100));
  const confidencePercentage = Math.round(confidence * 100);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-badge">
          <span className="badge-dot" />
          ISL Neural Engine v2
        </div>
        <h1 className="app-title">Indian Sign Language Translator</h1>
        <p className="app-subtitle">
          Real-time AI gesture recognition powered by client-side MediaPipe landmark extraction and BiLSTM sequence modeling.
        </p>
      </header>

      {/* Main Grid */}
      <main className="main-grid">
        {/* Left: Camera & Vision Feed */}
        <section className={`glass-card ${handsDetected > 0 ? "active-tracking" : ""}`}>
          <div className="video-card-header">
            <h2 className="card-title">
              <span>📹</span> Live Video Feed
            </h2>
            <div className="status-indicator">
              <span
                className={`pulse-indicator ${
                  isRunning ? (handsDetected > 0 ? "live" : "warning") : "offline"
                }`}
              />
              <span style={{ fontSize: 13, color: "var(--text-muted)" }}>
                {isRunning
                  ? handsDetected > 0
                    ? `${handsDetected} Hand${handsDetected > 1 ? "s" : ""} (${handsDetected * 21} Keypoints)`
                    : "Searching for hands"
                  : "Inactive"}
              </span>
            </div>
          </div>

          <div className="video-viewport">
            <video
              ref={videoRef}
              className="webcam-feed"
              autoPlay
              playsInline
              muted
            />

            <canvas
              ref={canvasRef}
              className="skeleton-canvas"
            />

            {!isRunning && (
              <div className="viewport-overlay">
                <span className="overlay-icon">📷</span>
                <p className="overlay-text">Camera is currently stopped</p>
                <p style={{ fontSize: 13, color: "var(--text-dim)" }}>
                  Click &ldquo;Start Camera & Tracking&rdquo; below to begin gesture recognition
                </p>
              </div>
            )}
          </div>

          {/* Status Bar */}
          <div className="viewport-status-bar">
            <div className="status-indicator">
              <span className={`pulse-indicator ${wsConnected ? "live" : isRunning ? "warning" : "offline"}`} />
              <span>
                Backend:{" "}
                <strong style={{ color: wsConnected ? "var(--accent-green)" : "#fca5a5" }}>
                  {wsConnected ? "Connected" : isRunning ? "Connecting..." : "Disconnected"}
                </strong>
              </span>
            </div>
            <div>
              Status: <em style={{ color: "var(--text-main)" }}>{statusText}</em>
            </div>
          </div>

          {/* Action Button */}
          <div className="controls-container">
            {!isRunning ? (
              <button
                type="button"
                className="btn-primary"
                onClick={startCamera}
                disabled={!mediaPipeReady}
              >
                {!mediaPipeReady
                  ? mediaPipeError || "Initializing MediaPipe Engine..."
                  : "Start Camera & Tracking"}
              </button>
            ) : (
              <button
                type="button"
                className="btn-danger"
                onClick={stopCamera}
              >
                Stop Camera
              </button>
            )}
          </div>
        </section>

        {/* Right: Recognition Output */}
        <section className="glass-card recognition-panel">
          <h2 className="card-title">
            <span>✨</span> Recognized Translation
          </h2>

          {/* Big Hero Result Box */}
          <div className={`hero-prediction-box ${prediction ? "locked" : ""}`}>
            <span className="prediction-label-tag">Output Sign</span>
            <div className={`prediction-text ${!prediction ? "empty" : ""}`}>
              {prediction ? prediction.replace(/_/g, " ") : "Waiting for sign..."}
            </div>
          </div>

          {/* Meters */}
          <div className="metric-container">
            {/* Sliding Window Frame Progress */}
            <div className="metric-row">
              <div className="metric-header">
                <span className="metric-title">Temporal Window Buffer</span>
                <span className="metric-value">
                  {frameCount} / {requiredFrames} frames ({bufferPercentage}%)
                </span>
              </div>
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{ width: `${bufferPercentage}%` }}
                />
              </div>
            </div>

            {/* Confidence Score */}
            <div className="metric-row">
              <div className="metric-header">
                <span className="metric-title">Model Confidence</span>
                <span className="metric-value">
                  {confidence > 0 ? `${(confidence * 100).toFixed(1)}%` : "—"}
                </span>
              </div>
              <div className="progress-track">
                <div
                  className={`progress-fill ${
                    confidencePercentage >= 65 ? "green" : confidencePercentage > 0 ? "amber" : ""
                  }`}
                  style={{ width: `${confidencePercentage}%` }}
                />
              </div>
            </div>
          </div>

          {/* Supported Vocabulary */}
          <div className="signs-section">
            <span className="signs-title">Supported Vocabulary (9 Classes)</span>
            <div className="signs-grid">
              {SUPPORTED_SIGNS.map((sign) => {
                const isActive = prediction === sign;
                return (
                  <span
                    key={sign}
                    className={`sign-chip ${isActive ? "active" : ""}`}
                  >
                    {sign.replace(/_/g, " ")}
                  </span>
                );
              })}
            </div>
          </div>

          {/* Guidance */}
          <div className="info-callout">
            💡 <strong>Quick Tip:</strong> Face the camera in good lighting. Perform a sign and hold it for ~1 second (20 frames). Lowering your hands automatically resets the sequence buffer.
          </div>
        </section>
      </main>
    </div>
  );
}