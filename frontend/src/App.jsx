import {
  useEffect,
  useRef,
  useState
} from "react";

import "./App.css";


function App() {

  const videoRef =
    useRef(null);

  const canvasRef =
    useRef(null);

  const socketRef =
    useRef(null);

  const streamRef =
    useRef(null);

  const intervalRef =
    useRef(null);


  // ========================================================
  // STATE
  // ========================================================

  const [cameraActive, setCameraActive] =
    useState(false);

  const [prediction, setPrediction] =
    useState("");

  const [confidence, setConfidence] =
    useState(0);

  const [status, setStatus] =
    useState("Camera is off");

  const [frames, setFrames] =
    useState(0);


  // ========================================================
  // START CAMERA
  // ========================================================

  async function startCamera() {

    try {

      setStatus(
        "Requesting camera..."
      );


      // ----------------------------------------------------
      // CAMERA
      // ----------------------------------------------------

      const stream =
        await navigator
          .mediaDevices
          .getUserMedia({

            video: {
              width: 640,
              height: 480
            },

            audio: false
          });


      streamRef.current =
        stream;


      if (videoRef.current) {

        videoRef.current.srcObject =
          stream;

        await videoRef.current.play();
      }


      // ----------------------------------------------------
      // WEBSOCKET
      // ----------------------------------------------------

      const BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL ||
  "ws://127.0.0.1:10000";

const socket =
  new WebSocket(
    `${BACKEND_URL}/ws/predict`
  );


      socketRef.current =
        socket;


      // ----------------------------------------------------
      // SOCKET OPEN
      // ----------------------------------------------------

      socket.onopen = () => {

        console.log(
          "WebSocket connected"
        );


        setStatus(
          "Camera active — show a sign"
        );


        // --------------------------------------------------
        // Send frames every 50 ms
        // --------------------------------------------------

        intervalRef.current =
          setInterval(
            sendFrame,
            50
          );
      };


      // ----------------------------------------------------
      // SOCKET MESSAGE
      // ----------------------------------------------------

      socket.onmessage = (
        event
      ) => {

        const data =
          JSON.parse(
            event.data
          );


        setFrames(
          data.frames || 0
        );


        // ==================================================
        // COLLECTING
        // ==================================================

        if (
          data.status ===
          "collecting"
        ) {

          setStatus(

            `Collecting frames: ${data.frames
            } / ${data.required || 20
            }`

          );

          return;
        }


        // ==================================================
        // NO HAND
        // ==================================================

        if (
          data.status ===
          "no_hand"
        ) {

          setStatus(
            "No hand detected"
          );

          return;
        }


        // ==================================================
        // FEATURE ERROR
        // ==================================================

        if (
          data.status ===
          "feature_error"
        ) {

          setStatus(
            "Feature extraction error"
          );

          return;
        }


        // ==================================================
        // UNCERTAIN
        // ==================================================

        if (
          data.status ===
          "uncertain"
        ) {

          setStatus(
            "Uncertain prediction"
          );


          setConfidence(
            data.confidence || 0
          );


          return;
        }


        if (data.status === "waiting") {
  setStatus("Sign detected — remove hand for next sign");
  setFrames(0);
  return;
}

if (data.status === "ready") {
  setStatus("Ready — show next sign");
  setFrames(0);
  return;
}
        // ==================================================
        // PREDICTION
        // ==================================================

        if (
          data.status ===
          "prediction"
        ) {

          setPrediction(
            data.text
          );


          setConfidence(
            data.confidence || 0
          );


          setStatus(
            "Sign detected"
          );


          speakPrediction(
            data.text,
            data.confidence
          );
        }

      };


      // ----------------------------------------------------
      // SOCKET ERROR
      // ----------------------------------------------------

      socket.onerror = (
        error
      ) => {

        console.error(
          "WebSocket error:",
          error
        );


        setStatus(
          "WebSocket error"
        );
      };


      // ----------------------------------------------------
      // SOCKET CLOSED
      // ----------------------------------------------------

      socket.onclose = () => {

        console.log(
          "WebSocket closed"
        );


        setStatus(
          "Connection closed"
        );
      };


      setCameraActive(
        true
      );

    }

    catch (error) {

      console.error(
        error
      );


      setStatus(
        "Could not access camera"
      );
    }
  }


  // ========================================================
  // SEND CAMERA FRAME
  // ========================================================

  function sendFrame() {

    if (
      !videoRef.current ||
      !canvasRef.current ||
      !socketRef.current
    ) {

      return;
    }


    if (
      socketRef.current.readyState !==
      WebSocket.OPEN
    ) {

      return;
    }


    const video =
      videoRef.current;

    const canvas =
      canvasRef.current;


    canvas.width =
      640;

    canvas.height =
      480;


    const context =
      canvas.getContext(
        "2d"
      );


    context.drawImage(

      video,

      0,
      0,

      640,
      480
    );


    canvas.toBlob(

      (blob) => {

        if (!blob) {

          return;
        }


        if (
          socketRef.current &&
          socketRef.current.readyState ===
          WebSocket.OPEN
        ) {

          socketRef.current.send(
            blob
          );
        }

      },

      "image/jpeg",

      0.7
    );
  }


  // ========================================================
  // SPEECH
  // ========================================================

  const lastSpokenRef =
    useRef("");


  const lastSpeechTimeRef =
    useRef(0);


  function speakPrediction(
    label,
    confidence
  ) {

    // Don't speak low-confidence predictions.
    if (
      confidence < 0.75
    ) {

      return;
    }


    const now =
      Date.now();


    // Don't repeatedly speak
    // the same prediction.

    if (

      label ===
      lastSpokenRef.current &&

      now -
      lastSpeechTimeRef.current <
      2500

    ) {

      return;
    }


    lastSpokenRef.current =
      label;


    lastSpeechTimeRef.current =
      now;


    const text =
      label.replaceAll(
        "_",
        " "
      );


    const utterance =
      new SpeechSynthesisUtterance(
        text
      );


    utterance.lang =
      "en-US";


    utterance.rate =
      0.9;


    window.speechSynthesis.cancel();


    window.speechSynthesis.speak(
      utterance
    );
  }


  // ========================================================
  // STOP CAMERA
  // ========================================================

  function stopCamera() {

    // ------------------------------------------------------
    // Stop frame interval
    // ------------------------------------------------------

    if (
      intervalRef.current
    ) {

      clearInterval(
        intervalRef.current
      );


      intervalRef.current =
        null;
    }


    // ------------------------------------------------------
    // Close WebSocket
    // ------------------------------------------------------

    if (
      socketRef.current
    ) {

      socketRef.current.close();


      socketRef.current =
        null;
    }


    // ------------------------------------------------------
    // Stop camera tracks
    // ------------------------------------------------------

    if (
      streamRef.current
    ) {

      streamRef.current
        .getTracks()
        .forEach(
          (track) =>
            track.stop()
        );


      streamRef.current =
        null;
    }


    // ------------------------------------------------------
    // Clear video
    // ------------------------------------------------------

    if (
      videoRef.current
    ) {

      videoRef.current.srcObject =
        null;
    }


    // ------------------------------------------------------
    // Reset UI
    // ------------------------------------------------------

    setCameraActive(
      false
    );


    setPrediction(
      ""
    );


    setConfidence(
      0
    );


    setFrames(
      0
    );


    setStatus(
      "Camera is off"
    );
  }


  // ========================================================
  // CLEANUP
  // ========================================================

  useEffect(() => {

    return () => {

      stopCamera();

    };

  }, []);


  // ========================================================
  // UI
  // ========================================================

  return (

    <div className="app">

      {/* ==================================================
          HEADER
          ================================================== */}

      <header className="header">

        <h1>
          ISL Translator
        </h1>


        <p>
          Real-time Indian Sign Language
          recognition
        </p>

      </header>


      {/* ==================================================
          MAIN
          ================================================== */}

      <main className="main">


        {/* =================================================
            CAMERA CARD
            ================================================= */}

        <div className="camera-card">

          <div className="video-container">

            <video
              ref={videoRef}
              className="video"
              muted
              playsInline
            />


            {!cameraActive && (

              <div className="camera-placeholder">

                <span>
                  Camera Off
                </span>

              </div>

            )}

          </div>


          {/* Hidden canvas */}

          <canvas
            ref={canvasRef}
            style={{
              display: "none"
            }}
          />


          {/* =================================================
              CONTROLS
              ================================================= */}

          <div className="controls">

            {!cameraActive ? (

              <button
                onClick={
                  startCamera
                }
              >
                Open Camera
              </button>

            ) : (

              <button
                className="stop-button"
                onClick={
                  stopCamera
                }
              >
                Stop Camera
              </button>

            )}

          </div>


          {/* =================================================
              STATUS
              ================================================= */}

          <div className="status">

            <span>
              Status:
            </span>


            <strong>
              {status}
            </strong>

          </div>


          {/* =================================================
              FRAME COUNTER
              ================================================= */}

          <div className="frame-info">

            Frames:
            {" "}
            {frames}
            {" / 20"}

          </div>

        </div>


        {/* =================================================
            RESULT CARD
            ================================================= */}

        <div className="result-card">

          <p className="result-title">
            Detected Sign
          </p>


          <div className="prediction">

            {prediction ||
              "Waiting..."}

          </div>


          <div className="confidence">

            Confidence:
            {" "}

            {(
              confidence * 100
            ).toFixed(1)}

            %

          </div>


          <p className="info">

            Show one complete sign
            to the camera.

          </p>

        </div>

      </main>

    </div>
  );
}


export default App;