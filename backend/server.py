import json
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from features import landmarks_to_features
from model import SignBiLSTM


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "best_model.pt"
LABELS_PATH = BASE_DIR / "labels.json"
SCALER_PATH = BASE_DIR / "scaler.pkl"

MEDIAPIPE_MODEL = (
    BASE_DIR
    / "assets"
    / "hand_landmarker.task"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Model was trained using 20 frames.
SEQUENCE_LENGTH = 20

# Minimum confidence required for a prediction.
CONFIDENCE_THRESHOLD = 0.60

# Confidence required before we consider a prediction stable.
STABLE_CONFIDENCE = 0.65

# Number of consecutive similar predictions required
# before accepting a sign as stable.
STABLE_PREDICTIONS_REQUIRED = 3

# Number of consecutive frames without hands required
# to consider the current sign finished.
RELEASE_FRAMES_REQUIRED = 5

# Frontend sends approximately one frame every 50 ms.
FRAME_INTERVAL_MS = 50


# ============================================================
# LOAD SCALER
# ============================================================

if not SCALER_PATH.exists():

    raise FileNotFoundError(
        f"Scaler not found: {SCALER_PATH}"
    )


scaler = joblib.load(
    SCALER_PATH
)

print("Scaler loaded successfully.")


# ============================================================
# LOAD LABELS
# ============================================================

if not LABELS_PATH.exists():

    raise FileNotFoundError(
        f"Labels file not found: {LABELS_PATH}"
    )


with open(
    LABELS_PATH,
    "r"
) as f:

    LABELS = json.load(f)


print(
    "Loaded labels:",
    LABELS
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Using device:",
    device
)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)


# ============================================================
# CREATE MODEL
# ============================================================

model = SignBiLSTM(
    input_size=126,
    hidden_size=128,
    num_layers=2,
    num_classes=len(LABELS)
)


# ============================================================
# LOAD WEIGHTS
# ============================================================

model.load_state_dict(
    state_dict
)

model.to(device)

model.eval()


print()
print("==============================")
print("MODEL LOAD SUCCESS")
print("==============================")

print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)

print(
    "Input features: 126"
)

print(
    "Sequence length:",
    SEQUENCE_LENGTH
)

print(
    "Classes:",
    len(LABELS)
)

print(
    "Labels:",
    LABELS
)

print("==============================")
print()


# ============================================================
# MEDIAPIPE
# ============================================================

mp_tasks = mp.tasks

BaseOptions = mp_tasks.BaseOptions

HandLandmarker = (
    mp_tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp_tasks.vision.HandLandmarkerOptions
)

RunningMode = (
    mp_tasks.vision.RunningMode
)




# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()


app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://isl-translator-v2-pink.vercel.app/"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws/predict")
async def predict(
    websocket: WebSocket
):

    await websocket.accept()

        # ========================================================
    # CREATE MEDIAPIPE LANDMARKER FOR THIS CONNECTION
    # ========================================================

    options = HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(
                MEDIAPIPE_MODEL.resolve()
            )
        ),

        running_mode=RunningMode.VIDEO,

        num_hands=2,

        min_hand_detection_confidence=0.5,

        min_hand_presence_confidence=0.5,

        min_tracking_confidence=0.5
    )

    landmarker = (
        HandLandmarker
        .create_from_options(
            options
        )
    )

    print()
    print("==============================")
    print("CLIENT CONNECTED")
    print("==============================")


    # ========================================================
    # FRAME BUFFER
    #
    # This is the sliding window.
    #
    # Example:
    #
    # 1  2  3  ... 20
    #       ↓ new frame
    # 2  3  4  ... 21
    #       ↓ new frame
    # 3  4  5  ... 22
    #
    # The deque automatically removes the oldest frame.
    # ========================================================

    sequence = deque(
        maxlen=SEQUENCE_LENGTH
    )


    # ========================================================
    # MEDIAPIPE TIMESTAMP
    #
    # IMPORTANT:
    # Each WebSocket connection gets its own timestamp.
    #
    # This prevents:
    #
    # ValueError:
    # Input timestamp must be monotonically increasing
    # ========================================================

    timestamp_ms = 0


    # ========================================================
    # HAND RELEASE TRACKING
    # ========================================================

    no_hand_count = 0


    # ========================================================
    # PREDICTION STABILIZATION
    # ========================================================

    last_prediction_index = None

    stable_prediction_count = 0

    accepted_prediction = None


    # ========================================================
    # DEBUG COUNTER
    # ========================================================

    prediction_window_count = 0


    try:

        while True:

            # ==================================================
            # RECEIVE FRAME
            # ==================================================

            data = await websocket.receive_bytes()


            # ==================================================
            # JPEG → OPENCV
            # ==================================================

            np_data = np.frombuffer(
                data,
                dtype=np.uint8
            )


            frame = cv2.imdecode(
                np_data,
                cv2.IMREAD_COLOR
            )


            if frame is None:

                continue


            # ==================================================
            # BGR → RGB
            # ==================================================

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # ==================================================
            # MEDIAPIPE IMAGE
            # ==================================================

            mp_image = mp.Image(

                image_format=
                    mp.ImageFormat.SRGB,

                data=frame_rgb
            )


            # ==================================================
            # TIMESTAMP
            # ==================================================

            timestamp_ms += FRAME_INTERVAL_MS


            # ==================================================
            # HAND DETECTION
            # ==================================================

            result = (
                landmarker
                .detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )


            hand_count = len(
                result.hand_landmarks
            )


            print(
                f"MediaPipe: {hand_count} hand(s) detected"
            )


            # ==================================================
            # NO HAND DETECTED
            # ==================================================

            if hand_count == 0:

                no_hand_count += 1


                # ------------------------------------------------
                # If enough frames contain no hands,
                # the current sign is finished.
                # ------------------------------------------------

                if (
                    no_hand_count
                    >= RELEASE_FRAMES_REQUIRED
                ):

                    # --------------------------------------------
                    # FULL RESET
                    # --------------------------------------------

                    if len(sequence) > 0:

                        print()
                        print(
                            "=============================="
                        )

                        print(
                            "HAND RELEASED"
                        )

                        print(
                            "Resetting sequence..."
                        )

                        print(
                            "=============================="
                        )
                        print()


                    sequence.clear()


                    last_prediction_index = None

                    stable_prediction_count = 0

                    accepted_prediction = None

                    prediction_window_count = 0


                    await websocket.send_json({

                        "status":
                            "ready",

                        "text":
                            "",

                        "confidence":
                            0,

                        "frames":
                            0,

                        "required":
                            SEQUENCE_LENGTH
                    })


                    continue


                # ------------------------------------------------
                # Hand has disappeared temporarily.
                # ------------------------------------------------

                await websocket.send_json({

                    "status":
                        "no_hand",

                    "text":
                        "",

                    "confidence":
                        0,

                    "frames":
                        len(sequence),

                    "required":
                        SEQUENCE_LENGTH
                })


                continue


            # ==================================================
            # HAND FOUND
            # ==================================================

            no_hand_count = 0


            # ==================================================
            # EXTRACT FEATURES
            # ==================================================

            features = (
                landmarks_to_features(
                    result
                )
            )


            # ==================================================
            # FEATURE SIZE CHECK
            # ==================================================

            if features.shape[0] != 126:

                print(
                    "ERROR: Expected 126 features,"
                    f" got {features.shape[0]}"
                )


                await websocket.send_json({

                    "status":
                        "feature_error",

                    "text":
                        "Feature size error",

                    "confidence":
                        0,

                    "frames":
                        len(sequence),

                    "required":
                        SEQUENCE_LENGTH
                })


                continue


            # ==================================================
            # SCALE FEATURES
            # ==================================================

            features_scaled = scaler.transform(
                features.reshape(1, -1)
            )[0]


            features_scaled = (
                features_scaled.astype(
                    np.float32
                )
            )


            # ==================================================
            # ADD NEW FRAME
            # ==================================================

            sequence.append(
                features_scaled
            )


            # ==================================================
            # STILL COLLECTING
            # ==================================================

            if len(sequence) < SEQUENCE_LENGTH:

                await websocket.send_json({

                    "status":
                        "collecting",

                    "text":
                        "",

                    "confidence":
                        0,

                    "frames":
                        len(sequence),

                    "required":
                        SEQUENCE_LENGTH
                })


                continue


            # ==================================================
            # SLIDING WINDOW READY
            # ==================================================

            prediction_window_count += 1


            # ==================================================
            # CONVERT TO NUMPY
            # ==================================================

            X = np.array(
                sequence,
                dtype=np.float32
            )


            # Shape:
            #
            # 20 × 126
            #
            # ↓
            #
            # 1 × 20 × 126

            X = torch.tensor(
                X,
                dtype=torch.float32
            )


            X = X.unsqueeze(0)

            X = X.to(device)


            # ==================================================
            # MODEL PREDICTION
            # ==================================================

            with torch.no_grad():

                outputs = model(X)


                probabilities = (
                    torch.softmax(
                        outputs,
                        dim=1
                    )
                )


                confidence, prediction = (
                    probabilities.max(
                        dim=1
                    )
                )


            predicted_index = (
                prediction.item()
            )


            confidence_value = (
                confidence.item()
            )


            # ==================================================
            # SAFETY CHECK
            # ==================================================

            if predicted_index >= len(LABELS):

                print(
                    "ERROR: Invalid prediction index:",
                    predicted_index
                )

                continue


            predicted_label = (
                LABELS[predicted_index]
            )


            # ==================================================
            # LOW CONFIDENCE
            # ==================================================

            if (
                confidence_value
                < CONFIDENCE_THRESHOLD
            ):

                print(
                    f"Window {prediction_window_count}: "
                    f"Uncertain → "
                    f"{predicted_label} "
                    f"| {confidence_value:.2f}"
                )


                # --------------------------------------------
                # Don't reset the sequence.
                #
                # The deque continues sliding.
                # --------------------------------------------

                await websocket.send_json({

                    "status":
                        "uncertain",

                    "text":
                        "",

                    "confidence":
                        confidence_value,

                    "frames":
                        SEQUENCE_LENGTH,

                    "required":
                        SEQUENCE_LENGTH
                })


                continue


            # ==================================================
            # STABILIZATION
            # ==================================================

            if (
                predicted_index
                ==
                last_prediction_index
            ):

                stable_prediction_count += 1

            else:

                last_prediction_index = (
                    predicted_index
                )

                stable_prediction_count = 1


            print(
                f"Window {prediction_window_count}: "
                f"{predicted_label} "
                f"| {confidence_value:.2f} "
                f"| stable "
                f"{stable_prediction_count}/"
                f"{STABLE_PREDICTIONS_REQUIRED}"
            )


            # ==================================================
            # CHECK WHETHER PREDICTION IS STABLE
            # ==================================================

            if (

                stable_prediction_count
                >=
                STABLE_PREDICTIONS_REQUIRED

                and

                confidence_value
                >=
                STABLE_CONFIDENCE

            ):

                # --------------------------------------------
                # Accept the sign.
                # --------------------------------------------

                if (
                    accepted_prediction
                    !=
                    predicted_index
                ):

                    accepted_prediction = (
                        predicted_index
                    )


                    print()
                    print(
                        "================================"
                    )

                    print(
                        f"STABLE PREDICTION: "
                        f"{predicted_label}"
                    )

                    print(
                        f"CONFIDENCE: "
                        f"{confidence_value:.2f}"
                    )

                    print(
                        "================================"
                    )
                    print()


                    await websocket.send_json({

                        "status":
                            "prediction",

                        "text":
                            predicted_label,

                        "confidence":
                            confidence_value,

                        "frames":
                            SEQUENCE_LENGTH,

                        "required":
                            SEQUENCE_LENGTH
                    })


                else:

                    # ----------------------------------------
                    # Same sign continues.
                    #
                    # Tell frontend that prediction is
                    # continuing, but don't trigger speech.
                    # ----------------------------------------

                    await websocket.send_json({

                        "status":
                            "tracking",

                        "text":
                            predicted_label,

                        "confidence":
                            confidence_value,

                        "frames":
                            SEQUENCE_LENGTH,

                        "required":
                            SEQUENCE_LENGTH
                    })


            else:

                # =================================================
                # HIGH CONFIDENCE BUT NOT YET STABLE
                # =================================================

                await websocket.send_json({

                    "status":
                        "stabilizing",

                    "text":
                        "",

                    "confidence":
                        confidence_value,

                    "frames":
                        SEQUENCE_LENGTH,

                    "required":
                        SEQUENCE_LENGTH
                })


    except WebSocketDisconnect:

        print()
        print("==============================")
        print("CLIENT DISCONNECTED")
        print("==============================")
        print()


    except Exception as e:

        print()
        print("==============================")
        print(
            "WebSocket error:",
            repr(e)
        )
        print("==============================")
        print()