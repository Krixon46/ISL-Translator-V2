import json
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import joblib

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

from features import landmarks_to_features
from model import SignLSTM


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = (
    BASE_DIR / "best_model.pt"
)

LABELS_PATH = (
    BASE_DIR / "labels.json"
)

SCALER_PATH = (
    BASE_DIR / "scaler.pkl"
)

MEDIAPIPE_MODEL = (
    BASE_DIR
    / "assets"
    / "hand_landmarker.task"
)


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# Kaggle training used:
#
# 20 frames
# 126 features per frame

SEQUENCE_LENGTH = 20

INPUT_SIZE = 126

HIDDEN_SIZE = 128

NUM_LAYERS = 2

NUM_CLASSES = 9

CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# LOAD SCALER
# ============================================================

if not SCALER_PATH.exists():

    raise FileNotFoundError(
        f"Scaler not found:\n{SCALER_PATH}"
    )


scaler = joblib.load(
    SCALER_PATH
)


print(
    "Scaler loaded successfully."
)


# ============================================================
# LOAD LABELS
# ============================================================

if not LABELS_PATH.exists():

    raise FileNotFoundError(
        f"Labels file not found:\n{LABELS_PATH}"
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
# CHECK LABEL COUNT
# ============================================================

if len(LABELS) != NUM_CLASSES:

    raise ValueError(

        f"Expected {NUM_CLASSES} labels, "
        f"but found {len(LABELS)} labels."
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
# LOAD PYTORCH MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


# best_model.pt contains ONLY the state_dict.
state_dict = torch.load(

    MODEL_PATH,

    map_location=device
)


# ============================================================
# CREATE MODEL
# ============================================================

model = SignLSTM(

    input_size=INPUT_SIZE,

    hidden_size=HIDDEN_SIZE,

    num_layers=NUM_LAYERS,

    num_classes=NUM_CLASSES
)


# ============================================================
# LOAD WEIGHTS
# ============================================================

model.load_state_dict(
    state_dict
)


model.to(device)

model.eval()


print(
    "Model loaded successfully."
)


print(
    f"Model input size: {INPUT_SIZE}"
)

print(
    f"Sequence length: {SEQUENCE_LENGTH}"
)

print(
    f"Number of classes: {NUM_CLASSES}"
)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_tasks = mp.tasks

BaseOptions = (
    mp_tasks.BaseOptions
)

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
# MEDIAPIPE OPTIONS
# ============================================================

options = HandLandmarkerOptions(

    base_options=BaseOptions(

        model_asset_path=
            str(
                MEDIAPIPE_MODEL.resolve()
            )
    ),

    running_mode=
        RunningMode.VIDEO,

    # IMPORTANT:
    # Kaggle model uses up to 2 hands.
    num_hands=2,

    min_hand_detection_confidence=
        0.5,

    min_hand_presence_confidence=
        0.5,

    min_tracking_confidence=
        0.5
)


# ============================================================
# CREATE HAND LANDMARKER
# ============================================================

landmarker = (
    HandLandmarker
    .create_from_options(
        options
    )
)


print(
    "MediaPipe hand landmarker loaded."
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
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

    print(
        "Client connected."
    )


    # --------------------------------------------------------
    # Sequence buffer
    # --------------------------------------------------------

    sequence = deque(
        maxlen=SEQUENCE_LENGTH
    )


    # --------------------------------------------------------
    # MediaPipe video timestamps
    # --------------------------------------------------------

    timestamp_ms = 0


    try:

        while True:

            # =================================================
            # RECEIVE JPEG FRAME
            # =================================================

            data = (
                await websocket
                .receive_bytes()
            )


            # =================================================
            # JPEG → NUMPY
            # =================================================

            np_data = np.frombuffer(

                data,

                dtype=np.uint8
            )


            # =================================================
            # NUMPY → OPENCV FRAME
            # =================================================

            frame = cv2.imdecode(

                np_data,

                cv2.IMREAD_COLOR
            )


            if frame is None:

                continue


            # =================================================
            # BGR → RGB
            # =================================================

            frame_rgb = cv2.cvtColor(

                frame,

                cv2.COLOR_BGR2RGB
            )


            # =================================================
            # CREATE MEDIAPIPE IMAGE
            # =================================================

            mp_image = mp.Image(

                image_format=
                    mp.ImageFormat.SRGB,

                data=frame_rgb
            )


            # =================================================
            # TIMESTAMP
            # =================================================

            timestamp_ms += 50


            # =================================================
            # HAND DETECTION
            # =================================================

            result = (
                landmarker
                .detect_for_video(

                    mp_image,

                    timestamp_ms
                )
            )


            # =================================================
            # NO HAND
            # =================================================

            if not result.hand_landmarks:

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


            # =================================================
            # EXTRACT FEATURES
            # =================================================

            # result.hand_landmarks can contain
            # up to two hands.
            #
            # Our feature extractor converts:
            #
            # Hand 1 → 63
            # Hand 2 → 63
            #
            # Total → 126

            features = (
                landmarks_to_features(
                    result.hand_landmarks
                )
            )


            # =================================================
            # SAFETY CHECK
            # =================================================

            if features.shape != (126,):

                print(
                    "ERROR: Invalid feature shape:",
                    features.shape
                )

                await websocket.send_json({

                    "status":
                        "feature_error",

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


            # =================================================
            # ADD FRAME TO SEQUENCE
            # =================================================

            sequence.append(
                features
            )


            # =================================================
            # WAIT FOR 20 FRAMES
            # =================================================

            if (
                len(sequence)
                < SEQUENCE_LENGTH
            ):

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


            # =================================================
            # CONVERT SEQUENCE TO NUMPY
            # =================================================

            X = np.array(

                sequence,

                dtype=np.float32
            )


            # Expected:
            #
            # 20 × 126

            if X.shape != (
                SEQUENCE_LENGTH,
                INPUT_SIZE
            ):

                print(
                    "ERROR: Invalid sequence shape:",
                    X.shape
                )

                continue


            # =================================================
            # APPLY SCALER
            # =================================================

            # The scaler was fitted during Kaggle
            # training on the 126 features.

            X = scaler.transform(
                X
            )


            # =================================================
            # NUMPY → PYTORCH
            # =================================================

            X = torch.tensor(

                X,

                dtype=torch.float32
            )


            # =================================================
            # ADD BATCH DIMENSION
            # =================================================

            # Before:
            #
            # 20 × 126
            #
            # After:
            #
            # 1 × 20 × 126

            X = X.unsqueeze(0)


            # =================================================
            # MOVE TO DEVICE
            # =================================================

            X = X.to(device)


            # =================================================
            # MODEL PREDICTION
            # =================================================

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


            # =================================================
            # GET PREDICTION
            # =================================================

            predicted_index = (
                prediction.item()
            )


            confidence_value = (
                confidence.item()
            )


            predicted_label = (
                LABELS[
                    predicted_index
                ]
            )


            # =================================================
            # CONFIDENCE FILTER
            # =================================================

            if (
                confidence_value
                < CONFIDENCE_THRESHOLD
            ):

                await websocket.send_json({

                    "status":
                        "uncertain",

                    "text":
                        "",

                    "confidence":
                        confidence_value,

                    "frames":
                        len(sequence),

                    "required":
                        SEQUENCE_LENGTH
                })

                continue


            # =================================================
            # PRINT PREDICTION
            # =================================================

            print(

                f"Prediction: "
                f"{predicted_label} "
                f"| Confidence: "
                f"{confidence_value:.2f}"
            )


            # =================================================
            # SEND PREDICTION TO FRONTEND
            # =================================================

            await websocket.send_json({

                "status":
                    "prediction",

                "text":
                    predicted_label,

                "confidence":
                    confidence_value,

                "frames":
                    len(sequence),

                "required":
                    SEQUENCE_LENGTH
            })


    # ========================================================
    # CLIENT DISCONNECTED
    # ========================================================

    except WebSocketDisconnect:

        print(
            "Client disconnected."
        )


    # ========================================================
    # OTHER ERROR
    # ========================================================

    except Exception as e:

        print(
            "WebSocket error:",
            repr(e)
        )