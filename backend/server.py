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

SEQUENCE_LENGTH = 20

# Minimum confidence to consider a prediction.
CONFIDENCE_THRESHOLD = 0.60

# Confidence required to accept a stable prediction.
STABLE_CONFIDENCE = 0.65

# Number of consecutive matching predictions.
STABLE_PREDICTIONS_REQUIRED = 3

# Number of consecutive no-hand frames
# required to release/reset the current sign.
RELEASE_FRAMES_REQUIRED = 5

# Frontend should ideally send around 10 FPS.
FRAME_INTERVAL_MS = 100


# ============================================================
# LOAD SCALER
# ============================================================

if not SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Scaler not found: {SCALER_PATH}"
    )

scaler = joblib.load(SCALER_PATH)

print("Scaler loaded successfully.")


# ============================================================
# LOAD LABELS
# ============================================================

if not LABELS_PATH.exists():
    raise FileNotFoundError(
        f"Labels file not found: {LABELS_PATH}"
    )

with open(LABELS_PATH, "r") as f:
    LABELS = json.load(f)

print("Loaded labels:", LABELS)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


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

model.load_state_dict(state_dict)

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

print("Input features: 126")

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
        "https://isl-translator-v2-pink.vercel.app"
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

    print()
    print("==============================")
    print("CLIENT CONNECTED")
    print("==============================")
    print()


    # ========================================================
    # CREATE MEDIAPIPE LANDMARKER
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


    # ========================================================
    # SLIDING WINDOW
    # ========================================================

    sequence = deque(
        maxlen=SEQUENCE_LENGTH
    )


    # ========================================================
    # MEDIAPIPE TIMESTAMP
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


    # ========================================================
    # ACCEPTED / LOCKED PREDICTION
    #
    # Once a sign is accepted:
    #
    # prediction_sent = True
    #
    # We DO NOT accept another prediction
    # until the hand is removed.
    # ========================================================

    accepted_prediction = None


    # ========================================================
    # DEBUG COUNTER
    # ========================================================

    prediction_window_count = 0


    # ========================================================
    # CLIENT CONNECTION STATE
    # ========================================================

    client_connected = True


    try:

        while client_connected:

            # ==================================================
            # RECEIVE FRAME
            # ==================================================

            try:

                data = (
                    await websocket.receive_bytes()
                )

            except WebSocketDisconnect:

                client_connected = False

                print()
                print("==============================")
                print("CLIENT DISCONNECTED")
                print("==============================")
                print()

                break


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
                f"MediaPipe: "
                f"{hand_count} hand(s) detected"
            )


            # ==================================================
            # NO HAND
            # ==================================================

            if hand_count == 0:

                no_hand_count += 1


                # ----------------------------------------------
                # HAND HAS BEEN REMOVED
                # ----------------------------------------------

                if (
                    no_hand_count
                    >= RELEASE_FRAMES_REQUIRED
                ):

                    print()
                    print("==============================")
                    print("HAND RELEASED")
                    print("RESETTING ALL STATE")
                    print("==============================")
                    print()


                    # ------------------------------------------
                    # CLEAR EVERYTHING
                    # ------------------------------------------

                    sequence.clear()

                    last_prediction_index = None

                    stable_prediction_count = 0

                    accepted_prediction = None

                    prediction_window_count = 0

                    no_hand_count = 0


                    # ------------------------------------------
                    # Tell frontend:
                    #
                    # 0 / 20
                    #
                    # ------------------------------------------

                    if client_connected:

                        try:

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

                        except (
                            WebSocketDisconnect,
                            RuntimeError
                        ):

                            client_connected = False


                    continue


                # ----------------------------------------------
                # Temporarily no hand
                # ----------------------------------------------

                if client_connected:

                    try:

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

                    except (
                        WebSocketDisconnect,
                        RuntimeError
                    ):

                        client_connected = False

                continue


            # ==================================================
            # HAND FOUND
            # ==================================================

            no_hand_count = 0


            # ==================================================
            # IMPORTANT:
            #
            # If a prediction was already accepted,
            # DO NOT continue making predictions.
            #
            # We simply tell the frontend that the sign
            # is locked until the hand is removed.
            # ==================================================

            if accepted_prediction is not None:

                predicted_label = (
                    LABELS[
                        accepted_prediction
                    ]
                )


                if client_connected:

                    try:

                        await websocket.send_json({

                            "status":
                                "tracking",

                            "text":
                                predicted_label,

                            "confidence":
                                1.0,

                            "frames":
                                SEQUENCE_LENGTH,

                            "required":
                                SEQUENCE_LENGTH
                        })

                    except (
                        WebSocketDisconnect,
                        RuntimeError
                    ):

                        client_connected = False


                continue


            # ==================================================
            # FEATURE EXTRACTION
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
                    "ERROR: Expected 126 features, "
                    f"got {features.shape[0]}"
                )


                if client_connected:

                    try:

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

                    except (
                        WebSocketDisconnect,
                        RuntimeError
                    ):

                        client_connected = False

                continue


            # ==================================================
            # SCALE FEATURES
            # ==================================================

            features_scaled = (
                scaler.transform(
                    features.reshape(1, -1)
                )[0]
            )


            features_scaled = (
                features_scaled.astype(
                    np.float32
                )
            )


            # ==================================================
            # ADD FRAME TO SLIDING WINDOW
            # ==================================================

            sequence.append(
                features_scaled
            )


            # ==================================================
            # STILL COLLECTING
            # ==================================================

            if len(sequence) < SEQUENCE_LENGTH:

                if client_connected:

                    try:

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

                    except (
                        WebSocketDisconnect,
                        RuntimeError
                    ):

                        client_connected = False

                continue


            # ==================================================
            # SLIDING WINDOW READY
            # ==================================================

            prediction_window_count += 1


            # ==================================================
            # NUMPY
            # ==================================================

            X = np.array(
                sequence,
                dtype=np.float32
            )


            # ==================================================
            # TORCH
            # ==================================================

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
                    f"Window "
                    f"{prediction_window_count}: "
                    f"Uncertain → "
                    f"{predicted_label} "
                    f"| {confidence_value:.2f}"
                )


                # Reset stabilization because
                # confidence wasn't sufficient.

                last_prediction_index = None

                stable_prediction_count = 0


                if client_connected:

                    try:

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

                    except (
                        WebSocketDisconnect,
                        RuntimeError
                    ):

                        client_connected = False

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
                f"Window "
                f"{prediction_window_count}: "
                f"{predicted_label} "
                f"| {confidence_value:.2f} "
                f"| stable "
                f"{stable_prediction_count}/"
                f"{STABLE_PREDICTIONS_REQUIRED}"
            )


            # ==================================================
            # STABLE PREDICTION
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

                # ----------------------------------------------
                # LOCK PREDICTION
                # ----------------------------------------------

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
                    "PREDICTION LOCKED "
                    "UNTIL HAND RELEASE"
                )

                print(
                    "================================"
                )
                print()


                if client_connected:

                    try:

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

                    except (
                        WebSocketDisconnect,
                        RuntimeError
                    ):

                        client_connected = False


            else:

                # =================================================
                # STABILIZING
                # =================================================

                if client_connected:

                    try:

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

                    except (
                        WebSocketDisconnect,
                        RuntimeError
                    ):

                        client_connected = False


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


    finally:

        # ========================================================
        # CLEAN UP MEDIAPIPE
        # ========================================================

        try:

            landmarker.close()

        except Exception:

            pass


        # ========================================================
        # CLEAN UP STATE
        # ========================================================

        sequence.clear()

        print(
            "Connection cleanup complete."
        )