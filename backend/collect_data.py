import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path

from features import landmarks_to_features


# ============================================================
# CONFIGURATION
# ============================================================

LABELS = [
    "HELLO",
    "YES",
    "NO",
    "THANK_YOU",
    "SORRY"
]

SAMPLES_PER_CLASS = 50
SEQUENCE_LENGTH = 30

COUNTDOWN_SECONDS = 3

CAMERA_INDEX = 0


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=1,

    min_detection_confidence=0.3,

    min_tracking_confidence=0.3
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


# ============================================================
# EXISTING SAMPLE COUNT
# ============================================================

def get_existing_samples(label):

    label_dir = DATA_DIR / label

    if not label_dir.exists():
        return 0

    return len(
        list(label_dir.glob("*.npy"))
    )


# ============================================================
# DRAW LANDMARKS + NUMBERS
# ============================================================

def draw_hand_landmarks(
    frame,
    hand_landmarks
):

    # --------------------------------------------------------
    # Draw normal MediaPipe skeleton
    # --------------------------------------------------------

    mp_drawing.draw_landmarks(

        frame,

        hand_landmarks,

        mp_hands.HAND_CONNECTIONS,

        mp_drawing_styles.get_default_hand_landmarks_style(),

        mp_drawing_styles.get_default_hand_connections_style()
    )


    # --------------------------------------------------------
    # Draw landmark numbers
    # --------------------------------------------------------

    for index, landmark in enumerate(
        hand_landmarks.landmark
    ):

        h, w, _ = frame.shape

        x = int(
            landmark.x * w
        )

        y = int(
            landmark.y * h
        )


        # Wrist gets a larger circle

        if index == 0:

            cv2.circle(
                frame,
                (x, y),
                8,
                (0, 0, 255),
                -1
            )

        else:

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


        # Landmark number

        cv2.putText(

            frame,

            str(index),

            (x + 7, y - 7),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (255, 255, 0),

            1
        )


# ============================================================
# SHOW COORDINATES
# ============================================================

def draw_coordinate_panel(
    frame,
    hand_landmarks
):

    h, w, _ = frame.shape

    panel_x = 10
    panel_y = 140

    # Background panel

    cv2.rectangle(

        frame,

        (panel_x, panel_y),

        (panel_x + 300, panel_y + 200),

        (0, 0, 0),

        -1
    )


    # Wrist coordinates

    wrist = hand_landmarks.landmark[0]


    cv2.putText(

        frame,

        "WRIST (0)",

        (20, 165),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (0, 0, 255),

        2
    )


    cv2.putText(

        frame,

        f"X: {wrist.x:.3f}",

        (20, 190),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255, 255, 255),

        1
    )


    cv2.putText(

        frame,

        f"Y: {wrist.y:.3f}",

        (20, 215),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255, 255, 255),

        1
    )


    cv2.putText(

        frame,

        f"Z: {wrist.z:.3f}",

        (20, 240),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255, 255, 255),

        1
    )


    # --------------------------------------------------------
    # Show relative coordinates of landmark 9
    # --------------------------------------------------------

    middle_mcp = hand_landmarks.landmark[9]


    relative_x = (
        middle_mcp.x - wrist.x
    )

    relative_y = (
        middle_mcp.y - wrist.y
    )

    relative_z = (
        middle_mcp.z - wrist.z
    )


    cv2.putText(

        frame,

        "LANDMARK 9 RELATIVE",

        (20, 275),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 255, 255),

        1
    )


    cv2.putText(

        frame,

        f"X: {relative_x:.3f}",

        (20, 300),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255, 255, 255),

        1
    )


    cv2.putText(

        frame,

        f"Y: {relative_y:.3f}",

        (20, 325),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255, 255, 255),

        1
    )


    cv2.putText(

        frame,

        f"Z: {relative_z:.3f}",

        (20, 350),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255, 255, 255),

        1
    )


# ============================================================
# RECORD ONE SAMPLE
# ============================================================

def record_sample(
    label,
    sample_number
):

    print()
    print("=" * 60)

    print(
        f"{label} | "
        f"Sample {sample_number}/{SAMPLES_PER_CLASS}"
    )

    print("=" * 60)

    print(
        "Press SPACE to start."
    )

    print(
        "Press Q to quit."
    )


    # ========================================================
    # WAIT FOR SPACE
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            continue


        frame = cv2.flip(
            frame,
            1
        )


        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        result = hands.process(
            rgb
        )


        # ----------------------------------------------------
        # Draw hand if detected
        # ----------------------------------------------------

        if result.multi_hand_landmarks:

            hand_landmarks = (
                result.multi_hand_landmarks[0]
            )

            draw_hand_landmarks(
                frame,
                hand_landmarks
            )

            draw_coordinate_panel(
                frame,
                hand_landmarks
            )

            detection_text = (
                "HAND DETECTED"
            )

            detection_color = (
                0, 255, 0
            )

        else:

            detection_text = (
                "NO HAND DETECTED"
            )

            detection_color = (
                0, 0, 255
            )


        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"{label} | Sample "
            f"{sample_number}/{SAMPLES_PER_CLASS}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            "SPACE = Start | Q = Quit",

            (20, 75),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2
        )


        cv2.putText(

            frame,

            detection_text,

            (20, 110),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            detection_color,

            2
        )


        cv2.imshow(
            "ISL Data Collection",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            return None


        if key == 32:

            break


    # ========================================================
    # COUNTDOWN
    # ========================================================

    countdown_start = time.time()


    while True:

        ret, frame = cap.read()

        if not ret:
            continue


        frame = cv2.flip(
            frame,
            1
        )


        elapsed = (
            time.time()
            - countdown_start
        )


        remaining = (
            COUNTDOWN_SECONDS
            - int(elapsed)
        )


        if remaining <= 0:
            break


        cv2.putText(

            frame,

            f"GET READY: {remaining}",

            (20, 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.2,

            (0, 255, 255),

            3
        )


        cv2.imshow(
            "ISL Data Collection",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            return None


    # ========================================================
    # RECORD 30 FRAMES
    # ========================================================

    sequence = []


    while len(sequence) < SEQUENCE_LENGTH:

        ret, frame = cap.read()

        if not ret:
            continue


        frame = cv2.flip(
            frame,
            1
        )


        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        result = hands.process(
            rgb
        )


        # ----------------------------------------------------
        # HAND FOUND
        # ----------------------------------------------------

        if result.multi_hand_landmarks:

            hand_landmarks = (
                result.multi_hand_landmarks[0]
            )


            # Draw trackers

            draw_hand_landmarks(
                frame,
                hand_landmarks
            )


            # Draw coordinates

            draw_coordinate_panel(
                frame,
                hand_landmarks
            )


            # ------------------------------------------------
            # Convert to wrist-relative features
            # ------------------------------------------------

            features = (
                landmarks_to_features(
                    hand_landmarks.landmark
                )
            )


            sequence.append(
                features
            )


            status = (
                "HAND DETECTED"
            )

            status_color = (
                0, 255, 0
            )


        else:

            status = (
                "NO HAND DETECTED"
            )

            status_color = (
                0, 0, 255
            )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"RECORDING: "
            f"{len(sequence)}/"
            f"{SEQUENCE_LENGTH}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 0, 255),

            2
        )


        cv2.putText(

            frame,

            f"Sign: {label}",

            (20, 75),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            status,

            (20, 110),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            status_color,

            2
        )


        cv2.imshow(

            "ISL Data Collection",

            frame
        )


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        if key == ord("q"):

            return None


    # ========================================================
    # SAVE
    # ========================================================

    label_dir = (
        DATA_DIR / label
    )

    label_dir.mkdir(
        exist_ok=True
    )


    timestamp = int(
        time.time() * 1000
    )


    save_path = (
        label_dir
        / f"{timestamp}.npy"
    )


    sequence_array = np.array(

        sequence,

        dtype=np.float32
    )


    np.save(

        save_path,

        sequence_array
    )


    print(
        f"Saved: {save_path}"
    )


    print(
        f"Shape: "
        f"{sequence_array.shape}"
    )


    return sequence_array


# ============================================================
# MAIN
# ============================================================

try:

    print()
    print("=" * 60)
    print("ISL DATA COLLECTION")
    print("=" * 60)

    print(
        f"Classes: {LABELS}"
    )

    print(
        f"Samples per class: "
        f"{SAMPLES_PER_CLASS}"
    )

    print(
        f"Frames per sample: "
        f"{SEQUENCE_LENGTH}"
    )

    print()
    print(
        "MediaPipe will track 21 hand landmarks."
    )

    print(
        "Landmark 0 = wrist."
    )

    print(
        "Features = wrist-relative XYZ."
    )

    print("=" * 60)


    for label in LABELS:

        existing = (
            get_existing_samples(
                label
            )
        )


        print(
            f"{label}: "
            f"{existing} existing samples"
        )


        start_number = (
            existing + 1
        )


        if (
            start_number
            > SAMPLES_PER_CLASS
        ):

            continue


        for sample_number in range(

            start_number,

            SAMPLES_PER_CLASS + 1

        ):

            result = record_sample(

                label,

                sample_number
            )


            if result is None:

                raise KeyboardInterrupt


finally:

    cap.release()

    cv2.destroyAllWindows()

    hands.close()

    print(
        "Data collection stopped."
    )