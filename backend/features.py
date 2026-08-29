import numpy as np


def process_single_hand(hand_landmarks):
    """
    Convert one hand into the EXACT 63 features
    used during Kaggle training.

    21 landmarks × 3 coordinates = 63

    IMPORTANT:
    Kaggle training used wrist-relative coordinates
    WITHOUT hand-size normalization.
    """

    wrist = hand_landmarks[0]

    features = []

    for landmark in hand_landmarks:

        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        features.extend([
            x,
            y,
            z
        ])

    return np.array(
        features,
        dtype=np.float32
    )


def landmarks_to_features(result):
    """
    Convert MediaPipe result into exactly 126 features.

    Hand 1:
        63 features

    Hand 2:
        63 features

    Total:
        126 features

    If only one hand is detected,
    the second hand is filled with zeros.

    This must match the Kaggle training pipeline exactly.
    """

    detected_hands = result.hand_landmarks

    all_features = []

    # ========================================================
    # FIRST HAND
    # ========================================================

    if len(detected_hands) >= 1:

        first_hand = process_single_hand(
            detected_hands[0]
        )

    else:

        first_hand = np.zeros(
            63,
            dtype=np.float32
        )

    all_features.extend(first_hand)


    # ========================================================
    # SECOND HAND
    # ========================================================

    if len(detected_hands) >= 2:

        second_hand = process_single_hand(
            detected_hands[1]
        )

    else:

        second_hand = np.zeros(
            63,
            dtype=np.float32
        )

    all_features.extend(second_hand)


    # ========================================================
    # FINAL FEATURE VECTOR
    # ========================================================

    features = np.array(
        all_features,
        dtype=np.float32
    )

    return features