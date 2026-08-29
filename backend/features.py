import numpy as np


def _get_coord(landmark, key):
    if isinstance(landmark, dict):
        return landmark.get(key, 0.0)
    return getattr(landmark, key, 0.0)


def process_single_hand(hand_landmarks):
    """
    Convert one hand into the EXACT 63 features
    used during Kaggle training.

    21 landmarks × 3 coordinates = 63

    IMPORTANT:
    Kaggle training used wrist-relative coordinates
    WITHOUT hand-size normalization.
    """
    if not hand_landmarks or len(hand_landmarks) < 21:
        return np.zeros(63, dtype=np.float32)

    wrist = hand_landmarks[0]
    wrist_x = _get_coord(wrist, "x")
    wrist_y = _get_coord(wrist, "y")
    wrist_z = _get_coord(wrist, "z")

    features = []

    for landmark in hand_landmarks:
        x = _get_coord(landmark, "x") - wrist_x
        y = _get_coord(landmark, "y") - wrist_y
        z = _get_coord(landmark, "z") - wrist_z

        features.extend([
            x,
            y,
            z
        ])

    return np.array(
        features,
        dtype=np.float32
    )


def landmarks_to_features(hands_data):
    """
    Convert hands landmark list (from JSON payload or MediaPipe) into exactly 126 features.

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
    if hasattr(hands_data, "hand_landmarks"):
        detected_hands = hands_data.hand_landmarks
    elif isinstance(hands_data, list):
        detected_hands = hands_data
    else:
        detected_hands = []

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