import numpy as np


def landmarks_to_features(hand_landmarks_list):
    """
    Convert up to two MediaPipe hands into 126 features.

    Each hand:
        21 landmarks × 3 coordinates = 63

    Two hands:
        63 × 2 = 126

    Features are wrist-relative XYZ coordinates.

    This matches the Kaggle training feature extraction.
    """

    hand_features = []

    # --------------------------------------------------------
    # Process up to 2 hands
    # --------------------------------------------------------

    for hand_idx in range(2):

        if hand_idx < len(hand_landmarks_list):

            landmarks = hand_landmarks_list[hand_idx]

            # Landmark 0 = wrist
            wrist = landmarks[0]

            features = []

            for landmark in landmarks:

                # Wrist-relative coordinates
                x = landmark.x - wrist.x
                y = landmark.y - wrist.y
                z = landmark.z - wrist.z

                features.extend([
                    x,
                    y,
                    z
                ])

            hand_features.extend(
                features
            )

        else:

            # No second hand detected.
            # Add 63 zeros.
            hand_features.extend(
                [0.0] * 63
            )

    return np.array(
        hand_features,
        dtype=np.float32
    )