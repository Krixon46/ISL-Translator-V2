import {
  FilesetResolver,
  HandLandmarker,
} from "@mediapipe/tasks-vision";

let handLandmarker = null;

export async function initializeMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks(
    "/mediapipe/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(
    vision,
    {
      baseOptions: {
        modelAssetPath: "/models/hand_landmarker.task",
      },

      runningMode: "VIDEO",

      numHands: 2,

      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    }
  );

  console.log("MediaPipe initialized successfully");

  return handLandmarker;
}

export function getHandLandmarker() {
  return handLandmarker;
}