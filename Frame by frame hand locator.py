import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor  # Corrected import

def hand_locator(frame, detector):
    """Detects hands in a frame and returns bounding box information."""
    image_height, image_width, _ = frame.shape
    temp_path = "temp_frame.jpg"
    cv2.imwrite(temp_path, frame)
    mp_image = mp.Image.create_from_file(temp_path)
    os.remove(temp_path)

    detection_result = detector.detect(mp_image)
    hand_info = []

    if detection_result.hand_landmarks:
        for landmarks in detection_result.hand_landmarks:
            landmark_array = np.array(
                [(lm.x * image_width, lm.y * image_height) for lm in landmarks]
            )
            x_min, y_min = np.min(landmark_array, axis=0)
            x_max, y_max = np.max(landmark_array, axis=0)
            bounding_box = [int(x_min), int(y_min), int(x_max), int(y_max)]
            hand_info.append({"bounding_box": bounding_box})
    print("handinfo",hand_info)
    return hand_info


def apply_sam2_masks_to_video(input_video, output_video, hand_detector, sam2_predictor):
    """Applies SAM2 masks to detected hands across video frames and saves output."""
    
    video = cv2.VideoCapture(input_video)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    frame_idx = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Detect hands and get bounding boxes
        hands = hand_locator(frame, hand_detector)
        bounding_boxes = [np.array(hand["bounding_box"], dtype=np.float32) for hand in hands]

        if bounding_boxes:
            # Set the frame for the image predictor
            sam2_predictor.set_image(frame)  

            combined_mask = np.zeros(frame.shape[:2], dtype=bool)

            # Process each hand separately
            for bbox in bounding_boxes:
                masks, _, _ = sam2_predictor.predict(box=bbox)  # Predict mask for hand
                for mask in masks:
                    mask_binary = mask > 0.0  # Convert logits to binary mask
                    combined_mask |= mask_binary  # Combine all hand masks

            # Overlay the mask onto the frame
            overlay = (combined_mask[:, :, None] * np.array([0, 0, 255], dtype=np.uint8)).astype(np.uint8)
            frame = cv2.addWeighted(frame, 0.4, overlay, 0.9, 0)  # Adjust opacity

        out.write(frame)
        print(f"Processed frame {frame_idx}")
        frame_idx += 1

    video.release()
    out.release()
    print(f"Processed video saved to {output_video}")


if __name__ == "__main__":
    input_video = "notebooks/videos/test.mp4"
    output_video = "output/output for frame by frame.mp4"
    hand_model = "hand_landmarker.task"
    sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"

    # Setup hand detector
    base_options = python.BaseOptions(model_asset_path=hand_model)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    hand_detector = vision.HandLandmarker.create_from_options(options)

    # Setup SAM2 image predictor
    # Load SAM2 model from local files
    sam_model = build_sam2(sam2_config, ckpt_path=sam2_checkpoint, device="cpu")

    # Initialize the SAM2 Image Predictor
    sam2_predictor = SAM2ImagePredictor(sam_model)

    # Process video
    apply_sam2_masks_to_video(input_video, output_video, hand_detector, sam2_predictor)
