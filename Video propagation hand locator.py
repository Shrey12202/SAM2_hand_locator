import os
import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sam2.build_sam import build_sam2_video_predictor


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
            # print("box:", hand_info)
    return hand_info


def apply_sam2_masks_to_video(input_video, output_video, hand_detector, sam2_predictor):
    video = cv2.VideoCapture(input_video)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Initialize SAM2 inference state
    inference_state = sam2_predictor.init_state(video_path=input_video)
    frame_idx = 0

    # First pass: Detect hands and add bounding boxes
    while True:
        ret, frame = video.read()
        if not ret:
            break

        hands = hand_locator(frame, hand_detector)
        for obj_id, hand in enumerate(hands):
            bounding_box = np.array(hand["bounding_box"], dtype=np.float32)
            sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=bounding_box,
            )
        print(f"Processed frame {frame_idx}")
        frame_idx += 1

    # Run propagation once for the entire video
    print("Running SAM2 propagation across the video...")
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze(0)
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Second pass: Apply masks and save the output video
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning
    frame_idx = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_idx in video_segments:
            combined_mask = np.zeros(frame.shape[:2], dtype=bool)
            for _, mask in video_segments[frame_idx].items():
                combined_mask |= mask

            # Apply only the mask 
            overlay = (combined_mask[:, :, None] * np.array([0, 0, 255], dtype=np.uint8)).astype(np.uint8)
            frame = cv2.addWeighted(frame, 0.4, overlay, 0.9, 0)  # Adjust mask opacity

        out.write(frame)
        frame_idx += 1

    video.release()
    out.release()
    print(f"Processed video saved to {output_video}")


if __name__ == "__main__":
    input_video = "notebooks/videos/test.mp4"
    output_video = "output/output_video_4.mp4"
    hand_model = "hand_landmarker.task"
    sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"

    # Setup hand detector
    base_options = python.BaseOptions(model_asset_path=hand_model)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    hand_detector = vision.HandLandmarker.create_from_options(options)

    # Setup SAM2 model
    sam2_predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint, device="cpu")

    # Process video
    apply_sam2_masks_to_video(input_video, output_video, hand_detector, sam2_predictor)
