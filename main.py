import os
import cv2

from ultralytics import YOLO
import supervision as sv
import numpy as np


LINE_START = sv.Point(320, 0)
LINE_END = sv.Point(320, 480)


def main():
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    model = YOLO("yolov8n.pt")
    model.fuse()
    detector = cv2.AKAZE_create(
        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,  # Descriptor type: MLDB or KAZE
        descriptor_size=0,  # Size of the descriptor in bits (0 for full-size)
        descriptor_channels=3,  # Number of channels in the descriptor
        threshold=0.001,  # Detector response threshold to accept/reject keypoints
        nOctaves=4,  # Maximum octave evolution of the image
        nOctaveLayers=4,  # Default number of sublevels per scale level
        diffusivity=cv2.KAZE_DIFF_CHARBONNIER,  # Diffusivity type (DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER)
    )

    # detector = cv2.ORB_create(
    #     nfeatures=5000,  # Maximum number of keypoints
    #     scaleFactor=1.2,  # Pyramid decimation ratio
    #     nlevels=8,  # Number of pyramid levels
    #     edgeThreshold=31,  # Size of the border where features are not detected
    #     firstLevel=0,  # Level of pyramid to put source image (0 means the original resolution)
    #     WTA_K=2,  # Number of points producing each element of the oriented BRIEF descriptor
    #     scoreType=cv2.ORB_HARRIS_SCORE,  # Type of the score
    #     patchSize=31,  # Size of the patch used by the oriented BRIEF descriptor
    #     fastThreshold=20,  # FAST threshold
    # )

    for frame_number, result in enumerate(
        model.track(
            source="test.mp4", show=False, stream=True, agnostic_nms=True, persist=True
        )
    ):
        frame = result.orig_img
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, _ = detector.detectAndCompute(gray_frame, None)
        frame_with_keypoints = cv2.drawKeypoints(
            frame, keypoints, None, color=(0, 255, 0), flags=0
        )
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            # print(detections.tracker_id)

        # print(detections)

        labels = []
        for bbox, _, confidence, class_id, tracker_id in detections:
            if class_id == 0:  # Check if the class_id is for "person"
                labels.append(f"{tracker_id} {model.names[class_id]} {confidence:0.2f}")
                # Get the bounding box coordinates
                bbox = tuple(map(int, bbox))

                # Extract the box from the frame
                box_image = frame_with_keypoints[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                tracker_dir = f"track_{tracker_id}"
                os.makedirs(tracker_dir, exist_ok=True)

                # Save the box image to a file
                image_filename = f"track_{tracker_id}/frame_{frame_number}.jpg"
                cv2.imwrite(image_filename, box_image)

        frame_with_keypoints = box_annotator.annotate(
            scene=frame_with_keypoints, detections=detections, labels=labels
        )

        cv2.imshow("yolov8", frame_with_keypoints)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
