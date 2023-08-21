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
    for frame_number, result in enumerate(
        model.track(
            source="test.mp4", show=False, stream=True, agnostic_nms=True, persist=True
        )
    ):
        frame = result.orig_img
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
                box_image = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                # print(box_image)
                cv2.imshow("croped", box_image)
                tracker_dir = f"track_{tracker_id}"
                os.makedirs(tracker_dir, exist_ok=True)

                # Save the box image to a file
                image_filename = f"track_{tracker_id}/frame_{frame_number}.jpg"
                print(image_filename)
                cv2.imwrite(image_filename, box_image)

        frame = box_annotator.annotate(
            scene=frame, detections=detections, labels=labels
        )

        cv2.imshow("yolov8", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
