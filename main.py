import argparse
from collections import defaultdict, deque
import os
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import time

SOURCE = np.array([[320,127], [422,134], [256,204], [494,236]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 30

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


class ViewTransformer:

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Ultralytics and Supervision"
    )
    # parser.add_argument(
    #     "--source_video_path",
    #     required=True,
    #     help="Path to the source video file",
    #     type=str,
    # )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )
    parser.add_argument(
        "--screenshot_dir",
        required=True,
        help="Directory to save screenshots of speeding vehicles",
        type=str,
    )

    # parser.add_argument(
    #     "--lp_model_path",
    #     required=True, 
    #     help="Path to the custom-trained YOLO model for license plate detection",
    #     type=str,
    #     )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    model = YOLO("yolov8n.pt")
    # lp_model = YOLO(args.lp_model_path)

    cam_detect = cv2.VideoCapture(1)
    fps_cam = int(round(cam_detect.get(cv2.CAP_PROP_FPS)))
    width_cam = int(cam_detect.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_cam = int(cam_detect.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_cam = min(640/width_cam, 640/height_cam) # lay ti le khung hinh
    nh_cam = int(scale_cam * height_cam)
    nw_cam = int(scale_cam * width_cam)


    byte_track = sv.ByteTrack(
        frame_rate=fps_cam, track_activation_threshold=args.confidence_threshold
    )

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh= (nh_cam, nw_cam)
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(nh_cam, nw_cam))
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
        text_padding=0,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=fps_cam * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    #frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=fps_cam))


    os.makedirs(args.screenshot_dir, exist_ok=True) #tạo ra các thư mục để lưu các ảnh tại đường dẫn
    with sv.VideoSink(args.target_video_path, sv.VideoInfo(width=width_cam, height=height_cam, fps=fps_cam)) as sink:
        while True:
            ret_cam, frame_cam = cam_detect.read()
            if not ret_cam:
                break
            #t1 = time.time()
            result = model(frame_cam)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > args.confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=args.iou_threshold) 
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = view_transformer.transform_points(points=points).astype(int)

            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)
#               time_run[tracker_id].append(t2-t1)
            labels = []
            speeding = False


            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < fps_cam / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / fps_cam
                    speed = distance / time * 3.6
                    labels.append(f"#{tracker_id} {int(speed)} km/h")

                    if speed > 100:
                        speeding = True

            annotated_frame = frame_cam.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            # # License Plate Detection
            # lp_results = lp_model(frame_cam)[0]
            # lp_detections = sv.Detections.from_ultralytics(lp_results)
            # lp_detections = lp_detections[lp_detections.confidence > args.confidence_threshold]

            # for lp_bbox in lp_detections.xyxy:
            #     x1, y1, x2, y2 = map(int, lp_bbox)
            #     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     cv2.putText(annotated_frame, "License Plate", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)


            if speeding:
                screenshot_path = os.path.join(
                    args.screenshot_dir, f"frame_{frame_cam:06d}.jpg"
                )
                
                cv2.imwrite(screenshot_path, annotated_frame)

            sink.write_frame(annotated_frame)
            cv2.imshow("frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            #t2=time.time()
        cv2.destroyAllWindows()