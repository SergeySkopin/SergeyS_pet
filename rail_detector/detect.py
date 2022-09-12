import os
import sys
import cv2
import datetime
import argparse

CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".")
sys.path.append(os.path.join(CURRENT_DIR, "../../"))
try:
    from segmentation.deploy import RailtrackSegmentation
    from cfg import BiSeNetV2Config
except Exception as e:
    print(e)
    sys.exit(0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-weights", type=str, required=True, help="weight_path")
    parser.add_argument("-source", type=str, required=True, help="input_data")
    parser.add_argument("-output", type=str, default="result.MP4")
    parser.add_argument("-imgsz_h", type=int, default=640)
    parser.add_argument("-imgsz_w", type=int, default=1280)

    args = parser.parse_args()

    return args


# TODO: add Hydra
def main():
    args = get_args()
    segmentation_handler = RailtrackSegmentation(
        args.weights, args.imgsz_h, args.imgsz_w, BiSeNetV2Config()
    )

    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise Exception("failed to open {}".format(args.source))

    width = int(capture.get(3))
    height = int(capture.get(4))

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    fps = 30.0
    out_video = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    _total_ms = 0
    count_frame = 0
    while capture.isOpened():
        ret, frame = capture.read()
        count_frame += 1

        if not ret:
            break

        start = datetime.datetime.now()
        _, overlay = segmentation_handler.run(frame, only_mask=False)
        _total_ms += (datetime.datetime.now() - start).total_seconds() * 1000
        out_video.write(overlay)

    print("processing time one frame {}[ms]".format(_total_ms / count_frame))

    capture.release()
    out_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
