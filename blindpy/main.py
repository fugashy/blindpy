from collections import namedtuple
from dataclasses import dataclass

import click
from ultralytics import YOLO
import torch
import pandas as pd
import cv2

import time


_COLUMN_NAMES = [
    "frame_id", "tracking_id", "cls", "conf", "x1", "y1", "x2", "y2"]


@click.group()
def blindpy():
    pass


@blindpy.command()
@click.argument("input_filepath", type=str)
@click.option("--output_filepath", type=str, default="/tmp/blindpy-yolo-results.txt")
@click.option("--model-name", type=str, default="yolov8s.pt")
def yolo(
        input_filepath,
        output_filepath,
        model_name):
    if not torch.backends.mps.is_available():
        print("MPS is not available...")
        return

    model = YOLO(model_name)

    results = model(source=input_filepath, show=False, save=False, device="mps")
    tuples = list()
    Bbx = namedtuple("bbx", _COLUMN_NAMES)
    bbxs = list()
    for i, r in enumerate(results):
        frame_id = i
        for bbx in r.boxes:
            tracking_id = bbx.id
            cls = int(bbx.cls)
            conf = float(bbx.conf)
            xyxy = bbx.xyxy
            x1 = float(xyxy[0,0])
            y1 = float(xyxy[0,1])
            x2 = float(xyxy[0,2])
            y2 = float(xyxy[0,3])

            bbxs.append(Bbx(frame_id, tracking_id, cls, conf, x1, y1, x2, y2))

    df = pd.DataFrame(bbxs)
    df.to_csv(output_filepath)
    print(f"output result to {output_filepath}")


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_num: int


def get_video_info(video):
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    return VideoInfo(width, height, fps, frame_num)



@blindpy.command()
@click.argument("video_path", type=str)
@click.argument("result_path", type=str)
@click.option("--output_video_path", type=str, default="/tmp/blindpy-blinden.mp4")
def seal(video_path, result_path, output_video_path):
    video = cv2.VideoCapture(video_path)
    info = get_video_info(video)
    print(f"size: ({info.width}, {info.height}), fps: {info.fps:1.2f}, num: {info.frame_num}")

    results = pd.read_csv(result_path, header=0)
    print(results)

    fmt = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
            output_video_path,
            fmt,
            info.fps, (info.width, info.height))
    if not writer.isOpened():
        print("failed to create a writer")
        return

    for frame_id in range(info.frame_num):
        ret, img = video.read()
        if not ret:
            print(f"failed to read image[{frame_id}]")
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = results[results["frame_id"] == frame_id]
        if not result.empty:
            pass

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img)

    time.sleep(1)

    video.release()
    writer.release()







def entry_point():
    blindpy()
