import argparse
import sys
import time
import math

import cv2
#import mediapipe as mp
from PIL import Image
import numpy as np
#from tensorflow.lite.python.interpreter import Interpreter
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
from picamera2 import Picamera2
  
detection_result_list = []   
bbox_center_coords = (0, 0)
target_radius = 0
target_labels = ['person', 'drone']
target_acquired = False
target_all = False
min_conf_threshold=0.5

def comma_separated_strings(value):
    # This custom type function splits a string by commas and returns the resulting list
    return value.split(',')

#def extract_target_values(detection_result):
#  """Extracts the bounding box values from the highest confidence detected object of target type."""
#  global target_acquired

#  for detection in detection_result.detections:
#    categories = detection.categories
#    for category in categories:
#      if category.category_name in target_labels:
#        print('target label found:', category.category_name)
#        bbox_values = detection.bounding_box.origin_x, detection.bounding_box.origin_y, detection.bounding_box.width, detection.bounding_box.height
#        print('bbox values:', bbox_values)
#        target_acquired = True
#        return bbox_values
#  target_acquired = False 
#  return None

#def extract_detection_values(detection_result):
#  """Extracts the bounding box values from the highest confidence detection result."""
#  global target_acquired

#  if len(detection_result.detections) > 0:
#    origin_x_values = [detection.bounding_box.origin_x for detection in detection_result.detections]
#    origin_y_values = [detection.bounding_box.origin_y for detection in detection_result.detections]  
#    width_values = [detection.bounding_box.width for detection in detection_result.detections]
#    height_values = [detection.bounding_box.height for detection in detection_result.detections]
#    bbox_values = origin_x_values[0], origin_y_values[0], width_values[0], height_values[0] # assuming first vals correspond to the highest confidence   
#    target_acquired = True
#    return bbox_values
#  else:
#    target_acquired = False
#    return None
#  

def calc_bbox_center(bbox):
  # get box coordinates in (left, top, right, bottom) format

  return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2 )

def calc_target_radius(bbox):
   radius = min(bbox[2], bbox[3]) / 2
   return radius

"""
def visualize_callback(result: vision.ObjectDetectorResult, output_image: PIL.Image, timestamp_ms: int):
  global bbox_center_coords
  global target_radius
  result.timestamp_ms = timestamp_ms
  detection_result_list.append(result) ##TODO :Verify label type before appending ?

  if target_all:
    bbox_values = extract_detection_values(result)
  else:
    bbox_values = extract_target_values(result)
  
  if target_acquired:
    bbox_center_coords = calc_bbox_center(bbox_values)
    target_radius = calc_target_radius(bbox_values)
"""
def run(model_path: str, camera_id: int, width: int, height: int, target_ratio: float, device='cpu', source=None) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    target_ratio: ratio of target bbox to targetting circle.
  """

  # Load your model
  print("Loading model...")
  model = YOLO(model_path)
  print(model.names)
  print("Model loaded.")

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()
  


#   height =480
#   width=640
  middle =((width//2),(height//2))

  cam = Picamera2()
  cam.configure(cam.create_video_configuration(main={"format": 'RGB888',"size": (width, height)}))
  cam.start()

  while True:
    frame = cam.capture_array()

    results = model(frame)
    print(results)

    # Extract class IDs and names
    # View results
    #if frame is None: break
    for r in results:
        
        annotator = Annotator(frame)
        
        boxes = r.boxes
        target_conf = -999
        target_bbox = None

        for box in boxes:
            
            print(box)
            
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            conf = box.conf
            annotator.box_label(b, model.names[int(c)])

            if conf > target_conf and model.names[int(c)] == 'drone':
              target_conf = conf
              target_bbox = b
          
    frame = annotator.result()  

    # Draw center point of image
    image_center_x = middle[0]
    image_center_y = middle[1]

    if target_bbox is not None:
      # Display center pixel of bounding box   
      bbox_center_coords = calc_bbox_center(target_bbox)
      cv2.circle(
        img=frame, 
        center=(int(bbox_center_coords[0]), int(bbox_center_coords[1])), 
        radius=5, 
        color=(0, 255, 0),
        thickness=-1
      )  # Draw center point

      # compare bbox center to center of image     
      pt1 = (int(image_center_x), int(image_center_y))
      pt2 = (int(bbox_center_coords[0]), int(bbox_center_coords[1]))
      cv2.line(frame, pt1, pt2, (0, 0, 255), thickness=2) # draw line between center and target points

      # Calculate bbox offset
      bbox_x_offset = int(bbox_center_coords[0] - image_center_x)
      bbox_y_offset = int(bbox_center_coords[1] - image_center_y)
      linear_offset = math.sqrt(bbox_x_offset**2 + bbox_y_offset**2)
      print('bbox offset: ', bbox_x_offset, bbox_y_offset)
      
      # Draw targeting circle
      target_radius = calc_target_radius(target_bbox)
      target_radius_size = int(target_radius * target_ratio)
      if target_radius_size >= linear_offset:
        cv2.circle(
          img=frame, 
          center=(int(image_center_x), int(image_center_y)), 
          radius=target_radius_size, 
          color=(0, 0, 255), 
          thickness=1
        )  # Draw "target aquired" circle
      else:
        cv2.circle(
          img=frame, 
          center=(int(image_center_x), int(image_center_y)), 
          radius=target_radius_size, 
          color=(255, 0, 0), 
          thickness=1
        )  # Draw targeting circle
    else:
      cv2.circle(
          img=frame, 
          center=(int(image_center_x), int(image_center_y)), 
          radius=100, 
          color=(255, 0, 0), 
          thickness=1
      )


    cv2.waitKey(1)
    cv2.imshow('preview', frame)  


  cv2.destroyWindow("preview")
  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='yolov5nu.pt')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--targetRatio',
      help='Size of the acceptable target relative to the bounding box size.',
      required=False,
      type=float,
      default=0.5)
  parser.add_argument(
      '--targetLabels',
      help='Comma-separated list of object labels to be considered valid targets',
      required=False,
      type=comma_separated_strings,
      default="person,drone")
  parser.add_argument(
      '--targetAll',
      help='False if only targeting provided labels.',
      required=False,
      type=bool,
      default=False)
  parser.add_argument(
      '--device',
      help='Device to do inference.',
      required=False,
      type=str,
      default='cpu')
  parser.add_argument('--source',
      help='Source to stream.',
      required=False,
      type=str,
      default=None)
  args = parser.parse_args()

  global target_labels
  global target_all
  target_labels = args.targetLabels
  target_all = args.targetAll
  print("target labels:", target_labels)
  if target_all:
    print("All targets valid")

  run(
    model_path=args.model, 
    camera_id=int(args.cameraId), 
    width=args.frameWidth, 
    height=args.frameHeight, 
    target_ratio=args.targetRatio,
    device=args.device, 
    source=args.source
  )


if __name__ == '__main__':
  main()