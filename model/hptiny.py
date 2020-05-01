# hptiny prediction script
#
# Author: Claudemir Casa
# Copyright: Copyright 2019. GIS Technologies
# License: GIS
# Version: 1.0.4
# Mmaintainer: claudemircasa
# Email: claudemir.casa@ufpr.br
# Status: under development

from threading import Thread
from queue import Queue
import argparse
import numpy as np
import onnxruntime as rt
from PIL import Image, ImageDraw
from os import path
import time
import cv2
from cv2 import dnn
import onnx
from onnx import optimizer
import uuid
import glob

class HPTiny:
    def __init__(self, options):
        self.options = options
        self.image_ext = ['jpeg', 'jpg', 'jpe', 'jp2', 'png', 'bmp', 'dib', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'tiff', 'tif']
        self.video_ext = ['mp4', 'avi', 'webm']
        self.boxes = []
        self.confidences = []
        self.classes = []
        
        with open('classes') as f:
            self.labels = f.read().strip().split("\n")
            self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")

        self.model = onnx.load(self.options.model)
        self.session = rt.InferenceSession(self.options.model)
        self.inputs = self.session.get_inputs()[0].name
        self.outputs = [o for o in self.session.get_outputs()]

    def check(self):
        return onnx.checker.check_model(self.model)
    
    def readable(self):
        return onnx.helper.printable_graph(self.model.graph)
    
    def optimize(self):
        # A full list of supported optimization passes can be found using get_available_passes()
        all_passes = optimizer.get_available_passes()
        print("Available optimization passes:")
        for p in all_passes:
            print(p)
        print()

        # Apply the optimization on the original model
        optimized_model = optimizer.optimize(self.model, all_passes)

        # save new model
        onnx.save(optimized_model, 'optimized_model.onnx')

    def predict(self, frm):
        frame = frm.copy()

        (height, width) = frame.shape[:2]
        resized = cv2.resize(frame, (self.options.size, self.options.size))
        
        blob = cv2.dnn.blobFromImage(resized, scalefactor=self.options.scale, size=(self.options.size, self.options.size), mean=(0,0,0), swapRB=True, crop=False)
        inferences = self.session.run(None, {self.inputs: blob.astype(np.float32)})

        self.boxes = []
        self.confidences = []
        self.classes = []

        # loop over each detection
        for scores, box in zip(inferences[0], inferences[1]):

            # extract class id and confidence
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > self.options.confidence:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                _box = box[0:4] * np.array([width, height, width, height])
                (x, y, w, h) = _box.astype('int')

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(x - (w / 2))
                y = int(y - (h / 2))
                
                # update our list of bounding box coordinates, confidences,
                # and class ids
                self.boxes.append([x, y, int(w), int(h)])
                self.confidences.append(float(confidence))
                self.classes.append(class_id)
        
        idxs = dnn.NMSBoxes(self.boxes, self.confidences, self.options.confidence, self.options.threshold)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (self.boxes[i][0], self.boxes[i][1])
                (w, h) = (self.boxes[i][2], self.boxes[i][3])

                # get image part
                roi = frm[y:y+h, x:x+w]
                mask = np.zeros((roi.shape[:2][0], roi.shape[:2][1], 3), np.uint8)

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.colors[self.classes[i]]]

                mask[:] = color
                #mask[:, :, :3] = color
                #mask[:, :, 3:] = 100

                #frame[y:y+h, x:x+w] = mask

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                if self.options.show_percentage > 0:
                    text = "{}: {:.4f}".format(self.labels[self.classes[i]], self.confidences[i])
                else:
                    text = "{}".format(self.labels[self.classes[i]])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

        return frame

    def run(self):
        _type = 0
        # detect from camera
        for i, j in zip(self.image_ext, self.video_ext):
            if (self.options.device.endswith(i)):
                _type = 1
                break
            elif (self.options.device.endswith(j)):
                _type = 2
                break
        
        if (_type == 0 or _type == 2):
            if (_type == 2):
                stream = cv2.VideoCapture(self.options.device)
            else:
                stream = cv2.VideoCapture(int(self.options.device))
            writer = None

            total = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
            while stream.isOpened():
                _, frame = stream.read()

                # Run detection
                start = time.time()
                pimage = self.predict(frame)
                end = time.time()

                # some information on processing single frame
                elapsed = (end - start)
                fps = 1 / elapsed
                if (fps < 1):
                    fps = 1

                print("[INFO] FPS: {:.2f} seconds".format(fps))
                print("[INFO] single frame took {:.4f} seconds".format(elapsed))
                if (_type == 2):
                    print("[INFO] estimated total time to finish: {:.4f}".format(elapsed * total))

                # write the output frame to disk
                if self.options.save_output > 0:
                    # check if the video writer is None
                    if writer is None:
                        # initialize our video writer
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        writer = cv2.VideoWriter('{}.avi'.format(self.options.output_name), fourcc, 15, (pimage.shape[1], pimage.shape[0]), True)
                    writer.write(pimage)

                cv2.imshow('', pimage)
                total += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # When everything done, release the capture
            if (writer):
                writer.release()
            stream.release()
            cv2.destroyAllWindows()
        elif (_type == 1):
            image = cv2.imread(self.options.device)

            # Run detection
            start = time.time()
            image = self.predict(image)
            end = time.time()

            elapsed = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elapsed))

            if self.options.save_output > 0:
                cv2.imwrite('{}.jpg'.format(self.options.output_name), image)

            cv2.imshow('', image)
            while True:
                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('device', type=str, help='device, video or image file')
    parser.add_argument('--show_percentage', type=int, default=1, help='show box prediction percentage')
    parser.add_argument('--save_output', type=int, default=0, help='save the output')
    parser.add_argument('--size', type=int, default=608, help='size of net input')
    parser.add_argument('--scale', type=float, default=(1/255))
    parser.add_argument('--confidence', type=float, default=0.01)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--model', type=str, default='model.onnx')
    parser.add_argument('--output_name', type=str, default=str(uuid.uuid4()))
    options = parser.parse_args()
    
    m = HPTiny(options=options)
    m.run()
