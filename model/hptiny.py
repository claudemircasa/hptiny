# hptiny prediction script
#
# Author: Claudemir Casa
# Copyright: C.CASA 2020. GIS Technologies
# License: Default
# Version: 1.0.4
# Maintainer: claudemircasa
# Email: claudemir.casa@ufpr.br
# Status: under development

import shutil
import random
import argparse
import numpy as np
import onnxruntime as rt
from PIL import Image, ImageDraw
from os import path, listdir, mkdir
import time
import cv2
from cv2 import dnn
import onnx
from onnx import optimizer
import uuid
import glob
from tqdm import tqdm

class HPTiny:
    def __init__(self, options):
        self.options = options
        self.image_ext = ['jpg', 'png', 'bmp']
        self.video_ext = ['mp4', 'avi']
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
        show_classes = self.options.show_classes.split()
        if (len(show_classes) > 0):
            show_classes = [int(i, base=16) for i in show_classes]
        for scores, box in zip(inferences[0], inferences[1]):

            class_id = np.argmax(scores)
            confidence = 0
            # extract class id and confidence
            if (len(show_classes) == 0):
                confidence = scores[class_id]
            else:
                if (class_id in show_classes):
                    confidence = scores[class_id]
                else:
                    continue
            
            confidence = (confidence * 100) / 1.0
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
        
        # normalize outputs because net outputs in a tiny range
        self.confidences = [ (c - min(self.confidences)) / (max(self.confidences) - min(self.confidences)) if (max(self.confidences) - min(self.confidences)) > 0.0 else 0.0 for c in self.confidences]
        self.confidences = [ (c * 100) / 10 for c in self.confidences]
        self.confidences = [ (1.0 if c > 1.0 else c) for c in self.confidences]
        idxs = dnn.NMSBoxes(self.boxes, self.confidences, self.options.confidence, self.options.threshold)

        # ensure at least one detection exists
        predicted_boxes = {}
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                if (predicted_boxes.get(self.classes[i]) is None):
                    predicted_boxes.update({self.classes[i]: []})

                # extract the bounding box coordinates
                (x, y) = (self.boxes[i][0], self.boxes[i][1])
                (w, h) = (self.boxes[i][2], self.boxes[i][3])

                # get image part
                roi = frm[y:y+h, x:x+w]

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.colors[self.classes[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                if self.options.show_confidence > 0:
                    text = "{}: {:.2f}".format(self.labels[self.classes[i]], self.confidences[i])
                else:
                    text = "{}".format(self.labels[self.classes[i]])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

                # {class_name, confidence, left, top, right, bottom}
                predicted_boxes[self.classes[i]].append([self.labels[self.classes[i]].replace(' ', '_'), self.confidences[i], x, y, x+w, y+h])

        return (frame, predicted_boxes)

    def create_detection_files(self):
        class_names = self.labels
        person, man, woman, boy, girl = (0, 1, 2, 3, 4)

        for i, _cls in enumerate(class_names):
            current_path = path.join('dataset', 'validation', _cls)

            if (path.exists(current_path)):
                files = []

                if (not path.exists(path.join('dataset','predictions'))):
                    mkdir(path.join('dataset','predictions'))

                for f in listdir(current_path):
                    if (path.isfile(path.join(current_path, f))):
                        files.append(path.join(current_path, f))

                imgs = tqdm(files)
                imgs.set_description(_cls)
                for img in imgs:
                    self.options.quiet=1
                    self.options.device=img

                    result = self.run()
                    if (result is None):
                        continue

                    image, predicted_boxes = result
                    
                    filename = path.splitext(path.basename(img))[0]
                    extension = path.splitext(path.basename(img))[1]

                    try:
                        # continue to next loop on error
                        with open('{}.txt'.format(path.join('dataset', 'predictions', filename)), 'w') as f:
                            if (i == person):
                                concat_array = []
                                if (man in predicted_boxes):
                                    for m in predicted_boxes[man]:
                                        m[0] = 'Person'
                                        concat_array.append(m)
                                if (woman in predicted_boxes):
                                    for w in predicted_boxes[woman]:
                                        w[0] = 'Person'
                                        concat_array.append(w)
                                if (boy in predicted_boxes):
                                    for b in predicted_boxes[boy]:
                                        b[0] = 'Person'
                                        concat_array.append(b)
                                if (girl in predicted_boxes):
                                    for g in predicted_boxes[girl]:
                                        g[0] = 'Person'
                                        concat_array.append(g)
                                if (i in predicted_boxes):
                                    for p in predicted_boxes[i]:
                                        p[0] = 'Person'
                                        concat_array.append(p)
                                
                                for l in concat_array:
                                    for cord in l: f.write(str(cord) + ' ')
                                    f.write('\n')
                            elif (i in predicted_boxes):
                                for l in predicted_boxes[i]:
                                    for cord in l: f.write(str(cord) + ' ')
                                    f.write('\n')
                                
                            f.close()
                        cv2.imwrite('{}{}'.format(path.join('dataset', 'predictions', filename), extension), image)
                    except:
                        continue

    def create_validation_dataset(self):
        class_names = self.labels
        for i, _cls in enumerate(class_names):
            current_path = path.join('dataset', 'train', _cls)

            if (path.exists(current_path)):
                original_files = []
                files = []

                for f in listdir(current_path):
                    if (path.isfile(path.join(current_path, f))):
                        original_files.append(f)
                
                percentage = 10
                total_validation_files = int((len(original_files) * percentage) / 100)

                for j in tqdm(range(total_validation_files)):
                    while True:
                        rindex = random.randint(0, len(original_files)-1)
                        if (not original_files[rindex] in files):
                            files.append(original_files[rindex])

                            # copy file to validation directory
                            src = path.join(current_path, original_files[rindex])
                            src_label = path.join(current_path, 'Label', path.splitext(path.basename(original_files[rindex]))[0] + '.txt')
                            dst = path.join('dataset', 'validation', _cls)
                            dst_label = path.join('dataset', 'validation', _cls, 'Label')
                            if (not path.exists(dst)):
                                makedirs(dst)
                            if (not path.exists(dst_label)):
                                makedirs(dst_label)
                            shutil.copy(src, dst)
                            shutil.copy(src_label, dst_label)
                            break

        return True

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
                image, confidences = self.predict(frame)
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
                        writer = cv2.VideoWriter('{}.avi'.format(self.options.output_name), fourcc, stream.get(cv2.CAP_PROP_FPS), (image.shape[1], image.shape[0]), True)
                    writer.write(image)

                if (not self.options.quiet):
                    cv2.imshow('', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                total += 1

            # When everything done, release the capture
            if (writer):
                writer.release()
            stream.release()
            cv2.destroyAllWindows()
        elif (_type == 1):
            image = cv2.imread(self.options.device)

            if (image is None):
                return None

            # Run detection
            start = time.time()
            image, predicted_boxes = self.predict(image)
            end = time.time()

            elapsed = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elapsed))

            if self.options.save_output > 0:
                cv2.imwrite('{}.jpg'.format(self.options.output_name), image)

            if (not self.options.quiet):
                cv2.imshow('', image)
                while True:
                    if (cv2.waitKey(1) & 0xFF == ord('q')):
                        break
            
            if (self.options.create_detection_files):
                return (image, predicted_boxes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='hptiny.py')
    parser.add_argument('device', type=str, help='device, video or image file')
    parser.add_argument('--show_confidence', type=int, default=1, help='show box prediction confidence')
    parser.add_argument('--show_classes', type=str, default='', help='list of interest classes to predict (string separated by spaces like \'0 1\')')
    parser.add_argument('--save_output', type=int, default=0, help='save the output')
    parser.add_argument('--size', type=int, default=608, help='size of net input')
    parser.add_argument('--scale', type=float, default=(1/255))
    parser.add_argument('--confidence', type=float, default=0.3)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--model', type=str, default='model.onnx')
    parser.add_argument('--output_name', type=str, default=str(uuid.uuid4()), help='name of prediction (output file)')
    parser.add_argument('--quiet', type=int, default=0, help='enable quiet mode')
    parser.add_argument('--create_validation_set', type=int, default=0, help='create validation dataset')
    parser.add_argument('--create_detection_files', type=int, default=0, help='create detection files to mAP calc')
    options = parser.parse_args()
    
    m = HPTiny(options=options)
    if (options.create_validation_set):
        m.create_validation_dataset()
    elif (options.create_detection_files):
        m.create_detection_files()
    else:
        m.run()
