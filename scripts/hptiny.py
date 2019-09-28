import argparse
import numpy as np
import onnxruntime as rt
from PIL import Image, ImageDraw
from os import path
from time import time
import cv2
from cv2 import dnn
import onnx

LABELS = [
    'Person',
    'Man',
    'Woman',
    'Boy',
    'Girl',
    'Human head',
    'Human face',
    'Human eye',
    'Human eyebrow',
    'Human nose',
    'Human mouth',
    'Human ear',
    'Human hair',
    'Human beard',
    'Human leg',
    'Human arm',
    'Human foot',
    'Human hand'
]

COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
SIZE = 608

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('device', type=str, help='device, video or image file')
    parser.add_argument('--model', type=str, default='model.onnx', help='path to model file')
    parser.add_argument('--classes', type=str, default='class.list', help='class list file')
    parser.add_argument('--confidence', type=float, default=0.01, help='minimum probability to filter weak detections')
    parser.add_argument('--threshold', type=float, default=0.01, help='threshold when applying non-maxima suppression')
    opt = parser.parse_args()

    Session = rt.InferenceSession(opt.model)
    input_name = Session.get_inputs()[0].name
    output_names = [Session.get_outputs()[0].name, Session.get_outputs()[1].name]

    source = None
    width, height = (None, None)
    if str(opt.device).endswith('.jpg'):
        source = cv2.imread(opt.device)
        (height, width) = source.shape[:2]

    blob = cv2.dnn.blobFromImage(source, scalefactor=(1 / 255.0), size=(SIZE, SIZE), swapRB=True, crop=False)

    start = time()
    inferences = Session.run(None, {input_name: blob.astype(np.float32)})
    end = time()

    print("[INFO] hptiny took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    classes = []

    # loop over each detection
    for scores, box in zip(inferences[0], inferences[1]):

        # extract class id and confidence
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        # filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
        if confidence > opt.confidence:
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
            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            classes.append(class_id)
    
    idxs = dnn.NMSBoxes(boxes, confidences, opt.confidence, opt.threshold)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classes[i]]]
            cv2.rectangle(source, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classes[i]], confidences[i])
            cv2.putText(source, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
    # show the output image
    cv2.imshow("Image", source)
    cv2.waitKey(0)
