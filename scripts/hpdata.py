import os
import argparse
import ntpath
import json
import shutil
import re
import cv2
from tqdm import tqdm

CLASSES_MAP = [
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
    'Human hand',
    #false positive classes
    'Animal',
    'Building',
    'Food',
    'Furniture',
    'Tool',
    'Vehicle',
    'Door',
    'Drink',
    'Telephone',
    'Weapon',
    'Toy',
    'Tableware',
    'Container',
    'Helmet',
    'Racket',
    'Bicycle',
    'Ball',
    'Cosmetics'
]

def parser_arguments():
    parser = argparse.ArgumentParser(description='hp data format')
    
    parser.add_argument('--input', required=False, default=".", help='Root directory of the entire dataset')
    parser.add_argument('--output', required=False, default="./hpdata", help='Output directory')

    return parser.parse_args()

if __name__ == '__main__':
    args = parser_arguments()

    if (not os.path.exists(args.output)):
        os.mkdir(args.output)

    for root, dirs, files in os.walk(args.input):
        for d in dirs:
            if (d in CLASSES_MAP):
                for file in tqdm(os.listdir(d)):
                    if os.path.isfile(os.path.join(d, file)):
                        if (file.split('.')[1] == 'jpg'):
                            file_name = str(file.split('.')[0]) + '.txt'
                            file_path = os.path.join(d, 'Label', file_name)

                            file_annotations = []
                            if (os.path.exists(file_path)):
                                f = open(file_path, 'r')
                                image = cv2.imread(os.path.join(d, file))

                                for line in f:
                                    line_annotations = []

                                    match_class_name = re.compile('^[a-zA-Z]+(\s+[a-zA-Z]+)*').match(line)
                                    class_name = line[:match_class_name.span()[1]]
                                    # XMin, YMin, XMax, YMax
                                    ax = line[match_class_name.span()[1]:].lstrip().rstrip().split(' ')
                                    ax[0] = float(ax[0]) / image.shape[1]
                                    ax[1] = float(ax[1]) / image.shape[0]
                                    ax[2] = float(ax[2]) / image.shape[1]
                                    ax[3] = float(ax[3]) / image.shape[0]

                                    # width, height, x, y
                                    # [ XMax - XMin, YMax - YMin, (XMax + XMin) / 2, (YMax + YMin) / 2 ]
                                    axf = [ax[2] - ax[0], ax[3] - ax[1], (ax[0] + ax[2]) / 2, (ax[1] + ax[3]) / 2]
                                    axfn = [axf[2], axf[3], axf[0], axf[1]]

                                    index = CLASSES_MAP.index(class_name)
                                    if (index):
                                        line_annotations.append(index)
                                        line_annotations.extend(axfn)

                                        file_annotations.append(line_annotations)
                                    else:
                                        continue

                            txt_file = open(os.path.join(args.output, '{0}.txt'.format(file.split('.')[0])), 'a+')
                            txt_file_lines = [l.rstrip('\n') for l in txt_file] 
                            for l in file_annotations:
                                lstr = '{0} {1} {2} {3} {4}'.format(l[0], l[1], l[2], l[3], l[4])
                                if (not lstr in txt_file_lines):
                                    txt_file.write('{0}\n'.format(lstr))

                            txt_file.close()

                            # copy image to output folder
                            shutil.copy(os.path.join(d, file), args.output)

