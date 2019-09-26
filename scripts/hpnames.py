import glob, os
from tqdm import tqdm

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = 'data/hpdata'
# Percentage of images to be used for the test set
percentage_test = 10
# Create and/or truncate train.txt and test.txt
file_train = open('data/hptiny.train.txt', 'a+')
file_test = open('data/hptiny.test.txt', 'a+')
# Populate train.txt and test.txt
counter = 1  
index_test = round(100 / percentage_test)  
for pathAndFilename in tqdm(glob.iglob(os.path.join(current_dir, "*.jpg"))):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    
    if counter == index_test:
        counter = 1
        file_path = os.path.join(current_dir, title + '.txt')
        if (os.path.exists(file_path)):
            if (os.path.getsize(file_path) > 0):
                file_test.write(current_dir + "/" + title + '.jpg' + "\n")

    file_train.write(current_dir + "/" + title + '.jpg' + "\n")
    counter = counter + 1