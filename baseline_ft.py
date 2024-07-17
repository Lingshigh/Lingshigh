import jittor as jt
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import argparse
from sklearn.linear_model import LogisticRegression
import numpy as np

# Enable CUDA
jt.flags.use_cuda = 0

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='A')
args = parser.parse_args()

# Load the model and preprocess
model, preprocess = clip.load("ViT-B-32.pkl")

# Read and process classes
classes_path = 'Dataset/classes.txt'
if not os.path.exists(classes_path):
    raise FileNotFoundError(f"No such file or directory: '{classes_path}'")

with open(classes_path) as f:
    classes = f.read().splitlines()

# Process class names
new_classes = []
for c in classes:
    c = c.split(' ')[0]
    if c.startswith('Animal'):
        c = c[7:]
    elif c.startswith('Thu-dog'):
        c = c[8:]
    elif c.startswith('Caltech-101'):
        c = c[12:]
    elif c.startswith('Food-101'):
        c = c[9:]
    new_classes.append(f'a photo of {c}')

print("Classes processed:", new_classes)

text = clip.tokenize(new_classes)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Load training data
imgs_dir = 'Dataset/'
train_labels_path = 'Dataset/train.txt'
if not os.path.exists(train_labels_path):
    raise FileNotFoundError(f"No such file or directory: '{train_labels_path}'")

with open(train_labels_path) as f:
    train_labels = f.read().splitlines()

train_imgs = [l.split(' ')[0] for l in train_labels]
train_labels = [jt.float32([int(l.split(' ')[1])]) for l in train_labels]

print("Training images and labels loaded.")

# Filter images to 4 per class
cnt = {}
new_train_imgs = []
new_train_labels = []
for i in range(len(train_imgs)):
    label = int(train_labels[i].numpy())
    if label not in cnt:
        cnt[label] = 0
    if cnt[label] < 4:
        new_train_imgs.append(train_imgs[i])
        new_train_labels.append(train_labels[i])
        cnt[label] += 1

print("Filtered training images and labels.")

# Calculate image features of training data
train_features = []
print('Training data processing:')
with jt.no_grad():
    for img in tqdm(new_train_imgs):
        img_path = os.path.join(imgs_dir, img)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"No such file: '{img_path}'")
        image = Image.open(img_path)
        image = preprocess(image).unsqueeze(0)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        train_features.append(image_features)

print("Training features calculated.")

train_features = jt.cat(train_features).numpy()
train_labels = jt.cat(new_train_labels).numpy()

print("Training features and labels prepared.")

# Training classifier
classifier = LogisticRegression(random_state=0,
                                C=8.960,
                                max_iter=1000,
                                verbose=1)
classifier.fit(train_features, train_labels)

print("Model trained.")

# Load testing data
split = 'TestSet'
imgs_dir = os.path.join('Dataset', split)
print(f"Testing images directory: {imgs_dir}")
if not os.path.exists(imgs_dir):
    raise FileNotFoundError(f"No such directory: '{imgs_dir}'")

# List directory contents for debugging
print(f"Contents of directory {imgs_dir}:")
print(os.listdir(imgs_dir))

test_imgs = os.listdir(imgs_dir)
if not test_imgs:
    raise FileNotFoundError(f"No images found in directory: '{imgs_dir}'")

print(f"Test images loaded: {len(test_imgs)} images found.")

# Process testing data
print('Testing data processing:')
test_features = []
with jt.no_grad():
    for img in tqdm(test_imgs):
        img_path = os.path.join(imgs_dir, img)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"No such file: '{img_path}'")
        image = Image.open(img_path)
        image = preprocess(image).unsqueeze(0)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        test_features.append(image_features)

if not test_features:
    raise ValueError("No test features calculated. Please check your test dataset.")

print("Test features calculated.")

test_features = jt.cat(test_features).numpy()

# Perform testing and save results
with open('result.txt', 'w') as save_file:
    predictions = classifier.predict_proba(test_features)
    for i, prediction in enumerate(predictions.tolist()):
        prediction = np.asarray(prediction)
        top5_idx = prediction.argsort()[-1:-6:-1]
        save_file.write(test_imgs[i] + ' ' +
                        ' '.join(str(idx) for idx in top5_idx) + '\n')

print("Results saved.")
