import os
import shutil

# 各路径定义
base_path = r"D:\开放域\Dataset"
train_images_path = os.path.join(base_path, "TrainSet")
train_labels_file = os.path.join(base_path, "train.txt")
class_names_file = os.path.join(base_path, "classes.txt")
textset_path = os.path.join(base_path, "Testset")

# 打印路径以调试
print(f"Train images path: {train_images_path}")
print(f"Train labels file: {train_labels_file}")
print(f"Class names file: {class_names_file}")
print(f"Textset path: {textset_path}")

# 创建目标textset文件夹，如果不存在则创建
if not os.path.exists(textset_path):
    os.makedirs(textset_path)

# 读取类别名称
try:
    with open(class_names_file, "r", encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"文件 {class_names_file} 未找到。请检查文件路径。")
    exit()

# 创建类别文件夹并读取训练集标签
class_images = {class_name: [] for class_name in class_names}
for i, class_name in enumerate(class_names):
    class_folder = os.path.join(textset_path, f"class_{i}")
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    class_images[class_name] = []

# 读取train.txt文件，收集每个类别的图片
try:
    with open(train_labels_file, "r", encoding='utf-8') as f:
        for line in f.readlines():
            image_path, class_label = line.strip().split()
            class_label = int(class_label)
            class_name = class_names[class_label]

            # 源文件路径
            src_image_path = os.path.join(train_images_path, image_path)

            # 目标文件夹路径
            class_folder = os.path.join(textset_path, f"class_{class_label}")
            dst_image_path = os.path.join(class_folder, os.path.basename(image_path))

            # 检查源文件是否存在
            if not os.path.exists(src_image_path):
                print(f"源文件 {src_image_path} 不存在。")
                continue

            # 复制文件
            shutil.copyfile(src_image_path, dst_image_path)
            print(f"已复制图片 {src_image_path} 到 {dst_image_path}")

except FileNotFoundError:
    print(f"文件 {train_labels_file} 未找到。请检查文件路径。")
    exit()

print("Testset 文件夹和图片复制完成.")
