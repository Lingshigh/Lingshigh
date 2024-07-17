import os

# 定义要检查的文件路径列表
file_paths = [
    "D:/开放域/Dataset/TrainSet/Animal/Bear/1.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Bear/2.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Bear/3.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Bear/4.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Bee/1.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Bee/2.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Bee/3.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Bee/4.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Bird/1.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Bird/2.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Bird/3.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Bird/4.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Butterfly/1.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Butterfly/2.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Butterfly/3.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Butterfly/4.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Camel/1.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Camel/2.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Camel/3.jpg",
    "D:/开放域/Dataset/TrainSet/Animal/Camel/4.jpg",
]

# 检查文件是否存在
for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"源文件 {file_path} 不存在。")
    else:
        print(f"源文件 {file_path} 存在。")
