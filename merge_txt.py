import os
import shutil

'''
把文件夹的多个文件夹下的内容合并，并重新生成标号
'''


def mergeDir(tmp1_path, tmp2_path,save_path):
    txt = os.listdir(tmp1_path)
    for t in txt:
        txt1_path = os.path.join(tmp1_path, t)
        txt2_path = os.path.join(tmp2_path, t)
        with open(txt1_path,"r") as f:
            data1 = f.read()
            print(data1)

        with open(txt2_path,"r") as f:
            data2 = f.read()
            print(data2)

        with open(os.path.join(save_path,t),"w") as f:
            f.write(data1+data2)


if __name__ == '__main__':
    tmp1_path = r'F:\Datasets\yolo_data\txt1'
    tmp2_path = r"F:\Datasets\yolo_data\txt2"
    save_path = r"F:\Datasets\yolo_data\labels"
    mergeDir(tmp1_path, tmp2_path,save_path)
