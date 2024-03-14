import os
import shutil

'''
把文件夹的多个文件夹下的内容合并，并重新生成标号
'''

def mergeDir(path):
    dir_list = os.listdir(path)
    imgs = []
    for d in dir_list:
        dir_path = os.path.join(path,d)
        img_list = os.listdir(dir_path)


if __name__ == '__main__':
    path = 'D:\code\data\labelme_data\data'
    mergeDir(path)