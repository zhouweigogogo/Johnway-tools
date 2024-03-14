import os
import shutil


def removeOther(path):
    data = os.listdir(path)
    json = [i for i in data if i[-5:] == '.json']
    jpg = [i for i in data if i[-4:] == '.jpg']
    for j in jpg:
        json_name = j[:-4] + '.json'
        print(json_name)
        if json_name not in json:
            delete_file(os.path.join(path, j))
            print("已删除：", os.path.join(path, j))

def delete_file(file_path):
    if os.path.exists(file_path):
        try:
            shutil.os.remove(file_path)
            print("文件删除成功！")
        except Exception as e:
            print("删除文件出错：" + str(e))
    else:
        print("文件不存在！")

if __name__ == '__main__':
    """
    labelme：删除同一文件夹下的多余的图片
    """
    path = r'F:\Datasets\yolo_data\add_haveball\labelme'
    removeOther(path)
