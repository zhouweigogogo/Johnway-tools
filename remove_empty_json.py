import json
import os
import shutil

rm_list = []


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
    path = 'F:\Datasets\yolo_data\\add_haveball\labelme\\'  # 文件路径
    dirs = os.listdir(path)

    for file in dirs:  # 循环读取路径下的文件并筛选输出
        if os.path.splitext(file)[1] == ".json":  # 筛选Json文件
            print("path = ", file)  # 此处file为json文件名，之前修改为与图片jpg同名
            # print(os.path.join(path,file))
            with open(os.path.join(path, file), 'r') as load_f:  # 若有中文，可将r改为rb
                load_dict = json.load(load_f)  # 用json.load()函数读取文件句柄，可以直接读取到这个文件中的所有内容，并且读取的结果返回为python的dict对象
            n = len(load_dict)  # 获取字典load_dict中list值
            print('n = ', n)
            print("shapes = ",
                  load_dict['shapes'])
            if len(load_dict['shapes']) == 0:
                delete_file(os.path.join(path,file))


