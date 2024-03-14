import os

path = r"F:\Code\DeepLearning\Semantic_Segmentation\Unet\dataset\daolu\json"

jsons = [i[:-5] for i in os.listdir(path) if i.endswith("json")]
pngs = [i[:-4] for i in os.listdir(path) if i.endswith("png")]

for json in jsons:
    json_path = os.path.join(path, json + ".json")
    if json not in pngs:
        os.remove(json_path)
        print(json)
