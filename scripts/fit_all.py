from Brdf import Brdf
from utilities import ReadImg, WriteImg
import sys
import os
import json


# Usage: python fit_all.py [list_file_path]
if __name__ == '__main__':
    print(sys.argv)
    listPath = os.path.realpath(sys.argv[1])
    try:
        with open(listPath, 'r') as f:
            config = json.load(f)
    except Exception:
        raise Exception('Invalid list file!')

    root_path = os.path.dirname(listPath)
    greycard_path = os.path.join(root_path, config['greycard'])
    colorcard_list = [os.path.join(root_path, colorcard) for colorcard in config['colorcard_list']]
    greycard_dist = config['greycard_dist']


    brdf = Brdf()
    brdf.Calibrate(root_path, colorcard_list, greycard_path, greycard_dist, True)
    # from hdri import ShowResponseCurve
    # ShowResponseCurve(brdf.response_curve, True)


    for name in config['photos']:
        print('#############################################################')
        print(name)
        brdf.Fit(os.path.join(root_path, name, 'input.json'), n_clusters=500, n_samples=2000)
        brdf.Process(n_iter=5)
        print('#############################################################')

