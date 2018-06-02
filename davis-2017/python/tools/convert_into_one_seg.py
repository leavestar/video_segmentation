import yaml
import os
from   davis import io
import numpy as np

def convert(seq_name):
  try:
    result_path = os.path.join(ROOT_DIR, 'davis-2017', 'data', 'DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS2', seq_name)
    SAVE_DIR = os.path.join(ROOT_DIR, 'davis-2017', 'data', 'DAVIS', 'Results', 'Segmentations', '480p', "OSVOS2-convert", seq_name)

    N_OBJECT = len([_ for _ in os.listdir(result_path) if not _.startswith(".")])
    TOTAL_IMAGE = sorted([_ for _ in os.listdir(os.path.join(result_path, "1")) if _.endswith(".png")])


    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        for image in TOTAL_IMAGE:
            for n_object in range(1, N_OBJECT + 1):
                # import pdb; pdb.set_trace()
                current_annatation = os.path.join(result_path, str(n_object), image)
                an, _ = io.imread_indexed(current_annatation)

                # if this is the first object, init base image
                if n_object == 1:
                    base_image = np.zeros_like(an).astype('uint8')
                base_image[an != 0] = n_object

            #   by this time we have updated base_image succefully
            custom_anno = os.path.join(SAVE_DIR, image)
            io.imwrite_indexed(custom_anno, base_image)
    else:
        print SAVE_DIR + " exists! continue"
  except:
    print "exception!" + seq_name



ROOT_DIR = "../../"
path = os.path.join(ROOT_DIR, 'davis-2017', 'data', 'db_info.yaml')
stream = file(path, 'r')
dict_ = yaml.load(stream)
seq = dict_['sequences']
for item in seq:
    print item['name']
    convert(item['name'])

