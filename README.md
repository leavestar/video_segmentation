OverLeaf Paper link:
https://www.overleaf.com/16187850bxhxfhtvsdbr#/61913142/

Github Repo Link:
https://github.com/hyuna915/video_segmentation

Davis Challenge References

Website http://davischallenge.org/index.html
2017 https://arxiv.org/abs/1704.00675
2018 https://arxiv.org/abs/1803.00557
Python repo for DAVIS: https://github.com/fperazzi/davis/tree/master/python

OnAVOS: https://www.vision.rwth-aachen.de/publication/00158/
https://arxiv.org/pdf/1706.09364.pdf
OSVOS: http://openaccess.thecvf.com/content_cvpr_2017/papers/Caelles_One-Shot_Video_Object_CVPR_2017_paper.pdf
http://people.ee.ethz.ch/~cvlsegmentation/osvos/
https://github.com/scaelles/OSVOS-TensorFlow
MSF:
https://graphics.ethz.ch/~perazzif/masktrack/files/masktrack.pdf

Mask RCNN: https://github.com/matterport/Mask_RCNN


Blog reading:
https://medium.com/@eddiesmo/a-meta-analysis-of-davis-2017-video-object-segmentation-challenge-c438790b3b56


Davis Environment setup
Verified works well for both Mac and Google cloud

create a virtualenv in python2.7
```
conda create -n final python=2.7 anaconda
source activate final
```
attempt to install all requirements.txt, mostly likely failed
```
conda install --yes --file requirements.txt
```
for failed dependency, search in https://anaconda.org/search, find the right version for your os (max or linux).
For example if functools32 prettytable easydict is not found
```
conda install -c conda-forge functools32
conda install -c conda-forge prettytable
conda install -c chembl easydict
```
install all the other requirements one by one
```
while read requirement; do conda install --yes $requirement; done < requirements.txt
```