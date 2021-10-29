# UVO_Challenge
This is an official repo for our UVO Challenge solutions for Video-based open-world segmentation.
Code based on [videowalk](https://github.com/ajabri/videowalk)

## Before Start
Make sure that you saved your **per-frame** predictions in a folder, with the structure described below:
```
$YOUR_FOLDER/
    00cwEcZZcu4/
        30.json
        31.json
        32.json
        ...
    2yB9Ajv5OR8/
    ...
```
the prediction of **each frame** is saved in COCO format, described [here](https://cocodataset.org/#format-results).

## Citation
If you find this project useful in your research, please consider cite:
```bash
@article{du20211st,
  title={1st Place Solution for the UVO Challenge on Video-based Open-World Segmentation 2021},
  author={Du, Yuming and Guo, Wen and Xiao, Yang and Lepetit, Vincent},
  journal={arXiv preprint arXiv:2110.11661},
  year={2021}
}
```
