# UVO_Challenge

## Update
Pretrained model deleted because I have no space left on my google drive.. Please contact me directly if you need the pretrained models.

## Team Alpes_runner Solutions
This is an official repo for our UVO Challenge solutions for Image/Video-based open-world segmentation. Our team "Alpes_runner" achieved the **best performance** on both Image/Video-based benchmarks. More details about the workshop can be found [here](https://sites.google.com/view/unidentified-video-object/home?authuser=0).

## Technical Reports
- For Track 1: [paper](https://arxiv.org/abs/2110.10239)
- For Track 2: [paper](https://arxiv.org/abs/2110.11661)

## Models
**Detection**
| Model                      | Pretrained datasets | Finetuned datasets | links   |
|----------------------------|------|---------|---------|
| UVO_Detector     | COCO | - |[config](./Track1/detection/configs/uvo/swin_l_carafe_simota_focal_giou_iouhead_tower_dcn_coco_384.py)/[weights](https://drive.google.com/file/d/1coV2E0qF13t4aEUT_9i2f1Dpcwk1e1rK/view?usp=sharing)|
| UVO_Detector     | COCO | UVO |[config](./Track1/detection/configs/uvo/swin_l_carafe_simota_focal_giou_iouhead_tower_dcn_coco_384_uvo_finetune.py)/[weights]()|

**Segmentation**
| Model                      | Pretrained datasets | Finetuned datasets | links   |
|----------------------------|------|---------|---------|
| UVO_Segementor    | COCO | - | [weights](https://drive.google.com/file/d/1oWkJA10VTUoEaRkoaPDz_gnQ5Bg67oHx/view?usp=sharing)|
| UVO_Segmentor     | COCO, PASCAL, OpenImage| - |[config](./Track1/segmentation/configs/swin/swin_l_upper_w_jitter_training.py)/[weights](https://1drv.ms/u/s!Ar4uxu1EELfHdafV-y_AWo5sJR0?e=9YLc8m)|
| UVO_Segmentor     | COCO, PASCAL, OpenImage | UVO|[config](./Track1/segmentation/configs/swin/swin_l_upper_w_jitter_uvo_finetune_training.py)/[weights]()|

## Citation
If you find this project useful in your research, please consider cite:
```bash
@article{du20211st,
  title={1st Place Solution for the UVO Challenge on Image-based Open-World Segmentation 2021},
  author={Du, Yuming and Guo, Wen and Xiao, Yang and Lepetit, Vincent},
  journal={arXiv preprint arXiv:2110.10239},
  year={2021}
}

@article{du20211st,
  title={1st Place Solution for the UVO Challenge on Video-based Open-World Segmentation 2021},
  author={Du, Yuming and Guo, Wen and Xiao, Yang and Lepetit, Vincent},
  journal={arXiv preprint arXiv:2110.11661},
  year={2021}
}
```

## Contact
Feel free to contact [me](yuming.du@enpc.fr) or open a new issue if you have any questions.
