# UVO_Challenge
This is an official repo for our UVO Challenge solutions for Image/Video-based open-world segmentation.

## Models
**Detection**
| Model                      | Pretrained datasets | Finetuned datasets | links   |
|----------------------------|------|---------|---------|
| UVO_Detector     | COCO | - |[config](./Track1/detection/configs/uvo/swin_l_carafe_simota_focal_giou_iouhead_tower_dcn_coco_384.py)/[weights](https://drive.google.com/file/d/1SmRUdYljUmYxLkbwfdJqys_lWPu7kN3q/view?usp=sharing)|
| UVO_Detector     | COCO | UVO |[config](./Track1/detection/configs/uvo/swin_l_carafe_simota_focal_giou_iouhead_tower_dcn_coco_384_uvo_finetune.py)/[weights](https://drive.google.com/file/d/1VdtZ6D34VlaUprUqoxRIiLj2ab5hLOaZ/view?usp=sharing)|

**Segmentation**
| Model                      | Pretrained datasets | Finetuned datasets | links   |
|----------------------------|------|---------|---------|
| UVO_Segementor    | COCO | - | [weights](https://drive.google.com/file/d/1VC7oS1x6ttQ4t-Px3r8BfbQP7FfhgRj-/view?usp=sharing)|
| UVO_Segmentor     | COCO, PASCAL, OpenImage| - |[config](./Track1/segmentation/configs/swin/swin_l_upper_w_jitter_training.py)/[weights](https://drive.google.com/file/d/1XKpm_VLLJ0mkgN9ZTDErG-4XuQHib_GK/view?usp=sharing)|
| UVO_Segmentor     | COCO, PASCAL, OpenImage | UVO|[config](./Track1/segmentation/configs/swin/swin_l_upper_w_jitter_uvo_finetune_training.py)/[weights](https://drive.google.com/file/d/10EeNkJkFRaPNhGeRAegrSdY-VslrgFrZ/view?usp=sharing)|

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
