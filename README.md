## PAMD
**[PAMD: Plausibility-Aware Motion Diffusion Model for Long Dance Generation](https://www.arxiv.org/abs/2505.20056), [Arxiv](https://arxiv.org/html/2505.20056v1)**<br>
*Abstract: Diffusion-based music-to-dance generation has made significant progress, yet existing methods still struggle to produce physically plausible motions. To address this, we propose Plausibility-Aware Motion Diffusion (PAMD), a framework for generating dances that are both musically aligned and physically realistic. The core of PAMD lies in the Plausible Motion Constraint (PMC), which leverages Neural Distance Fields (NDFs) to model the actual pose manifold and guide generated motions toward a physically valid pose manifold. To provide more effective guidance during generation, we incorporate Prior Motion Guidance (PMG), which uses standing poses as auxiliary conditions alongside music features. To further enhance realism for complex movements, we introduce the Motion Refinement with Foot-ground Contact (MRFC) module, which addresses foot-skating artifacts by bridging the gap between the optimization objective in linear joint position space and the data representation in nonlinear rotation space. Extensive experiments show that PAMD significantly improves musical alignment and enhances the physical plausibility of generated motions.*
## Requirements
* We follow the environment configuration of [EDGE](https://github.com/Stanford-TML/EDGE) 

## Chekpoint
* Download the saved model checkpoint from [Google Drive](https://drive.google.com/file/d/1kC_nK1eLYkGLDTVFofC1rLyvLFHzB9ue/view?usp=drive_link).

## Dataset Download
Download and process the AIST++ dataset (wavs and motion only) using:
```.bash
cd data
bash download_dataset.sh
python create_dataset.py --extract-baseline --extract-jukebox
```
This will process the dataset to match the settings used in the paper. The data processing will take ~24 hrs and ~50 GB to precompute all the Jukebox features for the dataset.

## Training
Once the AIST++ dataset is downloaded and processed, run the training script, e.g.
```.bash
accelerate launch train.py --batch_size 128  --epochs 2000 --feature_type jukebox --learning_rate 0.0002
```
to train the model with the settings from the paper. The training will log progress to `wandb` and intermittently produce sample outputs to visualize learning.

## Testing and  Evaluation
Download the long music from [Google Drive](https://drive.google.com/file/d/1d2sqwQfW3f4XcNyYx3oWXdDQphrhfokj/view?usp=drive_link).

Evaluate your model's outputs with the Beat Align Score, PFC, FID, Diversity score proposed in the paper:
1. Generate ~1k samples, saving the joint positions with the `--save_motions` argument
2. Run the evaluation script
```.bash
python test.py --music_dir custom_music/ --save_motions
python eval/beat_align_score.py
python eval/eval_pfc.py
python eval/metrics_diveristy.py
```

## Citation
```
@article{wang2025pamd,
  title={PAMD: Plausibility-Aware Motion Diffusion Model for Long Dance Generation},
  author={Wang, Hongsong and Zhu, Yin and Lai, Qiuxia and Zhang, Yang and Xie, Guo-Sen and Geng, Xin},
  journal={arXiv preprint arXiv:2505.20056},
  year={2025}
}
```
