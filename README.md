# Super-Resolved-Thermal-Imagery
Super-resolved Thermal Imagery for High-accuracy Facial Areas Detection and Analysis
Alicja Kwasniewska, Jacek Ruminski, Maciej Szankin, Mariusz Kaczmarek

Supplementary materials for preprint submitted to Engineering Applications of Artificial Intelligence.

## Instalation

Please run

```
conda env create -f dresnet.yml
```

to create the conda environmnet with required packages.

Then run:

```
git submodule init/update
```

and link deeply-recursive-cnn-tf as deeply_recursive_cnn_tf using

```
ln -s deeply-recursive-cnn-tf deeply_recursive_cnn_tf
```

## Training
Training scripts are not currently available, please contact authors for assistance.

## Inference

Current code supports inference using the provided DRESNet checkpoint trained on the collected thermal facial dataset using scale 2.

To run inference, use:

```
python test.py --model_name ckpts/ckpt_scale2/model_F96_D9_R3 --test_dir testing_images/
```

## Dataset

Due to privacy concerns dataset is not currently available, but please note that we are doing our best to provide it soon.

## Reference

If you find this research useful or plan to use the provided model and checkpoints please cite:

```
@article{kwasniewska2020super,
  title={Super-resolved thermal imagery for high-accuracy facial areas detection and analysis},
  author={Kwasniewska, Alicja and Ruminski, Jacek and Szankin, Maciej and Kaczmarek, Mariusz},
  journal={Engineering Applications of Artificial Intelligence},
  volume={87},
  pages={103263},
  year={2020},
  publisher={Elsevier}
}

@inproceedings{kwasniewska2019evaluating,
  title={Evaluating accuracy of respiratory rate estimation from super resolved thermal imagery},
  author={Kwasniewska, Alicja and Szankin, Maciej and Ruminski, Jacek and Kaczmarek, Mariusz},
  booktitle={2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
  pages={2744--2747},
  year={2019},
  organization={IEEE}
}
```

## Thank you note

We'd like to thank authors of https://arxiv.org/abs/1511.04491 because DRESNet was created as an enhancement of their SR model DRCN and adapted specifically for thermal images.
