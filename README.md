# CoolNet: Cross-modal fine-grained alignment and fusion network for multimodal aspect-based sentiment analysis

This repository contains the source code and datasets associated with the paper titled "Cross-modal fine-grained alignment and fusion network for multimodal aspect-based sentiment analysis."

<p align="center">
  <img src="./chimera.png" width="800"/>
</p>


## Data

- Step 1：Download each tweet's associated images via this link [Google Drive](https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view), and then put the associated images into folders "./datasets/twitter2015_images/" and "./datasets/twitter2017_images/";
- Step 2: Download each finetune file via this link [Google Drive](https://drive.google.com/drive/folders/1nRHti6jdcrZOkh6PaqnpgdllAxjUuMCO?usp=sharing), and then put the associaled finetune model files into folder "./finetune/roberta_15/final/" and "./finetune/roberta_17/final/"
- Step 3: Download the pre-trained roberta-base-cased and put the pre-trained roberta model under the folder "./model/roberta-base-cased/"
- Step 4: Download the VIG checkpoints [vig_s_80.6](https://objects.githubusercontent.com/github-production-release-asset-2e65be/222110214/6aed88e9-a18a-4d4b-94ef-ef7b263146a1?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20241221%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241221T090626Z&X-Amz-Expires=300&X-Amz-Signature=bac5ecb4aecd6064ede090b6ae0c3ea5b9eec8f43a4757ae130f0a72cec22111&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dvig_s_80.6.pth&response-content-type=application%2Foctet-stream) and put it under the folder "./"


## Requirement
conda env create > CoolNet.yaml



### Training for CoolNet
- python solve_final.py




###  References

```

@article{xiao2023cross,
  title={Cross-modal fine-grained alignment and fusion network for multimodal aspect-based sentiment analysis},
  author={Xiao, Luwei and Wu, Xingjiao and Yang, Shuwen and Xu, Junjie and Zhou, Jie and He, Liang},
  journal={Information Processing \& Management},
  volume={60},
  number={6},
  pages={103508},
  year={2023},
  publisher={Elsevier}
}


```


### Acknowledgements
- Using these two datasets means you have read and accepted the copyrights set by Twitter and dataset providers.
- Most of the codes are based on the codes provided by huggingface: https://github.com/huggingface/transformers.
