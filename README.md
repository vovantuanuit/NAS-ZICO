
<div align="center">
    <a href="./">
        <img src="./figs/zicoall.svg" width="79%"/>
    </a>
</div>


# Usage
Install the python environment; 
``` bash
conda env create -f environment.yml
conda activate zico
HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]
```


### Prepare Dataset
Download the Imagenet-100 from the links:
https://sutdapac-my.sharepoint.com/:u:/g/personal/vovan_tuan_sutd_edu_sg/EaWA3oLM575Nv_0mXoL7vlYBlhJ5IZvGc1YbjIjkavovUg?e=e5v6HM

UnZip and Move all downloaded folders into the `./dataset`

### ZeroCostProxy-based searching for and train ImageNet100 models, with FLOPs budget 450M

``` bash
scripts/ZiCo_NAS_ImageNet_flops450M.sh
```
noted: Change the '--zero_shot_score', example: zico, zico_from_layer_3

### Evaluate the checkpoints ZeroCostProxy-based pretrained models, with FLOPs budget 450M:

Download the checkpoints from the links:
https://sutdapac-my.sharepoint.com/:u:/g/personal/vovan_tuan_sutd_edu_sg/EWr0aqven9lMhukSqx9wFPIB4-lkB314-48fbZDR8Heq4g?e=m9OBD5

Move all downloaded folders into the `./save_dir`

``` bash
python val.py --fp16 --gpu 0 --arch ZiCo_imagenet1k_flops450M_res224_base --ckpt_path=./save_dir/ZiCo_NAS_ImageNet_flops450M_base/student_best-params_rank0.pth --data=$PATH_TO_IMAGENET
python val.py --fp16 --gpu 0 --arch ZiCo_imagenet1k_flops450M_res224_from_layer4 --ckpt_path=./save_dir/ZiCo_NAS_ImageNet_flops450M_from_layer4/student_best-params_rank0.pth --data=$PATH_TO_IMAGENET
python val.py --fp16 --gpu 0 --arch Synflow_imagenet1k_flops450M_res224_base --ckpt_path=./save_dir/Synflow_NAS_ImageNet_flops450M_base/student_best-params_rank0.pth --data=$PATH_TO_IMAGENET
python val.py --fp16 --gpu 0 --arch Synflow_imagenet1k_flops450M_res224_from_layer3 --ckpt_path=./save_dir/Synflow_NAS_ImageNet_flops450M_from_layer3/student_best-params_rank0.pth --data=$PATH_TO_IMAGENET
python val.py --fp16 --gpu 0 --arch Grad_imagenet1k_flops450M_res224_base --ckpt_path=./save_dir/Grad_NAS_ImageNet_flops450M_base/student_best-params_rank0.pth --data=$PATH_TO_IMAGENET
python val.py --fp16 --gpu 0 --arch Grad_imagenet1k_flops450M_res224_from_layer2 --ckpt_path=./save_dir/Grad_NAS_ImageNet_flops450M_from_layer2/student_best-params_rank0.pth --data=$PATH_TO_IMAGENET
```


The code is modified on `https://github.com/SLDGroup/ZiCo`

