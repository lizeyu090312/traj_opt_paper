# Co-Evolving Interpolants and Flows via Path-Flow Alignment

*Under review*

### Environment setup
Install the environment from `environment.yml`; additionally, install Flash Attention 3 wheels from this [link](https://windreamer.github.io/flash-attention3-wheels/). 

Download the FID reference from [VIRTUAL_imagenet256_labeled.npz](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) (see [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations)), store in `data/fid_ref/VIRTUAL_imagenet256_labeled.npz`.


### Downloading checkpoints
Download the checkpoints: `hf download Trajectory-Optimisation/sit-traj-opt-ckpt --repo-type model --local-dir finetuned_ckpt`. We initialise the SiT models using the `sit-**_initial.pt` checkpoints for path-flow alignment; we compare our models' FID to the best baseline FID amongst `sit-**_initial.pt` and `sit-**_final_finetune.pt`. The checkpoints `sit-**_final_finetune.pt` are initialised from `sit-**_initial.pt` and finetuned for at most an additional 12k steps using best vanilla SiT/MG finetuning settings. For each model, we use the better baseline result from its initial checkpoint and its fine-tuned checkpoint. The final checkpoint was selected from up to 12k fine-tuning steps using the SiT/MG vanilla training settings. Use `python scripts/download_sit_checkpoint.py` to download SiT's pretrained SiT-S checkpoint to `checkpoints/SiT-S-2-256_orig.pt`; this checkpoint is used to initialise the path network for non-model-guidance runs. 

Additionally, we provide cycle 3 checkpoints produced by applying our method to MG and SiT models in `finetuned_ckpt/**ours_cycle3.pt`. These checkpoints can be used to reproduce the final 50k-FID results in the paper, specifically by using `scripts/fid_driver.sh` after replacing the checkpoint paths with `finetuned_ckpt/**sit-xl-2_**ours_cycle3.pt`. 

### Data processing
Process the ImageNet data according to `preproccessing/README.md`, where the final latents are located in `data/vae-sd-ema-packed`. Please first download the ImageNet `tar` files and extract them. 


### Training using path-flow alignment (our method)
Before training, please edit the scripts to ensure the paths and SLURM parameters are correct and reflect your compute environment. 

To reproduce our method, please run `bash final_scripts/submit_sit_<model_size>.sh`, where `model_size` can be `s, b, l, xl`. We use `--gres=gpu:h200:2` for SiT-XL, SiT-XL-MG, and SiT-B-MG (on two H200 GPUs, SiT-XL path-flow alignment training and 10k-FID evaluation should take no more than 10 hours) and `--gres=gpu:h200:1` for all other configurations. 

The scripts will automatically submit a SLURM job with the appropriate hyperparameters and will perform 3 cycles of path-flow alignment as well as 10k-FID evaluation at the end of every cycle. 

#### Training using path-flow alignment (our method) with Model Guidance (MG)

To reproduce our method on MG-models, please run the following:

```bash
bash final_scripts/submit_sit_b_mg.sh
bash final_scripts/submit_sit_xl_mg.sh
```

### Generating images and computing FID
Before running `bash scripts/fid_driver.sh` to compute the FID values, change the paths in `scripts/fid_driver.sh` to point to the correct checkpoints. Use `scripts/fid_driver.sh` to reproduce the headline results. 


### Acknowledgements
This code is mainly built upon [SiT](https://github.com/willisma/SiT) and [MG](https://github.com/tzco/Diffusion-wo-CFG) repositories. 
