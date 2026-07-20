# Co-Evolving Interpolants and Flows via Path-Flow Alignment

*Under review*

### Environment setup
Install the environment from `environment.yml`; additionally, install Flash Attention 3 wheels from this [link](https://windreamer.github.io/flash-attention3-wheels/). 

Download the FID reference from [VIRTUAL_imagenet256_labeled.npz](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) (see [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations)), store in `data/fid_ref/VIRTUAL_imagenet256_labeled.npz`.


### Downloading checkpoints
Download the finetuned checkpoints: `hf download Trajectory-Optimisation/sit-traj-opt-ckpt --repo-type model --local-dir finetuned_ckpt`. We initialise the SiT models using the `sit-**_initial.pt` checkpoints for path-flow alignment; we compare our models' FID to the FID of `sit-**_final_finetune` checkpoints, which are initialised from `sit-**_initial.pt` and finetuned for an additional 12k steps using best vanilla SiT finetuning settings. Use `python scripts/download_sit_checkpoint.py` to download SiT's pretrained SiT-S checkpoint to `checkpoints/SiT-S-2-256_orig.pt`; this checkpoint is used to initialise the path network. 

### Data processing
Process the ImageNet data according to `preproccessing/README.md`, where the final latents are located in `data/vae-sd-ema-packed`. Please first download the ImageNet `tar` files and extract them. 


### Training using path-flow alignment (our method)
Before training, please edit the scripts to ensure the paths and SLURM parameters are correct and reflect your compute environment. 

To reproduce our method, please run `bash final_scripts/submit_sit_<model_size>.sh`, where `model_size` can be `s, b, l, xl`. We use `--gres=gpu:h200:2` for SiT-XL (on two H200 GPUs, SiT-XL path-flow alignment training and 10k-FID evaluation should take no more than 10 hours) and `--gres=gpu:h200:1` for all other model configurations. 

The scripts will automatically submit a SLURM job with the appropriate hyperparameters and will perform 3 cycles of path-flow alignment as well as 10k-FID evaluation at the middle and end of every cycle. 

#### Model Guidance (MG) compatibility

To train with Model Guidance, with the default `MG_START_STEP=-1`, `train.py`, `train_path.py`, and `final_scripts/traj_opt.sh` use the original objectives and model configuration unchanged.

The MG wrappers use `finetuned_ckpt/sit-b-2-mg_initial.pt` and `finetuned_ckpt/sit-xl-2-mg_initial.pt`. The SiT-XL/2-MG wrapper uses the B/2 MG checkpoint to initialize the path model and the XL/2 MG checkpoint to initialize the flow model and cycle-1 teacher:

```bash
bash final_scripts/submit_sit_b_mg.sh
bash final_scripts/submit_sit_xl_mg.sh
```

### Generating images and computing FID
Before running `bash scripts/fid_driver.sh` to compute the FID values, change the paths in `scripts/fid_driver.sh` to point to the correct checkpoints. Use `scripts/fid_driver.sh` to reproduce the headline results. 


### Acknowledgements
This code is based on [SiT](https://github.com/willisma/SiT). We appreciate SiT's open-sourced checkpoints, evaluation code, and training code. 
