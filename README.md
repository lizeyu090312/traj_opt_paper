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

MG is an opt-in extension. With the default `MG_START_STEP=-1`, `train.py`, `train_path.py`, and `final_scripts/traj_opt.sh` use the original objectives and model configuration unchanged.

To run path-flow alignment from a SiT-B/2 checkpoint already trained with the MG objective, set the checkpoint and submit the MG wrapper:

```bash
MG_CKPT=/absolute/path/to/mg_checkpoint.pt \
PROJECT_ROOT=/absolute/path/to/traj_opt_paper \
bash final_scripts/submit_sit_b_mg.sh
```

The wrapper applies MG to both parts of each alignment cycle:

- the path stage augments the teacher-tracking energy `E_track`;
- the flow stage augments the velocity target using the same conditional-minus-unconditional correction;
- the existing path architecture, path/flow mixing, checkpointing, EMA handling, and three-cycle schedule are otherwise unchanged.

The compatibility defaults mirror the MG implementation in `sit-traj-opt`: `MG_W_LO=MG_W_HI=1.45`, `MG_DROP_FRAC=0.1`, `MG_DATA_SIDE_THRESHOLD=0.75`, deterministic unconditional labels, `MG_CLASS_DROPOUT_PROB=0.0`, and `MG_LEARN_SIGMA=0`. These values can be overridden through environment variables. `MG_START_STEP=0` enables MG immediately; a positive value performs the reference warmup behavior in flow training before activating the MG target.

MG checkpoints trained by this repository use the normal model name, such as `SiT-B/2`. For a native MG-codebase SiT-XL/2 checkpoint that uses the opposite time/sign convention, select `SiT-XL/2-MG`; the adapter applies `v_ours(x,t) = -v_mg(x,1-t)` without changing the checkpoint state-dict layout.

The supplied `gen_configs/SiT-B-2_mg.sh` uses the MG evaluation settings of CFG 1.0 and 250 sampling steps. The MG wrapper additionally selects SDE Euler evaluation. Standard runs continue to use the existing ODE evaluation defaults.

### Generating images and computing FID
Before running `bash scripts/fid_driver.sh` to compute the FID values, change the paths in `scripts/fid_driver.sh` to point to the correct checkpoints. Use `scripts/fid_driver.sh` to reproduce the headline results. 


### Acknowledgements
This code is based on [SiT](https://github.com/willisma/SiT). We appreciate SiT's open-sourced checkpoints, evaluation code, and training code. 
