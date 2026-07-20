# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models import SiT_models
from sit_utils.download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from sit_utils.train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import argparse
import sys


UINT64_MASK = (1 << 64) - 1
TORCH_SEED_MODULUS = (1 << 63) - 1


def unwrap_checkpoint(obj):
    if isinstance(obj, dict):
        if "ema" in obj:
            return obj["ema"]
        if "model" in obj:
            return obj["model"]
    return obj


def infer_learn_sigma_from_state_dict(state_dict, model_name):
    patch_size = int(model_name.split("/")[-1].split("-")[0])
    out_dim = state_dict["final_layer.linear.weight"].shape[0]
    no_sigma_dim = (patch_size ** 2) * 4
    sigma_dim = (patch_size ** 2) * 8
    if out_dim == sigma_dim:
        return True
    if out_dim == no_sigma_dim:
        return False
    raise ValueError(
        f"Could not infer learn_sigma from final head shape {out_dim} for model {model_name}"
    )


def sample_seed_for_index(global_seed, index):
    """Derive a stable torch seed from the user seed and canonical image index."""
    x = (int(global_seed) + int(index) + 0x9E3779B97F4A7C15) & UINT64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    x = (x ^ (x >> 31)) & UINT64_MASK
    return x % TORCH_SEED_MODULUS


def seed_torch_for_index(global_seed, index):
    seed = sample_seed_for_index(global_seed, index)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def canonical_sample_name(index):
    return f"{index:06d}.png"


def canonical_sample_path(sample_dir, index):
    return os.path.join(sample_dir, canonical_sample_name(index))


def sample_folder_name_for_args(mode, args):
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    if args.sample_folder_name is not None:
        return args.sample_folder_name
    if mode == "ODE":
        return (
            f"{model_string_name}-{ckpt_string_name}-"
            f"cfg-{args.cfg_scale}-{args.per_proc_batch_size}-"
            f"{mode}-{args.num_sampling_steps}-{args.sampling_method}"
        )
    if mode == "SDE":
        return (
            f"{model_string_name}-{ckpt_string_name}-"
            f"cfg-{args.cfg_scale}-{args.per_proc_batch_size}-"
            f"{mode}-{args.num_sampling_steps}-{args.sampling_method}-"
            f"{args.diffusion_form}-{args.last_step}-{args.last_step_size}"
        )
    raise ValueError(f"Invalid sampling mode: {mode}")


def is_valid_sample_png(path, image_size):
    try:
        with Image.open(path) as img:
            if img.mode != "RGB":
                return False
            if img.size != (image_size, image_size):
                return False
            img.load()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def scan_sample_folder(sample_dir, total_samples, image_size):
    present_indices = set()
    unexpected_pngs = []
    for name in os.listdir(sample_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() != ".png":
            continue
        if not stem.isdigit():
            unexpected_pngs.append(name)
            continue
        index = int(stem)
        if name != canonical_sample_name(index) or index >= total_samples:
            unexpected_pngs.append(name)
            continue
        if os.path.isfile(os.path.join(sample_dir, name)):
            present_indices.add(index)

    valid_indices = sorted(present_indices)
    pending_indices = [index for index in range(total_samples) if index not in present_indices]
    invalid_indices = {
        index
        for index in valid_indices[-10:]
        if not is_valid_sample_png(canonical_sample_path(sample_dir, index), image_size)
    }
    if invalid_indices:
        valid_indices = [index for index in valid_indices if index not in invalid_indices]
        pending_indices.extend(sorted(invalid_indices))
        pending_indices.sort()

    return {
        "valid_indices": valid_indices,
        "pending_indices": pending_indices,
        "unexpected_pngs": sorted(unexpected_pngs),
    }


def format_index_preview(indices, limit=10):
    if not indices:
        return "none"
    preview = ", ".join(str(index) for index in indices[:limit])
    if len(indices) > limit:
        preview += f", ... ({len(indices)} total)"
    return preview


def scan_error_message(scan_result, sample_dir, total_samples):
    unexpected_pngs = scan_result["unexpected_pngs"]
    if unexpected_pngs:
        preview = ", ".join(unexpected_pngs[:10])
        if len(unexpected_pngs) > 10:
            preview += f", ... ({len(unexpected_pngs)} total)"
        return (
            f"Unexpected PNG files found in {sample_dir}: {preview}. "
            f"Expected only canonical files 000000.png through {total_samples - 1:06d}.png."
        )
    return None


def broadcast_object(obj, src=0):
    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def make_index_generator(global_seed, index):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sample_seed_for_index(global_seed, index))
    return generator


def make_batch_inputs(batch_indices, args, model, latent_size, device):
    z_parts = []
    y_parts = []
    for index in batch_indices:
        generator = make_index_generator(args.global_seed, index)
        z_parts.append(
            torch.randn(
                1,
                model.in_channels,
                latent_size,
                latent_size,
                generator=generator,
            ).to(device)
        )
        y_parts.append(
            torch.randint(
                0,
                args.num_classes,
                (1,),
                generator=generator,
            ).to(device)
        )
    return torch.cat(z_parts, dim=0), torch.cat(y_parts, dim=0)


def make_single_index_inputs(index, args, model, latent_size, device):
    seed_torch_for_index(args.global_seed, index)
    z = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)
    y = torch.randint(0, args.num_classes, (1,), device=device)
    return z, y


def save_sample_atomic(sample, path, rank):
    tmp_path = f"{path}.tmp-rank{rank}-pid{os.getpid()}"
    try:
        Image.fromarray(sample).save(tmp_path, format="PNG")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(mode, args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", rank % torch.cuda.device_count())
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(
        f"Starting rank={rank}, seed={seed}, global_seed={args.global_seed}, "
        f"world_size={dist.get_world_size()}."
    )

    # Create folder to save samples and decide resume work before loading heavy model/VAE state.
    folder_name = sample_folder_name_for_args(mode, args)
    sample_folder_dir = os.path.join(args.sample_dir, folder_name)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    n = args.per_proc_batch_size
    total_samples = args.num_fid_samples
    scan_result = None
    if rank == 0:
        scan_result = scan_sample_folder(sample_folder_dir, total_samples, args.image_size)
        error_message = scan_error_message(scan_result, sample_folder_dir, total_samples)
        scan_result["error_message"] = error_message
    scan_result = broadcast_object(scan_result, src=0)
    if scan_result["error_message"] is not None:
        raise RuntimeError(scan_result["error_message"])

    pending_indices = scan_result["pending_indices"]
    valid_count = len(scan_result["valid_indices"])
    pending_for_rank = pending_indices[rank::dist.get_world_size()]
    if rank == 0:
        print(f"Found {valid_count} existing target PNG samples in {sample_folder_dir}")
        print(f"Target total number of images: {total_samples}")
        print(f"Pending missing/invalid indices: {format_index_preview(pending_indices)}")
    if not pending_indices:
        if rank == 0:
            if args.make_npz:
                create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
            print("Done.")
        dist.barrier()
        dist.destroy_process_group()
        return

    if args.ckpt is None:
        assert args.model == "SiT-XL/2", "Only SiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
        assert args.image_size == 256, "512x512 models are not yet available for auto-download." # remove this line when 512x512 models are available
        ckpt_path = f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
        raw_obj = find_model(ckpt_path)
        state_dict = unwrap_checkpoint(raw_obj)
        learn_sigma = args.image_size == 256 if args.learn_sigma is None else args.learn_sigma
    else:
        raw_obj = torch.load(args.ckpt, map_location="cpu")
        state_dict = unwrap_checkpoint(raw_obj)
        learn_sigma = (
            infer_learn_sigma_from_state_dict(state_dict, args.model)
            if args.learn_sigma is None
            else args.learn_sigma
        )

    # Load model:
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
        attn_func=getattr(args, 'attn_func', None),
    ).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    model.load_state_dict(state_dict, strict=True)
    model.eval()  # important!
    
    
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse
            )
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    batch_size = 1 if mode == "SDE" else n
    pbar = range(0, len(pending_for_rank), batch_size)
    pbar = tqdm(pbar) if rank == 0 else pbar
    for start in pbar:
        batch_indices = pending_for_rank[start:start + batch_size]
        if not batch_indices:
            continue
        # Sample inputs:
        if mode == "SDE":
            z, y = make_single_index_inputs(batch_indices[0], args, model, latent_size, device)
        else:
            z, y = make_batch_inputs(batch_indices, args, model, latent_size, device)
        current_local_batch = len(batch_indices)
        
        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.full((current_local_batch,), 1000, device=device, dtype=y.dtype)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            model_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            model_fn = model.forward

        samples = sample_fn(z, model_fn, **model_kwargs)[-1]
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for index, sample in zip(batch_indices, samples):
            save_sample_atomic(sample, canonical_sample_path(sample_folder_dir, index), rank)

    # Make sure all processes have finished saving their samples before final validation/npz.
    dist.barrier()
    validation_status = None
    if rank == 0:
        final_scan_result = scan_sample_folder(sample_folder_dir, total_samples, args.image_size)
        error_message = scan_error_message(final_scan_result, sample_folder_dir, total_samples)
        if error_message is None and final_scan_result["pending_indices"]:
            error_message = (
                f"Sample folder {sample_folder_dir} still has missing/invalid target PNGs after generation: "
                f"{format_index_preview(final_scan_result['pending_indices'])}."
            )
        validation_status = {"error_message": error_message}
    validation_status = broadcast_object(validation_status, src=0)
    if validation_status["error_message"] is not None:
        raise RuntimeError(validation_status["error_message"])

    if rank == 0:
        if args.make_npz:
            create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"

    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--sample-folder-name", type=str, default=None,
                        help="Optional explicit sample subdirectory name under --sample-dir.")
    parser.add_argument("--per-proc-batch-size", type=int, default=4)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")
    parser.add_argument(
        "--learn-sigma",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If omitted, infer learn_sigma from the checkpoint head shape when --ckpt is used."
    )
    parser.add_argument(
        "--make-npz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to build a .npz archive from the generated PNG samples."
    )

    parser.add_argument("--attn-func", type=str, default=None,
                        choices=["base", "fa2", "fa3", "torch_sdpa"],
                        help="Attention backend. Default: timm built-in.")

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE

    args = parser.parse_known_args()[0]
    main(mode, args)
