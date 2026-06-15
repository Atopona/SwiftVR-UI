"""Gradio UI for SwiftVR video restoration."""

import argparse
import os
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import gradio as gr

from swiftvr import SwiftVRPipeline


DEFAULT_REPO_ID = "H-oliday/SwiftVR"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_OUTPUT_DIR = "outputs/gradio"

PIPELINE_LOCK = threading.Lock()
PIPELINE_CACHE = {"key": None, "pipe": None}


def _bool_from_string(value, default=False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _env_bool(names: Iterable[str], default=False) -> bool:
    for name in names:
        if name in os.environ:
            return _bool_from_string(os.environ[name], default)
    return bool(default)


def _expand_path(value: str) -> Path:
    return Path(str(value).strip()).expanduser().resolve()


def _parse_resolution(value: str) -> Optional[tuple[int, int]]:
    value = str(value or "").strip().lower()
    if not value:
        return None
    parts = re.split(r"[x,\s]+", value)
    parts = [p for p in parts if p]
    if len(parts) != 2:
        raise gr.Error("Resolution must look like 1920x1080.")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise gr.Error("Resolution width and height must be positive.")
    return width, height


def _as_int(value, name: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise gr.Error(f"{name} must be an integer.") from exc


def _as_optional_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    fps = float(value)
    if fps <= 0:
        raise gr.Error("FPS must be greater than 0.")
    return fps


def _upload_path(upload) -> Optional[Path]:
    if upload is None:
        return None
    if isinstance(upload, str):
        return Path(upload)
    if isinstance(upload, dict):
        for key in ("path", "name"):
            if upload.get(key):
                return Path(upload[key])
    for attr in ("path", "name"):
        value = getattr(upload, attr, None)
        if value:
            return Path(value)
    return None


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "frame.png"


def _numeric_sort_key(path: Path):
    try:
        return int(path.stem)
    except ValueError:
        return path.name


def _prepare_input(input_mode: str, video, frames, work_dir: Path) -> Path:
    if input_mode == "Video":
        input_path = _upload_path(video)
        if input_path is None or not input_path.exists():
            raise gr.Error("Upload a video first.")
        suffix = input_path.suffix or ".mp4"
        local_path = work_dir / f"input{suffix}"
        shutil.copy2(input_path, local_path)
        return local_path

    if not frames:
        raise gr.Error("Upload image frames first.")

    frame_dir = work_dir / "input_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in frames:
        path = _upload_path(item)
        if path is not None and path.exists():
            paths.append(path)
    if not paths:
        raise gr.Error("No readable image frames were uploaded.")

    for idx, path in enumerate(sorted(paths, key=_numeric_sort_key)):
        suffix = path.suffix.lower() or ".png"
        target = frame_dir / f"{idx:05d}_{_safe_filename(path.stem)}{suffix}"
        shutil.copy2(path, target)
    return frame_dir


def _checkpoint_files_exist(checkpoint_dir: Path) -> bool:
    return (
        (checkpoint_dir / "reae.safetensors").exists()
        and (checkpoint_dir / "prompt_embedding.safetensors").exists()
        and (checkpoint_dir / "transformer").is_dir()
    )


def _load_pipeline(checkpoint_dir: Path, device: str, dtype: str, attention_backend: str, torch_compile: bool):
    key = (str(checkpoint_dir), device, dtype, attention_backend, bool(torch_compile))
    if PIPELINE_CACHE["key"] == key and PIPELINE_CACHE["pipe"] is not None:
        return PIPELINE_CACHE["pipe"]

    pipe = SwiftVRPipeline.from_pretrained(str(checkpoint_dir)).to(
        device,
        dtype=dtype,
        attention_backend=attention_backend,
        torch_compile=torch_compile,
    )
    PIPELINE_CACHE["key"] = key
    PIPELINE_CACHE["pipe"] = pipe
    return pipe


def _zip_directory(source_dir: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir))
    return zip_path


def _preview_images(source_dir: Path, limit: int = 60) -> list[str]:
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return [
        str(path)
        for path in sorted(source_dir.rglob("*"))
        if path.suffix.lower() in image_exts
    ][:limit]


def download_checkpoint(repo_id: str, checkpoint_dir: str):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise gr.Error("Install the UI extra first: pip install -e .[ui]") from exc

    target = _expand_path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id or DEFAULT_REPO_ID,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    return f"Checkpoint ready: {target}"


def restore_video(
    input_mode,
    video,
    frames,
    checkpoint_dir,
    output_root,
    resolution,
    upscale,
    clip_len,
    dit_overlap,
    fps,
    quality,
    png_save,
    save_format,
    ffmpeg_preset,
    queue_size,
    device,
    dtype,
    attention_backend,
    torch_compile,
):
    started = time.strftime("%Y%m%d-%H%M%S")
    work_dir = _expand_path(output_root or DEFAULT_OUTPUT_DIR) / started
    work_dir.mkdir(parents=True, exist_ok=True)

    yield "Preparing input...", None, None, []

    checkpoint_path = _expand_path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
    if not _checkpoint_files_exist(checkpoint_path):
        raise gr.Error(f"Checkpoint files were not found in {checkpoint_path}.")

    clip_len = _as_int(clip_len, "Clip length")
    if clip_len % 4 != 0:
        raise gr.Error("Clip length must be a multiple of 4.")

    input_path = _prepare_input(input_mode, video, frames, work_dir)
    parsed_resolution = _parse_resolution(resolution)
    output_path = work_dir / ("png_frames" if png_save else "restored.mp4")

    yield "Loading model...", None, None, []

    with PIPELINE_LOCK:
        pipe = _load_pipeline(
            checkpoint_path,
            str(device or "cuda"),
            str(dtype or "bfloat16"),
            str(attention_backend or "auto"),
            bool(torch_compile),
        )

        yield "Restoring video...", None, None, []

        stats = pipe.restore_video(
            str(input_path),
            str(output_path),
            resolution=parsed_resolution,
            upscale=_as_int(upscale, "Upscale"),
            clip_len=clip_len,
            dit_overlap=_as_int(dit_overlap, "DiT overlap"),
            fps=_as_optional_float(fps),
            quality=_as_int(quality, "Quality"),
            png_save=bool(png_save),
            save_format=str(save_format or ""),
            ffmpeg_preset=str(ffmpeg_preset or ""),
            queue_size=_as_int(queue_size, "Queue size"),
            verbose=True,
        )

    result_path = Path(stats["output"])
    summary = (
        f"Done: {stats['frames']} frames in {stats['seconds']:.2f}s "
        f"({stats['fps']:.2f} fps). Output: {result_path}"
    )

    if png_save:
        zip_path = _zip_directory(result_path, work_dir / "restored_png_sequence.zip")
        yield summary, None, str(zip_path), _preview_images(result_path)
    else:
        yield summary, str(result_path), str(result_path), []


def build_demo() -> gr.Blocks:
    css = """
    .swiftvr-shell {max-width: 1180px; margin: 0 auto;}
    .swiftvr-status textarea {font-family: ui-monospace, SFMono-Regular, Consolas, monospace;}
    """
    with gr.Blocks(title="SwiftVR", css=css, theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_classes=["swiftvr-shell"]):
            gr.Markdown("# SwiftVR")

            with gr.Row():
                with gr.Column(scale=1):
                    checkpoint_dir = gr.Textbox(label="Checkpoint directory", value=DEFAULT_CHECKPOINT_DIR)
                    output_root = gr.Textbox(label="Output directory", value=DEFAULT_OUTPUT_DIR)
                with gr.Column(scale=1):
                    repo_id = gr.Textbox(label="Hugging Face repo", value=DEFAULT_REPO_ID)
                    download_btn = gr.Button("Download checkpoint", variant="secondary")

            with gr.Row():
                with gr.Column(scale=1):
                    input_mode = gr.Radio(["Video", "Image sequence"], label="Input", value="Video")
                    video = gr.Video(label="Video file", sources=["upload"], type="filepath")
                    frames = gr.File(
                        label="Image frames",
                        file_count="multiple",
                        file_types=["image"],
                        visible=False,
                    )

                    with gr.Accordion("Output", open=True):
                        png_save = gr.Checkbox(label="PNG sequence", value=False)
                        resolution = gr.Textbox(label="Resolution", placeholder="1920x1080")
                        upscale = gr.Slider(1, 8, value=4, step=1, label="Upscale")
                        fps = gr.Number(label="FPS", value=None, precision=2)
                        quality = gr.Slider(0, 100, value=85, step=1, label="Quality")

                with gr.Column(scale=1):
                    with gr.Accordion("Inference", open=True):
                        clip_len = gr.Number(label="Clip length", value=24, precision=0)
                        dit_overlap = gr.Number(label="DiT overlap", value=0, precision=0)
                        queue_size = gr.Slider(1, 8, value=3, step=1, label="Queue size")
                        device = gr.Textbox(label="Device", value="cuda")
                        dtype = gr.Dropdown(["bfloat16", "float16", "float32"], label="Dtype", value="bfloat16")
                        attention_backend = gr.Dropdown(
                            ["auto", "sdpa", "flash_attn_2", "flash_attn_3", "sageattention", "xformers"],
                            label="Attention backend",
                            value="auto",
                        )
                        torch_compile = gr.Checkbox(label="torch.compile", value=False)
                        save_format = gr.Dropdown(["", "yuv444p"], label="Save format", value="")
                        ffmpeg_preset = gr.Dropdown(
                            ["", "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
                            label="FFmpeg preset",
                            value="",
                        )

                    run_btn = gr.Button("Restore", variant="primary")

            with gr.Row():
                status = gr.Textbox(label="Status", lines=4, elem_classes=["swiftvr-status"])

            with gr.Row():
                output_video = gr.Video(label="Restored video")
                output_file = gr.File(label="Download")

            output_gallery = gr.Gallery(label="PNG preview", columns=4, height=360)

        input_mode.change(
            lambda mode: (gr.update(visible=mode == "Video"), gr.update(visible=mode == "Image sequence")),
            inputs=input_mode,
            outputs=[video, frames],
        )

        download_btn.click(
            download_checkpoint,
            inputs=[repo_id, checkpoint_dir],
            outputs=status,
        )

        run_btn.click(
            restore_video,
            inputs=[
                input_mode,
                video,
                frames,
                checkpoint_dir,
                output_root,
                resolution,
                upscale,
                clip_len,
                dit_overlap,
                fps,
                quality,
                png_save,
                save_format,
                ffmpeg_preset,
                queue_size,
                device,
                dtype,
                attention_backend,
                torch_compile,
            ],
            outputs=[status, output_video, output_file, output_gallery],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the SwiftVR Gradio UI.")
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--share", nargs="?", const="true", default=None)
    parser.add_argument("--inbrowser", action="store_true")
    args, unknown = parser.parse_known_args()

    share_arg = args.share
    for item in unknown:
        if item.lower().startswith("share="):
            share_arg = item.split("=", 1)[1]

    args.share = _bool_from_string(
        share_arg,
        _env_bool(("SWIFTVR_SHARE", "GRADIO_SHARE", "Share", "SHARE"), False),
    )
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    build_demo().queue(max_size=8).launch(
        server_name=cli_args.host,
        server_port=cli_args.port,
        share=cli_args.share,
        inbrowser=cli_args.inbrowser,
    )
