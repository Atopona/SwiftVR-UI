"""Gradio UI for SwiftVR video restoration."""

import argparse
import os
import re
import shutil
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from typing import Iterable, Optional

try:
    import gradio as gr
except ModuleNotFoundError as exc:
    if exc.name != "gradio":
        raise
    raise SystemExit(
        "当前 Python 环境没有安装 Gradio。\n\n"
        "请使用项目虚拟环境安装并启动：\n"
        "  bash scripts/install_linux.sh\n"
        "  .venv/bin/python app.py --share true\n\n"
        "也可以先激活虚拟环境：\n"
        "  source .venv/bin/activate\n"
        "  python app.py --share true"
    ) from exc

from swiftvr import SwiftVRPipeline


DEFAULT_REPO_ID = "H-oliday/SwiftVR"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_OUTPUT_DIR = "outputs/gradio"
VIDEO_MODE = "视频文件"
IMAGE_SEQUENCE_MODE = "图片序列"
SCALE_SIZE_MODE = "按倍率放大"
RESOLUTION_SIZE_MODE = "指定分辨率"
SCALE_CHOICES = ["1X", "2X", "3X", "4X", "6X", "8X"]

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
        raise gr.Error("输出分辨率格式应为 1920x1080。")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise gr.Error("输出分辨率的宽和高必须大于 0。")
    return width, height


def _as_int(value, name: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise gr.Error(f"{name} 必须是整数。") from exc


def _parse_upscale_factor(value) -> int:
    text = str(value or "").strip().lower().replace("x", "")
    try:
        factor = int(text)
    except Exception as exc:
        raise gr.Error("放大倍率必须是 2X、4X 这样的整数倍率。") from exc
    if factor <= 0:
        raise gr.Error("放大倍率必须大于 0。")
    return factor


def _as_optional_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    fps = float(value)
    if fps <= 0:
        raise gr.Error("FPS 必须大于 0。")
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
    if input_mode == VIDEO_MODE:
        input_path = _upload_path(video)
        if input_path is None or not input_path.exists():
            raise gr.Error("请先上传视频文件。")
        suffix = input_path.suffix or ".mp4"
        local_path = work_dir / f"input{suffix}"
        shutil.copy2(input_path, local_path)
        return local_path

    if not frames:
        raise gr.Error("请先上传图片序列。")

    frame_dir = work_dir / "input_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in frames:
        path = _upload_path(item)
        if path is not None and path.exists():
            paths.append(path)
    if not paths:
        raise gr.Error("没有找到可读取的图片帧。")

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

    try:
        pipe = SwiftVRPipeline.from_pretrained(str(checkpoint_dir)).to(
            device,
            dtype=dtype,
            attention_backend=attention_backend,
            torch_compile=torch_compile,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "no kernel image is available for execution on the device" in message:
            raise gr.Error(
                "当前 PyTorch CUDA 版本不支持这张 GPU。"
                "RTX PRO 6000 / Blackwell sm_120 请安装 CUDA 12.8 nightly PyTorch：\n\n"
                ".venv/bin/python -m pip uninstall -y torch torchvision torchaudio\n"
                ".venv/bin/python -m pip install --pre torch torchvision "
                "--index-url https://download.pytorch.org/whl/nightly/cu128"
            ) from exc
        raise
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


def _browser_video_preview(source_path: Path, work_dir: Path) -> tuple[str, str]:
    """Create an H.264 preview MP4 because browsers often cannot play x265."""
    preview_path = work_dir / "restored_preview_h264.mp4"
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = "ffmpeg"

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-tag:v",
        "avc1",
        "-movflags",
        "+faststart",
        str(preview_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception as exc:
        return str(source_path), f"\n预览转码失败，已使用原始 MP4：{exc}"

    return str(preview_path), f"\n浏览器预览文件：{preview_path}"


def download_checkpoint(repo_id: str, checkpoint_dir: str):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise gr.Error("缺少 huggingface_hub，请先安装 UI 依赖。") from exc

    target = _expand_path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id or DEFAULT_REPO_ID,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    return f"模型已就绪：{target}"


def restore_video(
    input_mode,
    video,
    frames,
    checkpoint_dir,
    output_root,
    size_mode,
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

    yield "正在准备输入文件...", None, None, []

    checkpoint_path = _expand_path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
    if not _checkpoint_files_exist(checkpoint_path):
        raise gr.Error(f"未在 {checkpoint_path} 找到完整模型文件。")

    clip_len = _as_int(clip_len, "分块长度")
    if clip_len % 4 != 0:
        raise gr.Error("分块长度必须是 4 的倍数。")

    input_path = _prepare_input(input_mode, video, frames, work_dir)
    if size_mode == RESOLUTION_SIZE_MODE:
        if not str(resolution or "").strip():
            raise gr.Error("请选择“指定分辨率”时填写输出分辨率，例如 1920x1080。")
        parsed_resolution = _parse_resolution(resolution)
    else:
        parsed_resolution = None
    output_path = work_dir / ("png_frames" if png_save else "restored.mp4")

    yield "正在加载模型...", None, None, []

    with PIPELINE_LOCK:
        pipe = _load_pipeline(
            checkpoint_path,
            str(device or "cuda"),
            str(dtype or "bfloat16"),
            str(attention_backend or "auto"),
            bool(torch_compile),
        )

        yield "正在修复视频...", None, None, []

        stats = pipe.restore_video(
            str(input_path),
            str(output_path),
            resolution=parsed_resolution,
            upscale=_parse_upscale_factor(upscale),
            clip_len=clip_len,
            dit_overlap=_as_int(dit_overlap, "DiT 重叠"),
            fps=_as_optional_float(fps),
            quality=_as_int(quality, "输出质量"),
            png_save=bool(png_save),
            save_format=str(save_format or ""),
            ffmpeg_preset=str(ffmpeg_preset or ""),
            queue_size=_as_int(queue_size, "队列深度"),
            verbose=True,
        )

    result_path = Path(stats["output"])
    summary = (
        f"完成：共 {stats['frames']} 帧，用时 {stats['seconds']:.2f} 秒，"
        f"平均 {stats['fps']:.2f} FPS。\n输出位置：{result_path}"
    )

    if png_save:
        zip_path = _zip_directory(result_path, work_dir / "restored_png_sequence.zip")
        yield summary, None, str(zip_path), _preview_images(result_path)
    else:
        preview_path, preview_note = _browser_video_preview(result_path, work_dir)
        yield summary + preview_note, preview_path, str(result_path), []


UI_CSS = """
.swiftvr-shell {max-width: 1200px; margin: 0 auto;}
.swiftvr-title h1 {font-size: 28px; line-height: 1.2; margin-bottom: 4px;}
.swiftvr-title p {font-size: 14px; margin-top: 0; color: #5f6470;}
.swiftvr-status textarea {font-family: ui-monospace, SFMono-Regular, Consolas, monospace; line-height: 1.55;}
.swiftvr-run button {min-height: 46px; font-size: 16px; font-weight: 700;}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="SwiftVR 视频修复") as demo:
        with gr.Column(elem_classes=["swiftvr-shell"]):
            gr.Markdown(
                "# SwiftVR 视频修复\n"
                "上传低质量视频或图片序列，选择倍率或固定分辨率后开始修复。",
                elem_classes=["swiftvr-title"],
            )

            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("### 输入")
                    input_mode = gr.Radio(
                        [VIDEO_MODE, IMAGE_SEQUENCE_MODE],
                        label="素材类型",
                        value=VIDEO_MODE,
                    )
                    video = gr.File(
                        label="上传视频",
                        file_count="single",
                        file_types=[".mp4", ".mov", ".mkv", ".avi", ".webm"],
                    )
                    frames = gr.File(
                        label="上传图片序列",
                        file_count="multiple",
                        file_types=["image"],
                        visible=False,
                    )

                with gr.Column(scale=4):
                    gr.Markdown("### 输出")
                    size_mode = gr.Radio(
                        [SCALE_SIZE_MODE, RESOLUTION_SIZE_MODE],
                        label="输出尺寸",
                        value=SCALE_SIZE_MODE,
                    )
                    with gr.Row():
                        upscale = gr.Dropdown(
                            SCALE_CHOICES,
                            value="4X",
                            label="放大倍率",
                        )
                        resolution = gr.Textbox(
                            label="输出分辨率",
                            placeholder="1920x1080",
                            visible=False,
                        )
                    with gr.Row():
                        quality = gr.Slider(0, 100, value=85, step=1, label="输出质量")
                        fps = gr.Number(label="输出 FPS", value=None, precision=2)
                    png_save = gr.Checkbox(label="导出 PNG 序列", value=False)

                    with gr.Accordion("模型与保存路径", open=True):
                        checkpoint_dir = gr.Textbox(label="模型目录", value=DEFAULT_CHECKPOINT_DIR)
                        output_root = gr.Textbox(label="输出目录", value=DEFAULT_OUTPUT_DIR)
                        repo_id = gr.Textbox(label="Hugging Face 仓库", value=DEFAULT_REPO_ID)
                        download_btn = gr.Button("下载模型", variant="secondary")

            with gr.Accordion("高级推理设置", open=False):
                with gr.Row():
                    clip_len = gr.Number(label="分块长度", value=24, precision=0)
                    dit_overlap = gr.Number(label="DiT 重叠", value=0, precision=0)
                    queue_size = gr.Slider(1, 8, value=3, step=1, label="队列深度")
                with gr.Row():
                    device = gr.Textbox(label="运行设备", value="cuda")
                    dtype = gr.Dropdown(["bfloat16", "float16", "float32"], label="数据精度", value="bfloat16")
                    torch_compile = gr.Checkbox(label="torch.compile", value=False)
                with gr.Row():
                    attention_backend = gr.Dropdown(
                        ["auto", "sdpa", "flash_attn_2", "flash_attn_3", "sageattention", "xformers"],
                        label="注意力后端",
                        value="auto",
                    )
                    save_format = gr.Dropdown(["", "yuv444p"], label="保存格式", value="")
                    ffmpeg_preset = gr.Dropdown(
                        ["", "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
                        label="FFmpeg 预设",
                        value="",
                    )

            run_btn = gr.Button("开始修复", variant="primary", elem_classes=["swiftvr-run"])

            with gr.Row():
                status = gr.Textbox(label="运行状态", lines=5, elem_classes=["swiftvr-status"])

            with gr.Row():
                output_video = gr.Video(label="视频预览")
                output_file = gr.File(label="下载结果")

            output_gallery = gr.Gallery(label="PNG 预览", columns=4, height=360)

        input_mode.change(
            lambda mode: (
                gr.update(visible=mode == VIDEO_MODE),
                gr.update(visible=mode == IMAGE_SEQUENCE_MODE),
            ),
            inputs=input_mode,
            outputs=[video, frames],
        )

        size_mode.change(
            lambda mode: (
                gr.update(visible=mode == SCALE_SIZE_MODE),
                gr.update(visible=mode == RESOLUTION_SIZE_MODE),
            ),
            inputs=size_mode,
            outputs=[upscale, resolution],
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
                size_mode,
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
    parser = argparse.ArgumentParser(description="启动 SwiftVR Gradio UI。")
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
    launch_kwargs = {
        "server_name": cli_args.host,
        "server_port": cli_args.port,
        "share": cli_args.share,
        "inbrowser": cli_args.inbrowser,
        "theme": gr.themes.Soft(),
        "css": UI_CSS,
    }
    demo = build_demo().queue(max_size=8)
    try:
        demo.launch(**launch_kwargs)
    except TypeError as exc:
        if "theme" not in str(exc) and "css" not in str(exc):
            raise
        launch_kwargs.pop("theme", None)
        launch_kwargs.pop("css", None)
        demo.launch(**launch_kwargs)
