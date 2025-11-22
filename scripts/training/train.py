from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
import os
import json
import torch
from datasets import Dataset, Features, Value, Image, load_dataset, load_from_disk
from PIL import Image as IM, ExifTags, UnidentifiedImageError, ImageOps
import base64
import logging
import math
import sys
import warnings
from functools import lru_cache
from io import BytesIO
import requests
from transformers import AutoModelForVision2Seq, AutoProcessor
import torchvision
from packaging import version
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from typing import Optional
from trl import SFTConfig, SFTTrainer
from transformers import Qwen2VLProcessor

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model_path', help='Path to pretrained model', required=True)
    parser.add_argument('--data_path', help='Input data path', required=True)
    parser.add_argument('--output_dir', help='Output directory', required=True)
    parser.add_argument('--deepspeed_config', help='Path to DeepSpeed configuration JSON file', required=True)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    args = parser.parse_args()
    args_dct = vars(args)

    data_path = os.path.join(args_dct['data_path'], "balanced_chart_to_text_llm_judge_fine_tuning_dataset.csv")
    image_path = args_dct['data_path']


    class ChartEvalDataset:
        def __init__(self, data_path, image_path):
            self.data_path = data_path
            self.image_path = image_path
            self.df = self._load_csv()

        def _load_csv(self):
            df = pd.read_csv(self.data_path)
            return df

        def _create_dataset(self):
            def _generate_examples(df, image_path):
                print(df)
                for i, row in df.iterrows():
                    print(row)
                    im_path = os.path.join(image_path, row["image_path"])
                    try:
                        img = IM.open(im_path)
                        if hasattr(img, "_getexif"):
                            exif = img._getexif()
                            if exif is not None:
                                orientation = exif.get(ExifTags.TAGS.get("Orientation"))
                                if orientation in [3, 6, 8]:
                                    img = ImageOps.exif_transpose(img)
                    except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                        print(f"Error loading image: {im_path}. Skipping. Error: {e}")
                        continue
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
                        continue

                    example = {
                        "image": img,
                        "prompt": row["prompt"],
                        "output": row["output"],
                    }
                    yield example

            features = Features({
                "image": Image(),
                "prompt": Value("string"),
                "output": Value("string"),
            })

            dataset = Dataset.from_generator(
                _generate_examples,
                gen_kwargs={"df": self.df, "image_path": self.image_path},
                features=features
            )
            return dataset


    dataset_creator = ChartEvalDataset(data_path, image_path)
    dataset = dataset_creator._create_dataset()
    print("Dataset loaded successfully!")

    splits = dataset.train_test_split(test_size=0.05, seed=42)
    print("Dataset splits loaded successfully!")


    def format_data(sample):
        return {"messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample["prompt"]},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["output"]}],
            },
        ]}


    print(splits)
    train_data = [format_data(sample) for sample in splits["train"]]
    valid_data = [format_data(sample) for sample in splits["test"]]
    print("Train and validation sampled successfully!")

    model_id = args_dct["model_path"]

    # Load model and processor (DeepSpeed will handle distribution)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",  # not supported for training if changed
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model and processor loaded successfully!")

    IMAGE_FACTOR = 28
    MIN_PIXELS = 4 * 28 * 28
    MAX_PIXELS = 16384 * 28 * 28
    MAX_RATIO = 200

    VIDEO_MIN_PIXELS = 128 * 28 * 28
    VIDEO_MAX_PIXELS = 768 * 28 * 28
    FRAME_FACTOR = 2
    FPS = 2.0
    FPS_MIN_FRAMES = 4
    FPS_MAX_FRAMES = 768

    VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
    logger.info(f"set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")


    def round_by_factor(number: int, factor: int) -> int:
        return round(number / factor) * factor


    def ceil_by_factor(number: int, factor: int) -> int:
        return math.ceil(number / factor) * factor


    def floor_by_factor(number: int, factor: int) -> int:
        return math.floor(number / factor) * factor


    def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS,
                     max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
        if max(height, width) / min(height, width) > MAX_RATIO:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}")
        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = floor_by_factor(height / beta, factor)
            w_bar = floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)
        return h_bar, w_bar


    def to_rgb(pil_image: IM.Image) -> IM.Image:
        if pil_image.mode == 'RGBA':
            white_background = IM.new("RGB", pil_image.size, (255, 255, 255))
            white_background.paste(pil_image, mask=pil_image.split()[3])
            return white_background
        else:
            return pil_image.convert("RGB")


    def fetch_image(ele: dict[str, str | IM.Image], size_factor: int = IMAGE_FACTOR) -> IM.Image:
        if "image" in ele:
            image = ele["image"]
        else:
            image = ele["image_url"]
        image_obj = None
        if isinstance(image, IM.Image):
            image_obj = image
        elif image.startswith("http://") or image.startswith("https://"):
            response = requests.get(image, stream=True)
            image_obj = IM.open(BytesIO(response.content))
        elif image.startswith("file://"):
            image_obj = IM.open(image[7:])
        elif image.startswith("data:image"):
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                image_obj = IM.open(BytesIO(data))
        else:
            image_obj = IM.open(image)
        if image_obj is None:
            raise ValueError(
                f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
        image = to_rgb(image_obj)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=size_factor,
            )
        else:
            width, height = image.size
            min_pixels = ele.get("min_pixels", MIN_PIXELS)
            max_pixels = ele.get("max_pixels", MAX_PIXELS)
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=size_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        image = image.resize((resized_width, resized_height))
        return image


    def smart_nframes(ele: dict, total_frames: int, video_fps: int | float) -> int:
        assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
        if "nframes" in ele:
            nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
        else:
            fps = ele.get("fps", FPS)
            min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
            max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
            nframes = total_frames / video_fps * fps
            if nframes > total_frames:
                logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
            nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
            nframes = floor_by_factor(nframes, FRAME_FACTOR)
        if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
            raise ValueError(f"nframes should be in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
        return nframes


    def _read_video_torchvision(ele: dict) -> (torch.Tensor, float):
        video_path = ele["video"]
        if version.parse(torchvision.__version__) < version.parse("0.19.0"):
            if "http://" in video_path or "https://" in video_path:
                warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
            if "file://" in video_path:
                video_path = video_path[7:]
        st = time.time()
        video, audio, info = io.read_video(
            video_path,
            start_pts=ele.get("video_start", 0.0),
            end_pts=ele.get("video_end", None),
            pts_unit="sec",
            output_format="TCHW",
        )
        total_frames, video_fps = video.size(0), info["video_fps"]
        logger.info(f"torchvision: {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        idx = torch.linspace(0, total_frames - 1, nframes).round().long()
        sample_fps = nframes / max(total_frames, 1e-6) * video_fps
        video = video[idx]
        return video, sample_fps


    def is_decord_available() -> bool:
        import importlib.util
        return importlib.util.find_spec("decord") is not None


    def _read_video_decord(ele: dict) -> (torch.Tensor, float):
        import decord
        video_path = ele["video"]
        st = time.time()
        vr = decord.VideoReader(video_path)
        if 'video_start' in ele or 'video_end' in ele:
            raise NotImplementedError("Decord video reader does not support start_pts and end_pts for now.")
        total_frames, video_fps = len(vr), vr.get_avg_fps()
        logger.info(f"decord: {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
        video = vr.get_batch(idx).asnumpy()
        video = torch.tensor(video).permute(0, 3, 1, 2)
        sample_fps = nframes / max(total_frames, 1e-6) * video_fps
        return video, sample_fps


    VIDEO_READER_BACKENDS = {
        "decord": _read_video_decord,
        "torchvision": _read_video_torchvision,
    }

    FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


    @lru_cache(maxsize=1)
    def get_video_reader_backend() -> str:
        if FORCE_QWENVL_VIDEO_READER is not None:
            video_reader_backend = FORCE_QWENVL_VIDEO_READER
        elif is_decord_available():
            video_reader_backend = "decord"
        else:
            video_reader_backend = "torchvision"
        print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
        return video_reader_backend


    def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR,
                    return_video_sample_fps: bool = False) -> torch.Tensor | list[IM.Image]:
        if isinstance(ele["video"], str):
            video_reader_backend = get_video_reader_backend()
            try:
                video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
            except Exception as e:
                logger.warning(
                    f"video_reader_backend {video_reader_backend} error, using torchvision as fallback, msg: {e}")
                video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)
            nframes, _, height, width = video.shape
            min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
            total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
            max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
            max_pixels_supposed = ele.get("max_pixels", max_pixels)
            if max_pixels_supposed > max_pixels:
                logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
            max_pixels = min(max_pixels_supposed, max_pixels)
            if "resized_height" in ele and "resized_width" in ele:
                resized_height, resized_width = smart_resize(
                    ele["resized_height"],
                    ele["resized_width"],
                    factor=image_factor,
                )
            else:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=image_factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
            video = transforms.functional.resize(
                video,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
            if return_video_sample_fps:
                return video, sample_fps
            return video
        else:
            assert isinstance(ele["video"], (list, tuple))
            process_info = ele.copy()
            process_info.pop("type", None)
            process_info.pop("video", None)
            images = [
                fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
                for video_element in ele["video"]
            ]
            nframes = ceil_by_factor(len(images), FRAME_FACTOR)
            if len(images) < nframes:
                images.extend([images[-1]] * (nframes - len(images)))
            if return_video_sample_fps:
                return images, process_info.pop("fps", 2.0)
            return images


    def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
        vision_infos = []
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        for conversation in conversations:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if (
                                "image" in ele
                                or "image_url" in ele
                                or "video" in ele
                                or ele["type"] in ("image", "image_url", "video")
                        ):
                            vision_infos.append(ele)
        return vision_infos


    def process_vision_info(conversations: list[dict] | list[list[dict]], return_video_kwargs: bool = False) -> tuple[
        list[IM.Image] | None, list[torch.Tensor | list[IM.Image]] | None, Optional[dict]]:
        vision_infos = extract_vision_info(conversations)
        image_inputs = []
        video_inputs = []
        video_sample_fps_list = []
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(fetch_image(vision_info))
            elif "video" in vision_info:
                video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
                video_sample_fps_list.append(video_sample_fps)
                video_inputs.append(video_input)
            else:
                raise ValueError("image, image_url or video must be in content.")
        if len(image_inputs) == 0:
            image_inputs = None
        if len(video_inputs) == 0:
            video_inputs = None
        if return_video_kwargs:
            return image_inputs, video_inputs, {'fps': video_sample_fps_list}
        return image_inputs, video_inputs


    # Update training configuration to include DeepSpeed configuration.
    args_train = SFTConfig(
        output_dir=args_dct["output_dir"],
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=2e-5,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        eval_strategy="steps",
        eval_steps=50,
        deepspeed=args_dct["deepspeed_config"]  # pass DeepSpeed configuration
    )

    args_train.remove_unused_columns = False


    def collate_fn(examples):
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        image_inputs = [process_vision_info(example["messages"])[0] for example in examples]
        batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        if isinstance(processor, Qwen2VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch


    peft_config = None
    trainer = SFTTrainer(
        model=model,
        args=args_train,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=collate_fn,
        peft_config=peft_config,
        # tokenizer=processor.tokenizer,
    )

    print("Training starting!")

    checkpoint_dir = args_train.output_dir
    if os.path.isdir(checkpoint_dir) and any(
            os.path.isdir(os.path.join(checkpoint_dir, d)) for d in os.listdir(checkpoint_dir) if
            d.startswith("checkpoint-")):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if
                       os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith("checkpoint-")]
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        print("No checkpoint found, starting training from scratch.")
        trainer.train()

    print("Training complete!")
    trainer.save_model(os.path.join(args_dct['output_dir'], 'trained_model'))
