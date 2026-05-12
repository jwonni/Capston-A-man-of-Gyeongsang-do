"""
Qwen2.5-VL OCR 래퍼.

모델은 첫 호출 시 로드되며 이후 싱글톤으로 재사용된다.
"""
from __future__ import annotations

import torch
from PIL import Image

from src.config import Config

_model = None
_processor = None


def load_qwen():
    """Qwen2.5-VL 모델 및 프로세서 로드 (싱글톤).

    VRAM이 충분하면 GPU에 전부 올리고, 부족하면 CPU와 나눠 쓴다.
    """
    global _model, _processor
    if _model is None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if torch.cuda.is_available() else 0
        )
        # bfloat16 기준 7B 모델 ~14GB → VRAM 부족 시 4bit 양자화 또는 CPU offload
        if vram_gb >= 14:
            load_kwargs = dict(
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map="cuda",
            )
        elif vram_gb >= 6:
            # GPU VRAM에 맞게 레이어를 자동 분배 (나머지는 CPU RAM)
            load_kwargs = dict(
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map="auto",
            )
        else:
            load_kwargs = dict(
                torch_dtype=torch.float32,
                device_map="cpu",
            )

        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            Config.QWEN_MODEL_ID,
            **load_kwargs,
        )
        _processor = AutoProcessor.from_pretrained(Config.QWEN_MODEL_ID)
    return _model, _processor


def _ocr_single_crop(crop_img: Image.Image, model, processor) -> str:
    """단일 크롭 이미지에서 텍스트 추출."""
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": crop_img},
            {"type": "text",
             "text": "이 이미지의 정확한 텍스트를 그대로 출력해라. 텍스트가 없으면 빈 문자열만 출력해라. 다른 말 하지 마라."},
        ],
    }]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=Config.QWEN_MAX_NEW_TOKENS_CROP,
            repetition_penalty=1.5,
            no_repeat_ngram_size=4,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    return processor.batch_decode(
        [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def merge_nearby_texts(
    results: list[dict],
    y_thresh: int = 3,
    x_thresh: int = 3,
) -> list[dict]:
    """X좌표가 같고(x_thresh 이내) Y 차이가 y_thresh 이하인 항목을 병합.

    줄바꿈으로 분리된 텍스트를 하나로 합친다.
    """
    if not results:
        return []
    used, merged = [False] * len(results), []
    for i, a in enumerate(results):
        if used[i]:
            continue
        group = [a]
        used[i] = True
        for j, b in enumerate(results):
            if used[j] or i == j:
                continue
            if abs(a["x"] - b["x"]) <= x_thresh and abs(a["y"] - b["y"]) <= y_thresh:
                group.append(b)
                used[j] = True
        if len(group) == 1:
            merged.append(a)
        else:
            gs = sorted(group, key=lambda r: r["y"])
            merged.append({
                "text": " ".join(r["text"] for r in gs),
                "x":    int(sum(r["x"] for r in gs) / len(gs)),
                "y":    int(sum(r["y"] for r in gs) / len(gs)),
            })
    return merged


def run_ocr(
    image_path: str,
    word_regions: list[dict],
    model=None,
    processor=None,
) -> list[dict]:
    """단어 영역 목록에 대해 OCR 수행.

    Args:
        image_path: 원본 이미지 경로
        word_regions: Hi-SAM에서 반환된 [{cx, cy, bbox}, ...] 목록
        model: Qwen 모델 (None이면 자동 로드)
        processor: Qwen 프로세서 (None이면 자동 로드)

    Returns:
        [{text, x, y}, ...] — 병합 후 OCR 결과
    """
    if model is None or processor is None:
        model, processor = load_qwen()

    orig_img = Image.open(image_path).convert("RGB")
    ocr_results = []

    for region in word_regions:
        x1, y1, x2, y2 = region["bbox"]
        crop = orig_img.crop((x1, y1, x2, y2))
        if crop.width < 64 or crop.height < 16:
            crop = crop.resize(
                (crop.width * Config.HISAM_UPSCALE, crop.height * Config.HISAM_UPSCALE),
                Image.LANCZOS,
            )
        text = _ocr_single_crop(crop, model, processor)
        if Config.QWEN_DROP_EMPTY and not text:
            continue
        ocr_results.append({"text": text, "x": region["cx"], "y": region["cy"]})

    return merge_nearby_texts(ocr_results)
