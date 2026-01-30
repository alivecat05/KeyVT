import os
import sys
import torch 
# add workspace to sys.path
sys.path.insert(0, '/root/dws/3D_QA/VG-LLM-main')
sys.path.insert(0, '/root/dws/3D_QA/VG-LLM-main/src')
sys.path.insert(0, '/root/dws/3D_QA/TStar/cdViews/cdviews') 
sys.path.insert(0,'/root/dws/3D_QA/TStar/Video-3D-LLM-main/llava/eval')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import base64
from PIL import Image
import decord
import numpy as np
import copy
from tqdm import tqdm
from io import BytesIO
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from qwen_vl_utils import extract_vision_info
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from qwen_vl.data.utils import load_and_preprocess_images
from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT

video_root = "/root/dws/3D_QA/Spatial-MLLM-master/evaluate/annotation/VSIBench"
import clip



# clip_model, preprocess = clip.load("ViT-L/14", device="cuda")

def get_topk(question,topk,images,clip_model,preprocess):

    text_input = clip.tokenize([question]).to("cuda")
    image_inputs = torch.stack([preprocess(image).to("cuda") for image in images])
    with torch.no_grad():
        image_features = clip_model.encode_image(image_inputs)
        text_features = clip_model.encode_text(text_input)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarities = (image_features @ text_features.T).squeeze(1)  # shape: (num_images,)

        topk_indices = similarities.topk(topk).indices.tolist()
        
        topk_images = [images[i] for i in topk_indices]
        
        return topk_images
        






class VGLLM_Inference:
    def __init__(self, pretrained,device):
        # load the model
        config = AutoConfig.from_pretrained(pretrained)
        self.model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
            pretrained,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            attn_implementation=None,
            # load_in_4bit=True,
        ).eval()
        self.model.to(device)
        self.max_num_frames = 10
        # load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")
        
        # load the tokenizer
        self.processor = AutoProcessor.from_pretrained(pretrained, padding_side="left")

    def call_model(
        self,
        contexts,
        visuals,
        add_frame_index: bool=False,
        gen_kwargs: dict = {},
    ):
        res = []
        messages = []
        processed_visuals = []
        for i, context in enumerate(contexts):
    
            message = [{"role": "system", "content": "You are a helpful assistant."}]

            if len(visuals) > 0:
                visual = visuals[i] if i < len(visuals) else None
                if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                    visual = os.path.join(video_root, visual.replace("./", ""))
                    vr = decord.VideoReader(visual)
                    image_num = len(vr)
                    # sample max_num_frames frame indices from the video
                    if image_num < self.max_num_frames:
                        frame_indices = np.arange(image_num)
                    else:
                        frame_indices = np.linspace(0, image_num - 1, self.max_num_frames).astype(int)
                    # read the frames
                    frames = [vr[i].asnumpy() for i in frame_indices]
                    visual_content = []
                    for frame in frames:
                        image = Image.fromarray(frame).convert("RGB").resize((256, 256))
                        visual_content.append({"type": "image", "image": image})
                    message.append({"role": "user", "content": visual_content + [{"type": "text", "text": context}]})
    
                elif isinstance(visual, Image.Image):  # Single image
                    base64_image = visual.convert("RGB")
                    buffer = BytesIO()
                    base64_image.save(buffer, format="JPEG")
                    base64_bytes = base64.b64encode(buffer.getvalue())
                    base64_string = base64_bytes.decode("utf-8")
                    message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}, {"type": "text", "text": context}]})
                elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                    image_content = []
                    image_count = 0
                    for v in visual:
                        base64_image = v.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        if add_frame_index:
                            image_content.append({"type": "text", "text": "Frame-{}: ".format(image_count)})    
                        image_content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
                        image_count += 1
                    message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                else:
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})
            else:
                message.append({"role": "user", "content": [{"type": "text", "text": context}]})
    
            messages.append(message)
    
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # image_inputs, video_inputs = process_vision_info(messages)
    
        geometry_encoder_inputs = []
        image_inputs = []
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size
        for idx,message in enumerate(messages):
            print(f"Processing message {idx+1}/{len(messages)}")
            vision_info = extract_vision_info(message)
            cur_geometry_encoder_inputs = []
            for ele in vision_info:
                if "image" in ele:
                    image = ele["image"]
                    if isinstance(image, Image.Image):
                        pass
                    elif isinstance(image, str) and "base64," in image:
                        _, base64_data = image.split("base64,", 1)
                        data = base64.b64decode(base64_data)
                        # fix memory leak issue while using BytesIO
                        with BytesIO(data) as bio:
                            image = copy.deepcopy(Image.open(bio))
                    else:
                        raise NotImplementedError("Unsupported image type")
    
                else:
                    raise NotImplementedError("Unsupported vision info type")
    
                assert isinstance(image, Image.Image), f"Unsupported image type: {type(image)}"
                image = load_and_preprocess_images([image])[0]
                cur_geometry_encoder_inputs.append(copy.deepcopy(image))
                _, height, width = image.shape
                # merge_size = 2
                if (width // patch_size) % merge_size > 0:
                    width = width - (width // patch_size) % merge_size * patch_size
                if (height // patch_size) % merge_size > 0:
                    height = height - (height // patch_size) % merge_size * patch_size
                image = image[:, :height, :width]
                image_inputs.append(image)
    
            geometry_encoder_inputs.append(torch.stack(cur_geometry_encoder_inputs))
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            do_rescale=False
        )
        device = self.model.device
        if getattr(self.model.config, "use_geometry_encoder", False):
            print("Using geometry encoder inputs")
            inputs["geometry_encoder_inputs"] = [feat.to(device) for feat in geometry_encoder_inputs]
        inputs = inputs.to(device)
    
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 4096
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
    
        pad_token_id = self.tokenizer.pad_token_id
    
        cont = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
        )
    
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for i, ans in enumerate(answers):
            answers[i] = ans
    
        for ans, context in zip(answers, contexts):
            res.append(ans)
    
        return res
@dataclass
class ModelConfig:
    """Arguments related to model loading and generation parameters."""
    model_path: str
    model_type: str
    temperature: float = 0.1
    top_p: float = 0.001
    max_tokens: int = 1024
    max_num_frames: int = 16
    use_vllm: bool = False
    device: str = "cuda"

def get_model_and_processor(config: ModelConfig):
    if "spatial-mllm" in config.model_type:
        from src.models import Qwen2_5_VL_VGGTForConditionalGeneration, Qwen2_5_VLProcessor
        model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(
            config.model_path, 
            torch_dtype="auto", 
            device_map="auto",
            # attn_implementation="flash_attention_2",
        )
        processor = Qwen2_5_VLProcessor.from_pretrained(config.model_path)
    elif 'llavavideo' in config.model_type:
        from llava.model.builder import load_pretrained_model
        model_name= 'llava_qwen'
        pretrained = '/root/dws/3D_QA/TStar/cdViews/model/LLaVA-Video-7B-Qwen2'
        tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device,attn_implementation=None)
        model.eval()
        return model, processor,tokenizer
    elif "vgllm" in config.model_type:
        print("Loading VGLLM model...") 
        print(f'Model device: {config.device}')
        model = VGLLM_Inference(config.model_path, device=config.device)
        processor = model.processor
        # model = model.model
    else:
        from transformers import AutoModelForCausalLM, AutoProcessor
        model = AutoModelForCausalLM.from_pretrained(config.model_path)
        processor = AutoProcessor.from_pretrained(config.model_path)
    return model, processor