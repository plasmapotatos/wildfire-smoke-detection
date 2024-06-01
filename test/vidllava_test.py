import av
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# Load the model in half-precision
device = "cuda:0"
model = VideoLlavaForConditionalGeneration.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf", torch_dtype=torch.float16, device_map=device
)
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

# Load the video as an np.arrau, sampling uniformly 8 frames
video_path = hf_hub_download(
    repo_id="raushan-testing-hf/videos-test",
    filename="sample_demo_1.mp4",
    repo_type="dataset",
)
container = av.open("videos/test/no_smoke.mp4")
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
video = read_video_pyav(container, indices)

# For better results, we recommend to prompt the model in the following format
prompt = "USER: <video>Do you see smoke in the video? Be careful to distinguish between clouds and smoke ASSISTANT:"
print(video.shape)
inputs = processor(text=prompt, videos=video, return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=60)
result = processor.batch_decode(
    out, skip_special_tokens=True, clean_up_tokenization_spaces=True
)

print(result)
