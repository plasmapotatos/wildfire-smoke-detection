PALIGEMMA_DETECT_PROMPT = """detect smoke"""

PALIGEMMA_SEGMENT_PROMPT = """segment sky"""

LLAVA_PROMPT = """"You are a proficient smoke detector at a fire tower. Does the following image contain wildfire smoke? Look carefully, and distinguish between clouds and smoke. Reason out your logic. Then, output one line which is either "yes" or "no".
"""

GPT4_BASIC_PROMPT = """You are a proficient smoke detector at a fire tower. Does the following image contain wildfire smoke? Look carefully, and distinguish between clouds and smoke. Output "yes" only if there is smoke, and "no" only if there is no smoke. Do not output any other text."""

GPT4_REASONING_PROMPT = """You are a proficient smoke detector at a fire tower. Does the following image contain wildfire smoke? Look carefully, and distinguish between clouds, fog, and smoke. Smoke generately emanates from a clear localized point, while fog is more spread out. Clouds generally appear in the sky. Output "yes" only if there is smoke, and "no" only if there is no smoke. Look specifically around the horizon level for any signs of smoke. Reason out your logic, and enclose it in <Reasoning> tags. Then, output one line which is either "yes" or "no", enclosing it in <Output> tags enclosing your final answer, for example <Output>yes<Output> or <Output>no<Output>. Do NOT use <\Output> or <Output\> tags.
"""

PHI3_PROMPT = """Is there smoke in the image? """
PHI3_ASSISTANT = """You are given an image of a horizon scene. Your task is to determine if there is smoke in the image. Look for any smoke-like objects that seem to expand in size, as this could indicate the presence of smoke. Output "yes" if you see smoke, and "no" otherwise."""
