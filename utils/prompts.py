PALIGEMMA_DETECT_PROMPT = """detect smoke"""

PALIGEMMA_SEGMENT_PROMPT = """segment sky"""

LLAVA_PROMPT = """"You are a proficient smoke detector at a fire tower. Does the following image contain wildfire smoke? Look carefully, and distinguish between clouds and smoke. Reason out your logic. Then, output one line which is either "yes" or "no".
"""

GPT4_BASIC_PROMPT = """You are a proficient smoke detector at a fire tower. Does the following image contain wildfire smoke? Look carefully, and distinguish between clouds and smoke. Output "yes" only if there is smoke, and "no" only if there is no smoke. Do not output any other text."""

GPT4_REASONING_PROMPT = """You are a proficient smoke detector at a fire tower. Does the following image contain wildfire smoke? Look carefully, and distinguish between clouds and smoke. Output "yes" only if there is smoke, and "no" only if there is no smoke. Look specifically around the horizon level for any signs of smoke. Reason out your logic, and enclose it in <Reasoning> <Reasoning/>. Then, output one line which is either "yes" or "no", enclosing it in <Output> <Output/> enclosing your final answer, for example <Output>yes<Output/> or <Output>no<Output/>.
"""

PHI3_PROMPT = """Is there smoke in the image? How confident are you?"""
PHI3_ASSISTANT = """You are given an image of a horizon scene. Your task is to determine if there is smoke in the image. Look for any smoke-like objects that seem to expand in size, as this could indicate the presence of smoke. Output "yes" if you see smoke, and "no" otherwise. Additionally, output a floating point number between 0 and 1 to indicating the chance of smoke. A value closer to 1 indicates higher chance of smoke."""
