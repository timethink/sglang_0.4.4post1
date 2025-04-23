"""
Usage:
python offline_batch_inference_async.py --model-path Qwen/Qwen2-VL-7B-Instruct

Note:
This demo shows the usage of async generation,
which is useful to implement an online-like generation with batched inference.
"""

import argparse
import asyncio
import dataclasses
import time

import sglang as sgl
from sglang.srt.server_args import ServerArgs


class InferenceEngine:
    def __init__(self, **kwargs):
        self.engine = sgl.Engine(**kwargs)

    async def generate(self, prompt, sampling_params):
        result = await self.engine.async_generate(prompt, sampling_params)
        return result


async def run_server(server_args):
    inference = InferenceEngine(**dataclasses.asdict(server_args))

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] 

    # Create a sampling params object.
    #sampling_params = {"temperature": 0.8, "top_p": 0.95}
    sampling_params = [
        {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 10}, 
        {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 20},
        {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 30},
        {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 40},
    ]
    tmp_prompts =[
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
    "Summarize the current president of the United States. The president of the United States is",
    ]
    new_sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 100}

    # Run the generation tasks concurrently in async mode.
    tasks = []
    for i, prompt in enumerate(prompts):
        task = asyncio.create_task(inference.generate(prompt, sampling_params[i]))
        tasks.append(task)
    
    new_tasks = []
    for i, task in enumerate(tasks):
        #start_time = time.time()
        result = await task
        if task.done():
            #print(f"Task {i} completed successfully.")
            new_task = asyncio.create_task(inference.generate(tmp_prompts[i], new_sampling_params))
            new_tasks.append(new_task)
        else:
            #print(f"Task {i} failed.")
            pass
        #end_time = time.time()
        #print(f"Prompt: {prompts[i]}\nGenerated text: {result['text']}")
        #print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    for task in new_tasks:
        result = await task
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    asyncio.run(run_server(server_args))
