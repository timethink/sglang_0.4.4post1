"""
Usage:
python3 offline_batch_inference.py  --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import dataclasses
import time
import sglang as sgl
from sglang.srt.server_args import ServerArgs


def main(
    server_args: ServerArgs,
):
    # Sample prompts.
    batch_size = 1
    prompts = [
        f"{i} Prompt: Hello, my name is" for i in range(batch_size)
    ] + [
        f"{i} Prompt: The president of the United States is" for i in range(batch_size)
    ] + [
        f"{i} Prompt: The capital of France is" for i in range(batch_size)
    ] + [
        f"{i} Prompt: The future of AI is" for i in range(batch_size)
    ]
    # Create a sampling params object.
    #sampling_params = {"temperature": 0.8, "top_p": 0.95}
    sampling_params = [
        {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 10} for _ in range(batch_size)
    ] + [
        {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 20} for _ in range(batch_size)
    ] + [
        {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 30} for _ in range(batch_size)
    ] + [
        {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 40} for _ in range(batch_size)
    ]
    tmp_prompts = [
        f"{i} Write a short, neutral self-introduction for a fictional character. Hello, my name is" for i in range(batch_size)
    ] + [
        f"{i} Provide a concise factual statement about France’s capital city. The capital of France is" for i in range(batch_size)
    ] + [
        f"{i} Explain possible future trends in artificial intelligence. The future of AI is" for i in range(batch_size)
    ] + [
        f"{i} Summarize the current president of the United States. The president of the United States is" for i in range(batch_size)
    ]
    new_sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 100}
    start_time = time.time()
    # Create an LLM.
    llm = sgl.Engine(**dataclasses.asdict(server_args))
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    outputs2 = llm.generate(tmp_prompts, new_sampling_params)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    # Print the outputs.
    #for prompt, output in zip(prompts, outputs):
    #    print("===============================")
    #    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    main(server_args)
