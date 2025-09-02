"""
Usage:
python3 offline_batch_inference.py  --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import dataclasses
import multiprocessing as mp

import sglang as sgl
from sglang.srt.server_args import ServerArgs


def start_background(server_args, prompts, sampling_params):
    llm = sgl.Engine(**dataclasses.asdict(server_args))
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def main(
    server_args: ServerArgs,
):
    # Sample prompts.
    prompts = [
        "Hello, my name is" * 1,
        "The president of the United States is" * 1,
        "The capital of France is" * 1,
        "The future of AI is" * 1,
    ] * 1
    # Create a sampling params object.
    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 20}

    # Create an LLM.
    ctx = mp.get_context("spawn")
    bg_args = dataclasses.replace(server_args, is_forever_bg_loop=True, central_ipc_port=32005)
    bg = ctx.Process(target=start_background, args=(bg_args, prompts, sampling_params))
    bg.start()

    outputs = start_background(server_args, prompts, sampling_params)

    # Print the outputs.
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
    
    bg.terminate()


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    main(server_args)
