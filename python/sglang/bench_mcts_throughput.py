"""
Benchmark the throughput in the offline mode.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (the same as bench_serving.py).

# Usage
## Sharegpt dataset with default args
python -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --num-prompts 10

## Random dataset with default args
python -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --dataset-name random --random-input 1024 --random-output 1024
"""

#添加shared_prefix
#python -m sglang.bench_offline_throughput --model-path /workspace/AlphaMath-7B --dataset-name generated-shared-prefix --gsp-num-groups 16 --gsp-prompts-per-group 4 --gsp-system-prompt-len 1024 --gsp-question-len 64 --gsp-output-len 128

import argparse
import dataclasses
import json
import logging
import os
import random
import time
from typing import Optional, Any, Dict, List, Callable, Type, Tuple
import torch
import numpy as np

from sglang.bench_serving import (
    get_dataset,
    get_tokenizer,
    #添加config加载
    get_config,
    sample_random_requests,
    set_ulimit,
)
from sglang.lang.backend.runtime_endpoint import Runtime
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs
#添加flush_cache
from sglang.api import flush_cache

@dataclasses.dataclass
class BenchArgs:
    backend: str = "engine"
    result_filename: str = ""
    dataset_name: str = "sharegpt"
    dataset_path: str = ""
    num_prompts: int = 1000
    sharegpt_output_len: Optional[int] = None
    sharegpt_context_len: Optional[int] = None
    random_input_len: int = 1024
    random_output_len: int = 1024
    random_range_ratio: float = 0.0
    gsp_num_groups: int = 64
    gsp_prompts_per_group: int = 16
    gsp_system_prompt_len: int = 2048
    gsp_question_len: int = 128
    gsp_output_len: int = 256
    seed: int = 1
    disable_ignore_eos: bool = False
    extra_request_body: Optional[str] = None
    apply_chat_template: bool = False
    profile: bool = False
    skip_warmup: bool = False
    do_not_exit: bool = False
    prompt_suffix: str = ""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--backend", type=str, default=BenchArgs.backend)
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            default="sharegpt",
            choices=["sharegpt", "random", "generated-shared-prefix"],
            help="Name of the dataset to benchmark on.",
        )
        parser.add_argument(
            "--dataset-path", type=str, default="", help="Path to the dataset."
        )
        parser.add_argument(
            "--num-prompts",
            type=int,
            default=BenchArgs.num_prompts,
            help="Number of prompts to process. Default is 1000.",
        )
        parser.add_argument(
            "--sharegpt-output-len",
            type=int,
            default=BenchArgs.sharegpt_output_len,
            help="Output length for each request. Overrides the output length from the ShareGPT dataset.",
        )
        parser.add_argument(
            "--sharegpt-context-len",
            type=int,
            default=BenchArgs.sharegpt_context_len,
            help="The context length of the model for the ShareGPT dataset. Requests longer than the context length will be dropped.",
        )
        parser.add_argument(
            "--random-input-len",
            type=int,
            default=BenchArgs.random_input_len,
            help="Number of input tokens per request, used only for random dataset.",
        )
        parser.add_argument(
            "--random-output-len",
            type=int,
            default=BenchArgs.random_output_len,
            help="Number of output tokens per request, used only for random dataset.",
        )
        parser.add_argument(
            "--random-range-ratio",
            type=float,
            default=BenchArgs.random_range_ratio,
            help="Range of sampled ratio of input/output length, "
            "used only for random dataset.",
        )
        parser.add_argument(
            "--gsp-num-groups",
            type=int,
            default=BenchArgs.gsp_num_groups,
            help="Number of groups with shared prefix, used"
            "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-prompts-per-group",
            type=int,
            default=BenchArgs.gsp_prompts_per_group,
            help="Number of prompts per group of shared prefix, used"
            "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-system-prompt-len",
            type=int,
            default=BenchArgs.gsp_system_prompt_len,
            help="System prompt length, used" "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-question-len",
            type=int,
            default=BenchArgs.gsp_question_len,
            help="Question length, used" "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-output-len",
            type=int,
            default=BenchArgs.gsp_output_len,
            help="Target length in tokens for outputs in generated-shared-prefix dataset",
        )
        parser.add_argument("--seed", type=int, default=1, help="The random seed.")
        parser.add_argument(
            "--disable-ignore-eos",
            action="store_true",
            help="Disable ignore EOS token",
        )
        parser.add_argument(
            "--extra-request-body",
            metavar='{"key1": "value1", "key2": "value2"}',
            type=str,
            default=BenchArgs.extra_request_body,
            help="Append given JSON object to the request payload. You can use this to specify"
            "additional generate params like sampling params.",
        )
        parser.add_argument(
            "--apply-chat-template",
            action="store_true",
            help="Apply chat template",
        )
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Use Torch Profiler. The endpoint must be launched with "
            "SGLANG_TORCH_PROFILER_DIR to enable profiler.",
        )
        parser.add_argument(
            "--skip-warmup",
            action="store_true",
            help="Skip the warmup batches.",
        )
        parser.add_argument(
            "--do-not-exit",
            action="store_true",
            help="Do not exit the program. This is useful for nsys profile with --duration and --delay.",
        )
        parser.add_argument(
            "--prompt-suffix",
            type=str,
            default="",
            help="Suffix applied to the end of all user prompts, followed by assistant prompt suffix.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

#定义MCTS树结构，用于存储每个节点的信息，每个节点包含了text信息和value信息
class Node:
    def __init__(self, tree_id, tag, prompt, parent=None):
        #tree_id表示树的id，每个树的id是唯一的
        self.tree_id = tree_id
        #tag表示节点的位置，根节点为0，第一个子节点是0.0，第二个子节点是0.1，第一个子节点的第一个子节点是0.0.0
        self.tag =  tag
        self.prompt = prompt
        self.parent = parent
        self.children = []
        self.N = 0  # 访问次数
        self.Q = 0  # 累计奖励

def print_tree(node, depth=0):
    print("  " * depth + node.prompt)
    for child in node.children:
        print_tree(child, depth + 1)

#将树打印到文件中
def print_tree_to_file(node, depth=0, filename="/workspace/Super_MARIO/bench_runtime/tree.json"):
    #with open(filename, "a") as f:
        #f.write("  " * depth + node.prompt + "\n")
        #for child in node.children:
        #    print_tree_to_file(child, depth + 1, filename)
    states = return_states(node, filename)
    #格式化json
    json_str = json.dumps(states, indent=4)
    with open(filename, "a") as f:
        f.write(json_str)

def return_states(root, file_name):
        output_image_path = file_name
        candidates = [root]
        states = {}
        #dot = Digraph(comment='MCTS Tree')
        #dot.attr(rankdir='500')
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = {
                "tree_id": node.tree_id,
                "tag": node.tag,
                "prompt": node.prompt,
                "q_value": node.Q,
                "visit_count": node.N,
                "is_terminal": not node.children,
                "token_ids_len": len(node.prompt),#这里之后可能要修改，实际上是prompt的长度
                "ucb": ucb(node) if node.parent else None,
            }
            if node.children:
                #for child in node.children:
                #    dot.edge(node.tag, child.tag)
                candidates.extend(node.children)
        #dot.render(output_image_path, format="png", cleanup=True)
        return states

def select_node(node):
    while node.children:
        node = select_best_child(node)
    return node

def select_best_child(node):
    return max(node.children, key=ucb)

def ucb(node):
    if node.N == 0:
        #这里不太确定该赋予什么值
        return float('inf')
    return node.Q / node.N + 2 * (2 * np.log(node.parent.N) / node.N) ** 0.5

def evaluate(prompt):
    # 评估函数，这里简单返回一个随机数,以Prompt的长度为随机数
    return random.random()

def backpropagate(node, reward):
    while node:
        node.N += 1
        node.Q += reward
        node = node.parent

def get_best_path(node):
    path = []
    while node:
        path.append(node.prompt)
        node = select_best_child(node)
    return path

def is_valid(output):
    return True

#添加，计算prefill
def calculate_prefill_flops( prefill_length: int, config) -> float:
    hidden_size = config.hidden_size
    num_hidden_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    flops_per_forward = 0
    # 估算每次前向传播的 FLOPS 数量
    flops_per_forward += (num_hidden_layers * (24 * hidden_size ** 2 + 4 * prefill_length * hidden_size) + 2 * hidden_size * vocab_size) * prefill_length

    return flops_per_forward
#添加，计算decode
def calculate_decode_flops( prefill_length: int, decode_length: int, config) -> float:
    hidden_size = config.hidden_size
    num_hidden_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    flops_per_forward = 0
    # 估算每次前向传播的 FLOPS 数量
    for i in range(decode_length):
        flops_per_forward += num_hidden_layers * (24 * hidden_size ** 2 + 4 * (prefill_length + i) * hidden_size) + 2 * hidden_size * vocab_size

    return flops_per_forward


def create_prompt(prompts, outputs, n):
    if len(prompts) * n!= len(outputs):
        raise ValueError("The number of prompts and outputs does not match.")
    new_prompts = []
    i = 0
    for prompt in prompts:
        new_prompt = prompt
        for j in range(n):
            output = outputs[i*n+j]
            new_prompt += output["text"]
        new_prompts.append(new_prompt)
        i += 1
    return new_prompts 


def throughput_test_once(
    backend_name: str,
    backend,
    reqs: List[Tuple[str, int, int]],
    ignore_eos: bool,
    extra_request_body: Dict,
    profile: bool,
    test_config: Optional[Dict[str, Any]] = None,
):
    #如果存在/workspace/Super_MARIO/bench_runtime文件夹，则删除文件夹再创建
    num_reqs = len(reqs)
    folder = f"/workspace/Super_MARIO/bench_runtime/parallel_origin_prompts{num_reqs}_in{test_config['random_input_len']}_out{test_config['random_output_len']}_mem{test_config['mem_fraction_static']}_model{test_config['model_name']}"

    
    if os.path.exists(folder):
        os.system(f"rm -rf {folder}")
    os.makedirs(folder, exist_ok=True)


    measurement_results = {
        "backend": backend_name,
        "successful_requests": len(reqs),
        "total_latency": -1,
        "total_input_tokens": sum(r[1] for r in reqs),
        "total_output_tokens": -1,
        "request_throughput": -1,
        "input_throughput": -1,
        "output_throughput": -1,
        "total_throughput": -1,
        #添加,total_prompt_tokens和total_cached_tokens,cache_hit_rate
        "total_prompt_tokens": -1,
        "total_cached_tokens": -1,
        "cache_hit_rate": -1,
        "MFU": -1,
    }
    

    prompt = [r[0] for r in reqs]
    #print(f"prompt: {prompt}\n")
    sampling_params = [
        {
            "temperature": 0.6,
            #添加，尝试注释掉max_new_tokens
            "max_new_tokens": r[2],
            "ignore_eos": ignore_eos,
            **extra_request_body,
        }
        for r in reqs
    ] 

    if profile:
        assert (
            "SGLANG_TORCH_PROFILER_DIR" in os.environ
        ), "Please set SGLANG_TORCH_PROFILER_DIR."
        os.makedirs(os.environ["SGLANG_TORCH_PROFILER_DIR"], exist_ok=True)
        backend.start_profile()

    #添加,可能要修改iteraions的值
    iterations = 40
    n = 4
    roots = []
    for i, single_prompt in enumerate(prompt):
        root = Node(tree_id=i, tag="0", prompt=single_prompt)
        roots.append(root)


    measurement_results_total = []
    for step in range(iterations):
        current_nodes = []
        current_prompts = []
        current_values = []
        for root in roots:
            node = select_node(root)  # 选择阶段
            current_nodes.append(node)
            #复制n份
            current_prompts.extend([node.prompt] * n)
            #复制n份UCB值
            ucb_value = ucb(node)
            current_values.extend([ucb_value] * n)
        """
        #将current_prompts写入文件
        filename = f"{folder}/prompts_{step}.json"
        with open(filename, "w") as f:
            f.write(str(current_prompts))
            f.write("\n")
        #将current_values写入文件
        filename = f"{folder}/values_{step}.json"
        with open(filename, "w") as f:
            f.write(str(current_values))
            f.write("\n")
        """
        st = time.perf_counter()
        #这里对比实验时，考虑要不要注释掉current_values
        outputs = backend.generate(current_prompts, sampling_params=sampling_params*n)#value=current_values)
        latency = time.perf_counter() - st
        if profile:
            backend.stop_profile()
            monitor_trace_file(os.getenv("SGLANG_TORCH_PROFILER_DIR"))

        gen_out = outputs

        #添加有一个delete_cache
        #tmp_outputs = backend.generate(current_prompts, sampling_params=sampling_params, value=current_values, delete_cache=True)
        #将gen_out写入文件
        """
        filename = f"{folder}/outputs_{step}.json"
        with open(filename, "w") as f:
            f.write(str(gen_out))
            f.write("\n")
        """
        if backend_name == "runtime":
            gen_out = json.loads(gen_out)

        server_info = backend.get_server_info()

        measurement_results["total_latency"] = latency
        
        measurement_results["total_output_tokens"] = sum(
            o["meta_info"]["completion_tokens"] for o in gen_out
        )
        #添加,total_prompt_tokens和total_cached_tokens,cache_hit_rate
        measurement_results["total_prompt_tokens"] = sum(
            o["meta_info"]["prompt_tokens"] for o in gen_out
        )
        #添加total_input_token,是total_prompt_tokens的1/n
        measurement_results["total_input_tokens"] = measurement_results["total_prompt_tokens"] / n

        measurement_results["total_cached_tokens"] = sum(
            o["meta_info"]["cached_tokens"] for o in gen_out
        )
        """
        #添加第一次发送的东西
        measurement_results["first_cached_tokens"] = sum(
            o["meta_info"]["first_cached_tokens"] for o in gen_out
        )
        measurement_results["first_cache_hit_rate"] = measurement_results["first_cached_tokens"] / measurement_results["total_prompt_tokens"]
        measurement_results["first_average_ttft"] = sum(
            o["meta_info"]["first_ttft"] for o in gen_out
        ) / len(gen_out)      
        """
        measurement_results["cache_hit_rate"] = measurement_results["total_cached_tokens"] / measurement_results["total_prompt_tokens"]

        #add average ttft
        #计算平均ttft
        measurement_results["average_ttft"] = sum(
            o["meta_info"]["ttft"] for o in gen_out
        ) / len(gen_out)



        measurement_results["request_throughput"] = (
            measurement_results["successful_requests"] / latency
        )
        measurement_results["input_throughput"] = (
            measurement_results["total_input_tokens"] / latency
        )
        measurement_results["output_throughput"] = (
            measurement_results["total_output_tokens"] / latency
        )
        measurement_results["total_throughput"] = (
            measurement_results["total_input_tokens"]
            + measurement_results["total_output_tokens"]
        ) / latency
        measurement_results["last_gen_throughput"] = server_info["last_gen_throughput"]
    
        #添加，计算MFU
        config = get_config(server_info["model_path"])
        total_flops = 0
        for output in gen_out:
            prefill_length = output["meta_info"]["prompt_tokens"]
            decode_length = output["meta_info"]["completion_tokens"]
            total_flops += calculate_prefill_flops(prefill_length, config)
            total_flops += calculate_decode_flops(prefill_length, decode_length, config)
        #理论flops
        if config.torch_dtype == torch.bfloat16:
            theoretical_flops_per_second = 312 * 10 ** 12
        else:
            print(config.torch_dtype)
            raise ValueError("Unsupported dtype, Please check the dtype of the model.")
        measurement_results["MFU"] = total_flops / (latency * theoretical_flops_per_second)



        #根据outputs更新每个node的children
        for i in range(len(current_nodes)):
            if len(current_nodes) * n != len(outputs):
                raise ValueError("The number of prompts and outputs does not match.")
            node = current_nodes[i]
            for j in range(n):
                output = outputs[i*n+j]
                if is_valid(output):
                    #暂时先让child的prompt为路径上的prompt+output
                    child = Node(tree_id=node.tree_id, tag=node.tag+"."+str(j), prompt=node.prompt+output["text"], parent=node)
                    node.children.append(child)
        #评估每个child的reward
        for node in current_nodes:
            for child in node.children:
                reward = evaluate(child.prompt)
                backpropagate(child, reward)
        
        #更新measurement_results_total
        current_measurement = measurement_results.copy()
        measurement_results_total.append(current_measurement)
        #把measurement_results写入文件
        filename = f"{folder}/measurement_results_{step}.json"
        with open(filename, "w") as f:
            f.write(str(measurement_results))
            f.write("\n")
        #把measurement_results_total写入
        filename = f"{folder}/measurement_results_total.json"
        with open(filename, "a") as f:
            f.write(str(measurement_results))
            f.write("\n")
        
        #打印树结构到文件
        tree_file_name = f"{folder}/tree_{step}.json"
        for root in roots:
            print_tree_to_file(root, 0, tree_file_name)
        #读取tree_cache.txt，创建tree{i}.txt文件，并将tree_cache.txt清空
        """
        tree_cache_file = "/workspace/Super_MARIO/bench_runtime/tree_cache.txt"
        with open(tree_cache_file, "r") as f:
            lines = f.readlines()
        #将lines中的内容全部写入tree{step}.txt文件中
        tree_file_name = f"{folder}/tree_cache_{step}.txt"
        with open(tree_file_name, "w") as f:
            for line in lines:
                f.write(line)
        #清空tree_cache.txt
        with open(tree_cache_file, "w") as f:
            f.write("")
        """
    
    #根据measurement_results_total，画出cache_hit_rate和MFU随step的变化图
    #提取latency,cache_hit_rate,MFU
    latencys = []
    cache_hit_rate = []
    MFU = []
    throughput = []
    TTFT = []
    first_cache_hit_rate = []
    first_average_ttft = []
    for measurement_result in measurement_results_total:
        latencys.append(measurement_result["total_latency"])
        cache_hit_rate.append(measurement_result["cache_hit_rate"])
        #first_cache_hit_rate.append(measurement_result["first_cache_hit_rate"])
        #first_average_ttft.append(measurement_result["first_average_ttft"])
        MFU.append(measurement_result["MFU"])
        throughput.append(measurement_result["total_throughput"])
        TTFT.append(measurement_result["average_ttft"])
    #将latencys写入文件
    filename = f"{folder}/latencys.json"
    with open(filename, "w") as f:
        f.write(str(latencys))
        f.write("\n")
    #将TTFT写入文件
    filename = f"{folder}/TTFT.json"
    with open(filename, "w") as f:
        f.write(str(TTFT))
        f.write("\n")
    #将cache_hit_rate和MFU写入文件
    filename = f"{folder}/cache_hit_rate.json"
    with open(filename, "w") as f:
        f.write(str(cache_hit_rate))
        f.write("\n")
    """    
    filename = f"{folder}/first_cache_hit_rate.json"
    with open(filename, "w") as f:
        f.write(str(first_cache_hit_rate))
        f.write("\n")
    filename = f"{folder}/first_average_ttft.json"
    with open(filename, "w") as f:
        f.write(str(first_average_ttft))
        f.write("\n")
    """
    filename = f"{folder}/MFU.json"
    with open(filename, "w") as f:
        f.write(str(MFU))
        f.write("\n")
    filename = f"{folder}/throughput.json"
    with open(filename, "w") as f:
        f.write(str(throughput))
        f.write("\n")
        
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(latencys)
    plt.xlabel("step")
    plt.ylabel("latency")
    plt.title("latency vs step")
    plt.savefig(f"{folder}/latency.png")
    plt.figure()
    plt.plot(cache_hit_rate)
    plt.xlabel("step")
    plt.ylabel("cache_hit_rate")
    plt.title("cache_hit_rate vs step")
    plt.savefig(f"{folder}/cache_hit_rate.png")
    """
    plt.figure()
    plt.plot(first_cache_hit_rate)
    plt.xlabel("step")
    plt.ylabel("first_cache_hit_rate")
    plt.title("first_cache_hit_rate vs step")
    plt.savefig(f"{folder}/first_cache_hit_rate.png")
    plt.figure()
    plt.plot(first_average_ttft)
    plt.xlabel("step")
    plt.ylabel("first_average_ttft")
    plt.title("first_average_ttft vs step")
    plt.savefig(f"{folder}/first_average_ttft.png")
    """
    plt.figure()
    
    plt.plot(MFU)
    plt.xlabel("step")
    plt.ylabel("MFU")
    plt.title("MFU vs step")
    plt.savefig(f"{folder}/MFU.png")
    plt.figure()
    plt.plot(throughput)
    plt.xlabel("step")
    plt.ylabel("throughput")
    plt.title("throughput vs step")
    plt.savefig(f"{folder}/throughput.png")
    plt.figure()
    plt.plot(TTFT)
    plt.xlabel("step")
    plt.ylabel("TTFT")
    plt.title("TTFT vs step")
    plt.savefig(f"{folder}/TTFT.png")

    return measurement_results_total



def monitor_trace_file(directory, interval=1):

    print(f"Monitoring {directory} for new trace files...")

    known_files = set(os.listdir(directory))

    while True:
        flag = False
        time.sleep(interval)
        current_files = set(os.listdir(directory))

        new_files = current_files - known_files
        for new_file in new_files:
            new_file_path = os.path.join(directory, new_file)
            print(f"New file detected: {new_file}")

            previous_size = 0
            while True:
                try:
                    current_size = os.path.getsize(new_file_path)
                except FileNotFoundError:
                    print(f"File {new_file} is no longer accessible.")
                    break

                if current_size > previous_size:
                    previous_size = current_size
                else:
                    flag = True
                    break

                time.sleep(interval)
        if flag:
            break


def throughput_test(
    server_args: ServerArgs,
    bench_args: BenchArgs,
):
    if bench_args.backend == "engine":
        backend = Engine(**dataclasses.asdict(server_args))
        if not backend:
            raise ValueError("Please provide valid engine arguments")
    elif bench_args.backend == "runtime":
        backend = Runtime(**dataclasses.asdict(server_args))
    else:
        raise ValueError('Please set backend to either "engine" or "runtime"')

    tokenizer_id = server_args.tokenizer_path or server_args.model_path
    tokenizer = get_tokenizer(tokenizer_id)

    #添加，加载config
    config = get_config(server_args.model_path)

    # Set global environmnets
    set_ulimit()
    random.seed(bench_args.seed)
    np.random.seed(bench_args.seed)

    # Parse args
    extra_request_body = {}
    if bench_args.extra_request_body:
        extra_request_body = json.loads(args.extra_request_body)

    # Read dataset
    input_requests = get_dataset(bench_args, tokenizer)
    #将inpur_requests保存到/workspace/Super_MARIO/bench_runtime/input_request.json文件中
    filename = "/workspace/Super_MARIO/input_request.json"
    with open(filename, "w") as f:
        f.write(str(input_requests))
        f.write("\n")
    warmup_requests = sample_random_requests(
        input_len=256,
        output_len=16,
        num_prompts=min(bench_args.num_prompts, 16),
        range_ratio=1.0,
        tokenizer=tokenizer,
        dataset_path=bench_args.dataset_path,
    )
    #添加
    """
    # Warm up
    #warm up是为了让模型预热，避免在benchmark的时候出现性能波动
    if not bench_args.skip_warmup:
        logging.info("\nWarmup...")
        throughput_test_once(
            backend_name=bench_args.backend,
            backend=backend,
            reqs=warmup_requests,
            ignore_eos=not bench_args.disable_ignore_eos,
            extra_request_body=extra_request_body,
            profile=False,
        )
        time.sleep(0.5)
    """
    logging.info("\nBenchmark...")
    #backend.release_memory_occupation()
    #flush_cache()
    #提取random_input_len和random_output_len还有mem_fraction_static,model_name
    test_config = {
        "random_input_len": bench_args.random_input_len,
        "random_output_len": bench_args.random_output_len,
        "mem_fraction_static": server_args.mem_fraction_static,
        "model_name": server_args.model_path,
    }
    result = throughput_test_once(
        backend_name=bench_args.backend,
        backend=backend,
        reqs=input_requests,
        ignore_eos=not bench_args.disable_ignore_eos,
        extra_request_body=extra_request_body,
        profile=bench_args.profile,
        test_config=test_config,
    )
    backend.shutdown()

    if bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            fout.write(json.dumps(result) + "\n")
    #将result保存到/workspace/Super_MARIO/bench_runtime/result.json文件中
    filename = "/workspace/Super_MARIO/bench_runtime/result.json"
    with open(filename, "w") as f:
        f.write(str(result))
        f.write("\n")


def print_result(result):
    print(
        "\n{s:{c}^{n}}".format(s=" Offline Throughput Benchmark Result ", n=50, c="=")
    )
    print("{:<40} {:<10}".format("Backend:", result["backend"]))
    print("{:<40} {:<10}".format("Successful requests:", result["successful_requests"]))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", result["total_latency"]))
    print("{:<40} {:<10}".format("Total input tokens:", result["total_input_tokens"]))
    print(
        "{:<40} {:<10}".format("Total generated tokens:", result["total_output_tokens"])
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Last generation throughput (tok/s):", result["last_gen_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", result["request_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", result["input_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", result["output_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total token throughput (tok/s):", result["total_throughput"]
        )
    )
    #添加,total_prompt_tokens和total_cached_tokens,cache_hit_rate
    print(
        "{:<40} {:<10}".format(
            "Total prompt tokens:", result["total_prompt_tokens"]
        )
    )
    print(
        "{:<40} {:<10}".format(
            "Total cached tokens:", result["total_cached_tokens"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Cache hit rate:", result["cache_hit_rate"]
        )
    )
    print("=" * 50)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )
    #添加，清空/workspace/Super_MARIO/bench_runtime/tree.json文件
    """
    filename = "/workspace/Super_MARIO/bench_runtime/tree.json"
    with open(filename, "w") as f:
        f.write("")
    """
    throughput_test(server_args, bench_args)

    while bench_args.do_not_exit:
        pass