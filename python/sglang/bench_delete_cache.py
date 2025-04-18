import sglang as sgl

def main():
    llm = sgl.Engine(model_path="/workspace/Llama-3.1-8B-Instruct")
    prompt1 = ["The weather today is sunny. I will go to the park. "]
    prompt2 = ["The weather today is rainy. I will stay at home. "]
    prompt3 = ["The weather today is cloudy. I will go to the beach. "]
    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 0}

    output1 = llm.generate(prompt1, sampling_params)
    print(f"The cached token of output1 is {output1[0]['meta_info']['cached_tokens']}")
    _ = llm.generate(prompt1, sampling_params, delete_cache=True)
    output2 = llm.generate(prompt2, sampling_params)
    print(f"The cached token of output2 is {output2[0]['meta_info']['cached_tokens']}")
    _ = llm.generate(prompt2, sampling_params, delete_cache=False)
    output3 = llm.generate(prompt3, sampling_params)
    print(f"The cached token of output3 is {output3[0]['meta_info']['cached_tokens']}")

if __name__ == "__main__":
    main()