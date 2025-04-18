import sglang as sgl

def main():
    llm = sgl.Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")
    prompt = ["The weather today is sunny. "]
    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 0, "n":4}

    output = llm.generate(prompt, sampling_params)
    print(output)

if __name__ == "__main__":
    main()