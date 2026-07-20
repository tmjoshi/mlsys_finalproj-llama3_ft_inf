from llama.tokenizer import Tokenizer
from llama.model import ModelArgs, Llama
import torch
import os
import ipdb
import json

def inference():
    torch.manual_seed(1)

    checkppoint_dir = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B")
    tokenizer_path = os.path.join(checkppoint_dir, "tokenizer.model")
    tokenizer = Tokenizer(tokenizer_path)
    # model_path = os.path.join(checkppoint_dir, "consolidated.00.pth")

    model_args = ModelArgs(**json.load(open("./finetuned_llama/model_args.json")))
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Llama(model_args)
    model.load_state_dict(torch.load("./finetuned_llama/finetuned_llama_state_dict.bin", map_location="cpu"), strict=True)
    model.eval()

    device = "cuda"
    model.to(device)

    prompts = [
	        "I believe the meaning of life is",
	        "Simply put, the theory of relativity states that ",
	        """A brief message congratulating the team on the launch:
	
	        Hi everyone,
	        
	        I just """,
	        
	        """Translate English to French:
	        
	        sea otter => loutre de mer
	        peppermint => menthe poivrée
	        plush girafe => girafe peluche
	        cheese =>""",
    ]

    model.eval()
    results = model.generate(tokenizer, prompts, max_gen_len=64, temperature=0.6, top_p=0.9, kv_caching=False, device=device)

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")

    
if __name__ == "__main__":
    inference()

