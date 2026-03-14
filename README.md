# Notes on LLM Memory requirements

## 1) Model Weights: 

If model weights loaded in 16-bit precision (FP16/BF16),   
Formula: approx (Number of billion parameters * sizeof(fp16)) GB   
Eg: Llama 3.2 1B: 1 * 2 bytes = 2 GB memory   
So model weights need **2GB** memory for the Llama 3.2 1B model  
 
## 2) K/V Cache: 

**Size of K/V Cache per token in bytes = 2 * (num_layers) * (num_heads * dim_head) *  precision_in_bytes**  

we know: sizeof(fp16) = 2 bytes 

In my project, for Llama 3.2 1B (from model args in model.py):   
16 decoder layers   
num_kv_heads = 8   
head_dim = d_model / n_heads = 2048 / 32 = 64   
For Llama 3.2 1B, as GQA, it is '(num_kv_heads * head_dim)', so I'm making sure to plug that in and not '(num_heads * head_dim)'  

Now, plugging everything in, this makes:   
Size of K/V Cache per token in bytes = 2 * 16 * (8 * 64) * 2 bytes  
= 32,768 B   
= 33 KB   
= 0.033 MB   

So? 1 token tensor in the K/V Cache needs **0.033 MB**      


**Total size of KV cache in bytes = (batch_size) * (sequence_length) * 2 * (num_layers) * (hidden_size) *  precision_in_bytes**     
 
Eg: in my project, for Llama 3.2 1B, the method for K/V Cache storage is: K/V Cache pre-allocation   
precision_in_bytes for FP16 = 2 bytes    
from model args in the project code:     
1) max_batch_size = 4    
2) max_seq_len = 256     
3) num_kv_heads * head_dim = 8 * 64 = 512  
4) num_layers = 16  
  
All of these tensor dimensions were pre-allocated for the K/V Cache tensor  
 
Now, plugging all these values in,      
Total size of K/V Cache in bytes = 4 * 256 * 2 * 16 * 512 * 2 bytes   
= 33,554,432 bytes   
= 33.5 MB  
= 0.0335 GB   
So? the total size of the K/V Cache tensor = **0.0335 GB** for the Llama 3.2 1B model     


Finally,    
Total Memory = Model Weights + Total size of KV Cache = 2 GB + 0.0335 GB = 2.0335 GB     
Llama 3.2 1B (with all assumptions according to my project) consumes: **2.0335 GB**  

Reasoning for GPU selection:
1) Whether it fits:  
   - P100 has 16 GB memory  
   - Looking at Total Memory (about 2 GB), the model fits on P100  
4) P100 is a relatively smaller GPU, maybe more available compared to larger GPUs, like A40 (48 GB) or A100 (80 GB), as USC Research Labs must be using the larger ones 




Sources I used in calculations: 
1) USC EE508 Notes
2) Lei Gao, Kevin Yang, Chaoyi Jiang, USC EE508 Final Project: Efficient Processing of LLMs
3) Sebastian Raschka, Understanding and Coding the KV Cache in LLMs from Scratch: https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms
4) Nvidia Developer Technical Blog, Mastering LLM Techniques: Inference Optimization: https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
