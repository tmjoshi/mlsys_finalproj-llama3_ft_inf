# Notes on LLM Memory requirements

## 1) Model Weights: 

If model weights loaded in 16-bit precision (FP16/BF16),   
Formula: approx (Number of billion parameters * sizeof(fp16)) GB   
Eg: Llama 3.2 1B: 1 * 2 bytes = 2GB memory   
So model weights need **2GB** memory for the Llama 3.2 1B model  
 
## 2) K/V Cache: 

**Size of K/V Cache per token in bytes = 2 * (num_layers) * (num_heads * dim_head) *  precision_in_bytes**  

(num_heads * dim_head) = d_model, so plug that in  

Eg: in my project, for Llama 3.2 1B, we used 16 layers, d_model = 2048, sizeof(fp16) = 2 bytes  

Now, plugging everything in, this makes:  
Size of K/V Cache per token in bytes = 2 * 16 * 2048 * 2 bytes   
= 131,072 B   
= 131 KB   
= 0.13 MB   

So? 1 token tensor in the K/V Cache needs **0.13 MB**    


**Total size of KV cache in bytes = (batch_size) * (sequence_length) * 2 * (num_layers) * (d_model) *  precision_in_bytes**   
 
Eg: in my project, for Llama 3.2 1B, the method for K/V Cache storage is: K/V Cache pre-allocation   
precision_in_bytes for FP16 = 2 bytes   
from model args in the project code:    
max_batch_size = 4   
max_seq_len = 256    
d_model = 2048 (hidden_size is 'd_model')   
num_layers = 16   
All of these tensor dimensions were pre-allocated for the K/V Cache tensor  
 
Now, plugging all these values in,    
Total size of K/V Cache in bytes = 4 * 256 * 2 * 16 * 2048 * 2 bytes ~= 134,217,728 bytes = 134 MB = 0.13 GB   
So? the total size of the K/V Cache tensor = **0.13 GB** for the Llama 3.2 1B model   

 
{On a slightly deviating note:-  
some quick calculations:  
If we want max_seq_len number of tokens generated per time step (which is 256):   
Every time step results in: 256 * (1 token's memory) = 256 * 0.13 MB = 33.28 MB   
So? Every time step results in 33.28 MB being added to the existing K/V Cache tensor memory}   


Finally,   
Total Memory = Model Weights + Total size of KV Cache = 2 GB + 0.13 GB = 2.13 GB   
Llama 3.2 1B (with all assumptions according to my project) consumes: **2.13 GB**  




Sources used in calculations: 
1) USC EE508 Notes
2) Lei Gao, Kevin Yang, Chaoyi Jiang, USC EE508 Final Project: Efficient Processing of LLMs
3) Sebastian Raschka, Understanding and Coding the KV Cache in LLMs from Scratch: https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms
4) Nvidia Developer Technical Blog, Mastering LLM Techniques: Inference Optimization: https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
