import torch
import os
import time
import utils
import logging
import json
from llama.tokenizer import Tokenizer
from llama.model import ModelArgs, Llama
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, asdict
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast

IGNORE_INDEX = -100

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def _tokenize_fn_llama(texts, tok: Tokenizer, max_length: int):
    ids_list, lens = [], []
    for txt in texts:
        ids = tok.encode(txt, bos=True, eos=False)
        if len(ids) > max_length:
            logging.warning(f"Truncating sequence from {len(ids)} to {max_length}")
            ids = ids[:max_length]
        ids_list.append(torch.tensor(ids, dtype=torch.long))
        lens.append(len(ids))
    return ids_list, lens


def preprocess_llama(sources, targets, tok: Tokenizer, max_length: int):
    full_texts = [s + t for s, t in zip(sources, targets)]
    full_ids, _    = _tokenize_fn_llama(full_texts, tok, max_length)
    _,       src_lens = _tokenize_fn_llama(sources, tok, max_length)

    labels = []
    for seq, slen in zip(full_ids, src_lens):
        lbl = seq.clone()
        lbl[:slen] = IGNORE_INDEX   
        labels.append(lbl)

    return dict(input_ids=full_ids, labels=labels)


class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: Tokenizer, max_length: int):

        logging.warning("Loading data...")
        data = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        sources = []
        targets = []

        eos_token_str = tokenizer.decode([tokenizer.eos_id])
        for ex in data:
            prompt = (PROMPT_DICT["prompt_input"] 
                       if ex.get("input", "") 
                       else PROMPT_DICT["prompt_no_input"])
            sources.append(prompt.format_map(ex))
            targets.append(ex["output"] + eos_token_str)

        logging.warning("Tokenizing inputs... This may take some time...")
        d = preprocess_llama(sources, targets, tokenizer, max_length)
        self.input_ids = d["input_ids"]
        self.labels    = d["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx],
                    labels=self.labels[idx])


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: Tokenizer

    def __call__(self, instances):
        input_ids = [inst["input_ids"] for inst in instances]
        labels    = [inst["labels"]    for inst in instances]

        pad_val = self.tokenizer.eos_id 

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_val
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = input_ids.ne(pad_val)

        return dict(input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels)


def get_peak_memory_mb():
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def compute_shift_logits_labels(logits: torch.Tensor, labels: torch.Tensor):
    """
    Shift logits/labels so that at position t the model predicts token t+1,
    then flatten for CrossEntropyLoss.
    """

    
    shift_logits = logits[..., :-1, :].contiguous()  
    shift_labels = labels[..., 1:].contiguous()    

    
    B, Tm1, V = shift_logits.size()
    shift_logits = shift_logits.view(-1, V)         
    shift_labels = shift_labels.view(-1)           

    return shift_logits, shift_labels


def finetune(data_path, checkpoint_dir, output_dir, batch_size=2, num_epochs=3, lr=1e-5, accumulate_steps=8, grad_acc=False, mixed_p=False, use_lora=False):
    # load model and tokenizer
    tok = Tokenizer(os.path.join(checkpoint_dir, "tokenizer.model"))
    assert hasattr(tok, "pad_id") and hasattr(tok, "eos_id")

    ckpt = torch.load(os.path.join(checkpoint_dir, "consolidated.00.pth"), map_location="cpu")
    model_args = ModelArgs()
    model = Llama(model_args)
    model.load_state_dict(ckpt, strict=False)
    max_length = model_args.max_seq_len
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()

    # freeze pretrained weights if LoRA enabled
    if use_lora:
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            if not ("lora_A" in name or "lora_B" in name):
                param.requires_grad = False

    
    if grad_acc:
        accumulation_steps = accumulate_steps
        mini_batch_size = batch_size // accumulation_steps
    else:
        accumulation_steps = 1
        mini_batch_size = batch_size
    
    # data loader
    ds     = SupervisedDataset(data_path, tok, max_length=max_length)
    coll   = DataCollatorForSupervisedDataset(tok)
    loader = DataLoader(ds, batch_size=mini_batch_size, shuffle=True, collate_fn=coll)
    
    if use_lora:
        optim = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0, foreach=False)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0, foreach=False)
    
    loss_function = CrossEntropyLoss(ignore_index = IGNORE_INDEX)
    scaler = GradScaler() if mixed_p else None

    summed_duration = 0.0
    total_iters = 0
    optim.zero_grad()
    for epoch in range(num_epochs):
        for step, batch in enumerate(loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            torch.cuda.synchronize()
            start_time = time.time()

            if mixed_p:
                with autocast():
                    outputs = model(tokens=input_ids, start_pos=0)
                    logits = outputs                        

                    shift_logits, shift_labels = compute_shift_logits_labels(logits, labels)

                    loss = loss_function(shift_logits, shift_labels)

                scaler.scale(loss).backward()

                
            else: # no mixed precision
                outputs = model(tokens=input_ids, start_pos=0)
                logits = outputs

                shift_logits, shift_labels = compute_shift_logits_labels(logits, labels)
                loss = loss_function(shift_logits, shift_labels)

                loss.backward()
                
            if (step) % 20 == 0:
                print(f"Epoch {epoch:>2} | Step {step:>4}/{len(loader)} | loss = {loss.item():.4f}")

            do_update = (not grad_acc) or ((step+1) % accumulation_steps == 0)
            if do_update:
                if mixed_p:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad()

            torch.cuda.synchronize()
            end_time = time.time()
            summed_duration += (end_time - start_time)
            total_iters += 1

    print(f"Avg Training Time per step (seconds): {(summed_duration / total_iters):.3f}")
    print(f"Peak memory usage: {get_peak_memory_mb():.2f} MB")
    if use_lora:
        total_params = sum(param_tensor.numel() for param_tensor in model.parameters())
        trainable_params = sum(param_tensor.numel() for param_tensor in model.parameters() if param_tensor.requires_grad)
        print(f"Percentage of trainable parameters: {((trainable_params / total_params)*100):.2f}%")

    # save model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "finetuned_llama_state_dict.bin"))

    with open(os.path.join(output_dir, "model_args.json"), "w") as f:
        json.dump(asdict(model_args), f, indent=2)

if __name__ == "__main__":
    finetune(
        data_path      = "/home1/tmjoshi/ml-systems-final-project-BaloneyGit-main/ml-systems-final-project-BaloneyGit-main/alpaca_data_200.json",
        checkpoint_dir = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B"),
        output_dir     = "./finetuned_llama",
        grad_acc = False,
        mixed_p = True,
        use_lora = False
    )

