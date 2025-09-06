import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# -----------------------
# 模型与 Tokenizer
# -----------------------
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

max_length = 1024
train_on_inputs = False

# -----------------------
# Tokenize 函数
# -----------------------
def tokenize(prompt, add_eos_token=False):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < max_length
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result

# -----------------------
# Prompt 生成
# -----------------------
def generate_prompt(instruction, input=None, label=None):
    res = instruction if not input else f"{instruction}\n{input}"
    if label:
        res = f"{res}\n{label}"
    return res

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(
        data_point["instruction"],
        None,
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)

    if not train_on_inputs:
        user_prompt = generate_prompt(data_point["instruction"], None, None)
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = (
            [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        )

    return tokenized_full_prompt

# -----------------------
# 数据加载
# -----------------------
data_affirm = load_dataset("json", data_files="../data/safety_affirm.json")
train_data_affirm = data_affirm["train"].map(generate_and_tokenize_prompt)

data_reject = load_dataset("json", data_files="../data/unsafety_affirm_fixed.json")
train_data_reject = data_reject["train"].map(generate_and_tokenize_prompt)

# -----------------------
# LoRA 配置
# -----------------------
config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------
# DataLoader 与 collate_fn
# -----------------------
def collate_fn(batch):
    input_ids = pad_sequence(
        [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_mask = pad_sequence(
        [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = pad_sequence(
        [torch.tensor(x["labels"], dtype=torch.long) for x in batch],
        batch_first=True,
        padding_value=-100,
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_loader_affirm = DataLoader(
    train_data_affirm, shuffle=True, batch_size=8, collate_fn=collate_fn
)
train_loader_reject = DataLoader(
    train_data_reject, shuffle=True, batch_size=8, collate_fn=collate_fn
)

# -----------------------
# Optimizer
# -----------------------
optimizer = AdamW(model.parameters(), lr=5e-5)

# -----------------------
# 训练循环
# -----------------------
num_epochs = 10
save_every = 80  # 每多少步保存
cnt = 0

for epoch in range(num_epochs):
    for batch_affirm, batch_reject in tqdm(
        zip(train_loader_affirm, train_loader_reject), total=len(train_loader_affirm)
    ):
        # Affirm loss
        input_ids = batch_affirm["input_ids"].to(device)
        attention_mask = batch_affirm["attention_mask"].to(device)
        labels = batch_affirm["labels"].to(device)
        loss_affirm = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss

        # Reject loss
        input_ids = batch_reject["input_ids"].to(device)
        attention_mask = batch_reject["attention_mask"].to(device)
        labels = batch_reject["labels"].to(device)
        loss_reject = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss

        # 总 loss
        loss = loss_affirm / (loss_reject + 1e-8)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if cnt % save_every == 0 and cnt != 0:
            output_dir = f"./resp_glad/{cnt}_lora"
            model.save_pretrained(output_dir)
            print(f"模型已保存到 {output_dir}")

        cnt += 1
output_dir = "./resp_glad/final_lora"
model.save_pretrained(output_dir)
print(f"训练完成，最终模型已保存到 {output_dir}")

print("训练完成 ✅")
