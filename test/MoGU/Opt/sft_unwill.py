import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# ---------------------------
# 模型和 tokenizer
# ---------------------------
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

max_length = 1024
train_on_inputs = False

# ---------------------------
# 数据处理
# ---------------------------
def tokenize(prompt, add_eos_token=False):
    result = tokenizer(
        prompt,
        truncation=True,
        padding=False,
        max_length=max_length,
        add_special_tokens=True,
        return_tensors=None
    )
    if add_eos_token and result["input_ids"][-1] != tokenizer.eos_token_id:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result

def generate_prompt(instruction, input=None, label=None):
    # OPT 简单指令模板
    prompt = instruction
    if input:
        prompt += f"\n{input}"
    if label:
        prompt += f"\n{label}"
    return prompt

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
        tokenized_full_prompt["labels"] = [-100]*user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
    return tokenized_full_prompt

# ---------------------------
# 加载训练数据
# ---------------------------
data_affirm = load_dataset("json", data_files="../data/safety_reject.json")
train_data_affirm = data_affirm['train'].map(generate_and_tokenize_prompt)

data_reject = load_dataset("json", data_files="../data/unsafety_reject_fixed.json")
train_data_reject = data_reject['train'].map(generate_and_tokenize_prompt)

# ---------------------------
# LoRA 配置
# ---------------------------
config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],  # OPT 推荐微调 q_proj / v_proj
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# ---------------------------
# 训练设置
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader_affirm = DataLoader(train_data_affirm, shuffle=False, batch_size=1)
train_loader_reject = DataLoader(train_data_reject, shuffle=False, batch_size=1)
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 10
batch_size = 8
cnt = 0
loss_all = torch.tensor([0.0]).to(device)
optimizer.zero_grad()

# ---------------------------
# 训练循环
# ---------------------------
for epoch in range(num_epochs):
    for batch_affirm, batch_reject in tqdm(zip(train_loader_affirm, train_loader_reject), total=len(train_loader_affirm)):

        # Affirm batch
        input_ids = torch.tensor(batch_affirm['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(batch_affirm['attention_mask']).unsqueeze(0).to(device)
        labels = torch.tensor(batch_affirm['labels']).unsqueeze(0).to(device)
        loss_affirm = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

        # Reject batch
        input_ids = torch.tensor(batch_reject['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(batch_reject['attention_mask']).unsqueeze(0).to(device)
        labels = torch.tensor(batch_reject['labels']).unsqueeze(0).to(device)
        loss_reject = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

        loss_all += (loss_reject / loss_affirm)

        # 梯度更新
        if cnt != 0 and cnt % batch_size == 0:
            loss_mean = loss_all / batch_size
            print(f"Step {cnt}, loss: {loss_mean.item()}")
            loss_mean.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_all = torch.tensor([0.0]).to(device)

        # 保存 LoRA 模型
        if cnt % 80 == 0:
            output_dir = f'./resp_opt/{cnt}_lora'
            model.save_pretrained(output_dir)

        cnt += 1

# 保存最终模型
# model.save_pretrained('./resp_opt/final_lora')
print("训练完成。")
