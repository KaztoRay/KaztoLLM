import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. 설정 및 경로
MODEL_ID = "meta-llama/Meta-Llama-3-8B" # 또는 13B (64GB RAM이므로 시도 가능)
DATASET_PATH = "security_dataset.jsonl" # 준비한 방대한 데이터 파일
OUTPUT_DIR = "./joon-security-llm-v2"

# 2. 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cpu", # 연탄맥 CPU 활용
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

# 3. LoRA 설정 (더 깊은 학습을 위해 rank를 높임)
# r=16 이상으로 설정하여 복잡한 보안 논리 학습
config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # 더 많은 레이어 학습
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

# 4. 데이터셋 로드 및 전처리
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    contexts = examples.get("context", "")
    responses = examples["response"]
    texts = []
    for instruction, context, response in zip(instructions, contexts, responses):
        text = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}</s>"
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)
dataset = dataset.map(lambda samples: tokenizer(samples["text"], truncation=True, padding="max_length", max_length=512), batched=True)

# 5. 하이퍼파라미터 설정 (CPU 학습 최적화)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8, # 실제 배치는 16 효과 (RAM 활용)
    num_train_epochs=5,            # 반복 학습으로 지식 각인
    learning_rate=1e-4,
    logging_steps=5,
    save_strategy="steps",
    save_steps=50,
    use_cpu=True,
    report_to="none"
)

# 6. 트레이너 실행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print(f"--- 보안 통합 학습 시작 (분야: Web, System, DeFi, IoT, Game 등) ---")
trainer.train()

# 7. 최종 저장
model.save_pretrained("./final_security_expert_adapter")
print("학습이 완료되었습니다. 어댑터가 저장되었습니다.")