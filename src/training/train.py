import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. 모델 설정 (승인 필요 없는 Mistral v0.3)
MODEL_ID = "mistralai/Mistral-7B-v0.3"
DATASET_PATH = "security_dataset_v2.jsonl"
OUTPUT_DIR = "./jkazto-security-v1"

print("🔍 Mistral-7B 모델 및 토크나이저 로딩 중... (승인 절차 없음)")

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# 모델 로드 (CPU 최적화)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

# 2. Mistral용 LoRA 설정
# Mistral은 q, v 외에 k, o, gate, up, down 등 모든 레이어를 학습할 때 보안 지식이 더 잘 주입됩니다.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. 데이터셋 가공 (Mistral 프롬프트 형식 적용)
def formatting_mistral_func(example):
    # Mistral 특유의 [INST] 태그를 사용하여 지시어 이행 능력을 높입니다.
    text = (
        f"<s>[INST] 분야: {example.get('domain', '보안 전문가')}\n"
        f"분석 요청: {example['instruction']}\n"
        f"맥락: {example.get('context', 'N/A')} [/INST]\n"
        f"전문가 분석: {example['response']} </s>"
    )
    return {"text": text}

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(formatting_mistral_func)
tokenized_dataset = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=1024),
    batched=True,
    remove_columns=dataset.column_names
)

# 4. 연탄맥 최적화 학습 설정 (64GB RAM 활용)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=5,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=1,
    save_strategy="epoch",
    use_cpu=True, # 강제 CPU 모드
    report_to="none"
)

# 5. 트레이너 실행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("🔒 [Mistral Security V1] 연탄맥 통합 보안 학습 시작...")
trainer.train()

# 6. 결과 저장
model.save_pretrained("./final_mistral_security_adapter")
print("✅ 학습 완료! './final_mistral_security_adapter'가 생성되었습니다.")