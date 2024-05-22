from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from datasets import load_dataset

# Cargar el modelo y el tokenizador
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Cargar y preparar el conjunto de datos
dataset = load_dataset('squad')
train_dataset = dataset['train'].map(lambda x: tokenizer(x['question'], x['context'], truncation=True, padding='max_length'), batched=True)
eval_dataset = dataset['validation'].map(lambda x: tokenizer(x['question'], x['context'], truncation=True, padding='max_length'), batched=True)

# Definir los argumentos del entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Crear el trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Ajustar el modelo
trainer.train()
