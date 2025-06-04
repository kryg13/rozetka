import os
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def check_requirements():
    """
    Перевіряє наявність потрібних бібліотек.
    """
    try:
        import torchvision
        print(f"torchvision версія: {torchvision.__version__}")
    except ImportError as e:
        print(f"Помилка з torchvision: {e}")
        print("Спробуйте оновити torchvision: pip install --upgrade torchvision")
        return False
    print(f"PyTorch версія: {torch.__version__}")
    print("Навчання відбуватиметься на CPU.")
    return True

def prepare_text_from_csv(csv_path="rozetka_reviews.csv", txt_path="reviews_temp.txt"):
    """
    Конвертує відгуки з колонки 'Відгук' у CSV у текстовий файл для навчання моделі.
    """
    if not os.path.exists(csv_path):
        print(f"Помилка: Файл {csv_path} не знайдено.")
        return False
    
    print(f"Читання CSV-файлу {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        if 'Відгук' not in df.columns:
            print("Помилка: У CSV-файлі відсутня колонка 'Відгук'.")
            return False
        
        reviews = df['Відгук'].dropna().tolist()
        with open(txt_path, "w", encoding="utf-8") as f:
            for review in reviews:
                clean_review = str(review).replace('\n', ' ').strip()
                if clean_review:
                    f.write(clean_review + "\n")
        print(f"Збережено {len(reviews)} відгуків у {txt_path}")
        return True
    except Exception as e:
        print(f"Помилка при обробці CSV: {e}")
        return False

def train_gptneo(csv_path="rozetka_reviews.csv", output_dir="gptneo_rozetka_model", model_name="EleutherAI/gpt-neo-1.3B"):
    """
    Навчає модель GPT-Neo на CPU на основі відгуків із CSV-файлу та зберігає її.
    """
    # Перевірка вимог
    if not check_requirements():
        print("Не вдалося перевірити вимоги. Перевірте встановлені бібліотеки.")
        return

    # Явне використання CPU
    device = torch.device("cpu")
    print(f"Використовується пристрій: {device}")

    # Конвертація CSV у текстовий файл
    txt_path = "reviews_temp.txt"
    if not prepare_text_from_csv(csv_path, txt_path):
        print("Не вдалося підготувати текстовий файл для навчання.")
        return

    # Завантаження токенізатора та моделі
    print("Завантаження токенізатора та моделі GPT-Neo...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name)
    except Exception as e:
        print(f"Помилка при завантаженні моделі або токенізатора: {e}")
        return
    tokenizer.pad_token = tokenizer.eos_token  # Встановлення pad_token

    # Перенесення моделі на CPU
    model.to(device)

    # Підготовка датасету
    print("Підготовка датасету...")
    try:
        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=txt_path,
            block_size=128
        )
    except Exception as e:
        print(f"Помилка при підготовці датасету: {e}")
        return

    # Налаштування колатора даних
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Налаштування параметрів навчання
    print("Налаштування параметрів навчання...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,  # Зменшено для CPU
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        prediction_loss_only=True,
        learning_rate=2e-5,
        warmup_steps=100,
        fp16=False  # FP16 вимкнено для CPU
    )

    # Ініціалізація тренера
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Навчання моделі
    print("Починаємо навчання GPT-Neo...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Помилка під час навчання: {e}")
        return

    # Збереження моделі та токенізатора
    print(f"Збереження моделі та токенізатора в {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Модель успішно збережена в {output_dir}")

    # Видалення тимчасового текстового файлу
    if os.path.exists(txt_path):
        os.remove(txt_path)
        print(f"Тимчасовий файл {txt_path} видалено.")

if __name__ == "__main__":
    train_gptneo(csv_path="rozetka_reviews.csv")