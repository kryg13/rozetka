import os
import time
import threading
import logging
import csv
import pandas as pd
from flask import Flask, render_template_string, request
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainerCallback
import torch

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
status_lines = []
lock = threading.Lock()

def add_status(line):
    with lock:
        status_lines.append(line)
        logger.info(line)

def get_status():
    with lock:
        return "\n".join(status_lines)

def clear_status():
    with lock:
        status_lines.clear()

# --- Збір відгуків ---
def get_reviews(product_id=None, category_url=None, max_products=100, max_reviews_per_product=50):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    all_reviews = []

    try:
        if product_id:
            url = f'https://rozetka.com.ua/ua/{product_id}/comments/'
            logger.info(f"Збір відгуків для product_id: {product_id}")
            reviews = collect_reviews_from_url(driver, url, max_reviews_per_product)
            all_reviews.extend([["Single Product", url, review] for review in reviews])
        elif category_url:
            logger.info(f"Збір відгуків для категорії: {category_url}")
            product_links = get_product_links(driver, category_url, max_products)
            for product_url in product_links:
                review_url = product_url.rstrip("/") + "/comments/"
                reviews = collect_reviews_from_url(driver, review_url, max_reviews_per_product)
                all_reviews.extend([[category_url.split('/')[-2], product_url, review] for review in reviews])
                time.sleep(2)
                if len(all_reviews) >= 10000:
                    break
    finally:
        driver.quit()
    
    add_status(f"Зібрано {len(all_reviews)} відгуків загалом")
    return all_reviews

def get_product_links(driver, category_url, max_products):
    product_links = set()
    if not check_page_accessible(driver, category_url):
        logger.error(f"Сторінка категорії {category_url} недоступна")
        return []
    
    while len(product_links) < max_products:
        time.sleep(2)
        elements = driver.find_elements(By.CSS_SELECTOR, "a.tile-title")
        for el in elements:
            link = el.get_attribute("href")
            if link and "/p" in link:
                product_links.add(link.split('?')[0])
        try:
            next_btn = driver.find_element(By.CSS_SELECTOR, "a.pagination__direction--forward")
            next_btn.click()
        except NoSuchElementException:
            logger.info(f"Більше сторінок для {category_url} немає")
            break
        except WebDriverException as e:
            logger.error(f"Помилка при переході на наступну сторінку {category_url}: {e}")
            break
    return list(product_links)[:max_products]

def check_page_accessible(driver, url):
    try:
        time.sleep(2)
        driver.get(url)
        return True
    except WebDriverException as e:
        logger.error(f"Не вдалося отримати доступ до {url}: {e}")
        return False

def collect_reviews_from_url(driver, url, max_reviews):
    reviews = []
    try:
        if not check_page_accessible(driver, url):
            return reviews
        time.sleep(5)
        
        while True:
            try:
                show_more_button = driver.find_element(By.XPATH, '//button[contains(text(), "Показати ще")]')
                driver.execute_script("arguments[0].click();", show_more_button)
                time.sleep(2)
            except NoSuchElementException:
                break

        review_elements = driver.find_elements(By.CLASS_NAME, "comment__text")
        for el in review_elements:
            text = el.text.strip()
            if text and len(reviews) < max_reviews:
                reviews.append(text)

        reply_elements = driver.find_elements(By.CLASS_NAME, "reply__body")
        for el in reply_elements:
            text = el.text.strip()
            if text and len(reviews) < max_reviews:
                reviews.append("🔁 " + text)
    except WebDriverException as e:
        logger.error(f"Помилка при зборі відгуків для {url}: {e}")
    return reviews[:max_reviews]

def save_reviews_to_file(reviews, filename="reviews.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for _, _, review in reviews:
            clean = review.replace('\n', ' ').strip()
            if clean:
                f.write(clean + "\n")
    add_status(f"Збережено {len(reviews)} відгуків у {filename}")

def save_reviews_to_csv(reviews, filename="rozetka_reviews.csv"):
    with open(filename, mode="w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Категорія", "URL товару", "Відгук"])
        writer.writerows(reviews)
    add_status(f"Збережено {len(reviews)} відгуків у {filename}")

# --- Підготовка та навчання моделі ---
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log_str = ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in logs.items())
            add_status(f"[Epoch {state.epoch:.2f} Step {state.global_step}] {log_str}")

def train_gptneo(txt_path="reviews.txt", output_dir="gptneo_rozetka_model", epochs=1):
    add_status("Завантаження токенізатора та моделі GPT-Neo...")
    model_name = "EleutherAI/gpt-neo-1.3B"
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name)
    except Exception as e:
        add_status(f"Помилка при завантаженні моделі або токенізатора: {e}")
        return False
    
    tokenizer.pad_token = tokenizer.eos_token
    model.to(torch.device("cpu"))

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=txt_path,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        logging_dir='./logs',
        report_to=[],
        prediction_loss_only=True,
        learning_rate=2e-5,
        warmup_steps=100,
        fp16=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[LoggingCallback()]
    )

    add_status("Починаємо навчання GPT-Neo...")
    try:
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        add_status(f"Модель збережена в: {output_dir}")
        return True
    except Exception as e:
        add_status(f"Помилка під час навчання: {e}")
        return False

# --- Донавчання моделі ---
def finetune_gptneo(product_id, model_dir="gptneo_rozetka_model", txt_path="reviews.txt"):
    clear_status()
    if not os.path.exists(model_dir):
        add_status(f"Модель у {model_dir} не знайдена. Спочатку натренуйте модель за допомогою train.py.")
        return False

    add_status(f"Починаємо донавчання моделі для product_id: {product_id}")
    reviews = get_reviews(product_id=product_id, max_reviews_per_product=50)
    if not reviews:
        add_status("Відгуків не знайдено, припинення.")
        return False

    save_reviews_to_file(reviews, txt_path)
    save_reviews_to_csv(reviews)

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        model = GPTNeoForCausalLM.from_pretrained(model_dir)
    except Exception as e:
        add_status(f"Помилка при завантаженні моделі або токенізатора: {e}")
        return False

    tokenizer.pad_token = tokenizer.eos_token
    model.to(torch.device("cpu"))

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=txt_path,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=model_dir,
        overwrite_output_dir=False,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        logging_dir='./logs',
        report_to=[],
        prediction_loss_only=True,
        learning_rate=1e-5,  # Зменшена швидкість для донавчання
        warmup_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[LoggingCallback()]
    )

    add_status("Починаємо донавчання GPT-Neo...")
    try:
        trainer.train()
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)
        add_status(f"Модель донавчена і збережена в: {model_dir}")
        return True
    except Exception as e:
        add_status(f"Помилка під час донавчання: {e}")
        return False

# --- Генерація тексту ---
def generate_text(prompt, model_dir="gptneo_rozetka_model", max_length=100, content_type="review"):
    if not os.path.exists(model_dir):
        add_status(f"Модель у {model_dir} не знайдена. Будь ласка, натренуйте модель.")
        return "Модель не знайдена. Спочатку натренуйте модель."

    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPTNeoForCausalLM.from_pretrained(model_dir)
    model.to(torch.device("cpu"))

    if content_type == "ad":
        prompt = f"Створи короткий рекламний опис для товару: {prompt}"
    inputs = tokenizer(prompt, return_tensors="pt").to(torch.device("cpu"))
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Фонова задача ---
def background_task(product_id=None, category_url=None, generate_type="review"):
    clear_status()
    if product_id:
        add_status(f"Починаємо обробку product_id: {product_id}")
        reviews = get_reviews(product_id=product_id)
        if not reviews:
            add_status("Відгуків не знайдено, припинення.")
            return
        save_reviews_to_file(reviews)
        save_reviews_to_csv(reviews)
        if finetune_gptneo(product_id):
            prompt = f"Напиши відгук про товар з id {product_id}" if generate_type == "review" else f"Опис товару з id {product_id}"
            sample = generate_text(prompt, content_type=generate_type)
            add_status(f"\n--- Згенерований {'відгук' if generate_type == 'review' else 'рекламний опис'} ---")
            add_status(sample)
    elif category_url:
        add_status(f"Починаємо обробку категорії: {category_url}")
        reviews = get_reviews(category_url=category_url)
        if not reviews:
            add_status("Відгуків не знайдено, припинення.")
            return
        save_reviews_to_file(reviews)
        save_reviews_to_csv(reviews)
        if train_gptneo():
            prompt = f"Напиши відгук про товар" if generate_type == "review" else "Опис товару"
            sample = generate_text(prompt, content_type=generate_type)
            add_status(f"\n--- Згенерований {'відгук' if generate_type == 'review' else 'рекламний опис'} ---")
            add_status(sample)
    else:
        add_status("Помилка: Вкажіть product_id або category_url")

# --- HTML-шаблон ---
INDEX_HTML = """
<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Генерація тексту для Rozetka</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 font-sans">
    <div class="container mx-auto p-4">
        <h1 class="text-2xl font-bold mb-4">Генерація тексту для Rozetka</h1>
        <div class="bg-white p-6 rounded shadow">
            <form id="inputForm" class="mb-4">
                <label class="block mb-2">Введіть product_id (наприклад, ergo-43gus6500/p362684427) або URL категорії:</label>
                <input type="text" name="input_id" class="w-full p-2 border rounded mb-2" placeholder="Введіть product_id або URL">
                <label class="block mb-2">Тип контенту:</label>
                <select name="content_type" class="p-2 border rounded mb-2">
                    <option value="review">Відгук</option>
                    <option value="ad">Рекламний опис</option>
                </select>
                <div class="flex space-x-2">
                    <button type="button" onclick="startTask()" class="bg-blue-500 text-white p-2 rounded hover:bg-blue-600">Запустити</button>
                </div>
            </form>
            <h2 class="text-xl font-semibold mb-2">Статус:</h2>
            <pre id="status" class="bg-gray-100 p-4 rounded h-64 overflow-auto">Очікування запуску...</pre>
            <h2 class="text-xl font-semibold mb-2">Згенерований текст:</h2>
            <pre id="generated" class="bg-gray-100 p-4 rounded"></pre>
        </div>
    </div>
    <script>
        async function startTask() {
            const form = document.getElementById('inputForm');
            const input = form.querySelector('input[name="input_id"]').value;
            const contentType = form.querySelector('select[name="content_type"]').value;
            let url = '';
            if (input.includes('rozetka.com.ua')) {
                url = `/parse?category_url=${encodeURIComponent(input)}&content_type=${contentType}`;
            } else {
                url = `/parse?product_id=${encodeURIComponent(input)}&content_type=${contentType}`;
            }
            document.getElementById('status').innerText = 'Запущено обробку...';
            await fetch(url);
            updateStatus();
        }

        async function updateStatus() {
            const response = await fetch('/status');
            const status = await response.text();
            document.getElementById('status').innerText = status || 'Очікування...';
            const generated = status.split('--- Згенерований')[1];
            if (generated) {
                document.getElementById('generated').innerText = generated.trim();
            }
            setTimeout(updateStatus, 2000);
        }
        updateStatus();
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/parse")
def parse():
    product_id = request.args.get("product_id")
    category_url = request.args.get("category_url")
    content_type = request.args.get("content_type", "review")
    if not product_id and not category_url:
        return "Вкажіть product_id або category_url", 400
    threading.Thread(target=background_task, args=(product_id, category_url, content_type), daemon=True).start()
    return "Запущено обробку"

@app.route("/status")
def status():
    return get_status()

if __name__ == "__main__":
    app.run(debug=True, port=5000)