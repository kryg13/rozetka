from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, WebDriverException
import time
import csv
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Категорії Rozetka
categories = {
    "Телефони": "https://rozetka.com.ua/ua/mobile-phones/c80003/",
    "Телевізори": "https://rozetka.com.ua/ua/televisions/c80037/",
    "Пилососи": "https://rozetka.com.ua/ua/vacuum-cleaners/c80125/",
    "Дитячі іграшки": "https://rozetka.com.ua/ua/toys/c83849/"
}

# Налаштування Selenium
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
all_reviews = []

def check_page_accessible(url):
    """Перевіряє, чи доступна сторінка"""
    try:
        time.sleep(2)  # Затримка перед запитом
        driver.get(url)
        return True
    except WebDriverException as e:
        logger.error(f"Не вдалося отримати доступ до {url}: {e}")
        return False

def get_product_links(category_url, max_products=1000):
    """Збирає посилання на продукти з категорії"""
    if not check_page_accessible(category_url):
        logger.error(f"Сторінка категорії {category_url} недоступна")
        return []

    product_links = set()
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

def get_reviews(product_url, max_reviews=50):
    """Збирає відгуки з сторінки товару"""
    reviews = []
    url = product_url.rstrip("/") + "/comments/"
    try:
        if not check_page_accessible(url):
            logger.error(f"Не вдалося завантажити сторінку відгуків {url}")
            return reviews
        time.sleep(5)  # Затримка для завантаження сторінки, як у вашому коді
        
        # Клікаємо "Показати ще", поки є
        while True:
            try:
                show_more_button = driver.find_element(By.XPATH, '//button[contains(text(), "Показати ще")]')
                driver.execute_script("arguments[0].click();", show_more_button)
                time.sleep(2)
            except NoSuchElementException:
                break

        # Основні коментарі
        review_elements = driver.find_elements(By.CLASS_NAME, "comment__text")
        for el in review_elements:
            text = el.text.strip()
            if text and len(reviews) < max_reviews:
                reviews.append(text)

        # Відповіді брендів
        reply_elements = driver.find_elements(By.CLASS_NAME, "reply__body")
        for el in reply_elements:
            text = el.text.strip()
            if text and len(reviews) < max_reviews:
                reviews.append("🔁 " + text)
                
    except WebDriverException as e:
        logger.error(f"Помилка: Не вдалося завантажити відгуки для {url}: {e}")
    return reviews[:max_reviews]

# Основна логіка
for category_name, category_url in categories.items():
    logger.info(f"\n📦 Категорія: {category_name}")
    product_links = get_product_links(category_url, max_products=300)

    for product_url in product_links:
        logger.info(f"  → Витягую відгуки: {product_url}/comments/")
        reviews = get_reviews(product_url, max_reviews=50)
        logger.info(f"     Знайдено відгуків: {len(reviews)}")

        for review in reviews:
            all_reviews.append([category_name, product_url, review])
        time.sleep(2)  # Затримка між запитами
        if len(all_reviews) >= 10000:
            break
    if len(all_reviews) >= 10000:
        break

# Збереження результатів
output_file = "rozetka_reviews.csv"
with open(output_file, mode="w", encoding="utf-8-sig", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Категорія", "URL товару", "Відгук"])
    writer.writerows(all_reviews)

logger.info(f"\n✅ Готово! Зібрано {len(all_reviews)} відгуків. Збережено у: {output_file}")
driver.quit()