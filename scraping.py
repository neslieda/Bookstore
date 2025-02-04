import requests
from bs4 import BeautifulSoup
import csv

base_url = "https://books.toscrape.com/catalogue/"
page_url = "https://books.toscrape.com/catalogue/page-{}.html"

book_data = []

for page in range(1, 51):
    url = page_url.format(page)
    response = requests.get(url)
    if response.status_code != 200:
        break

    soup = BeautifulSoup(response.text, "html.parser")
    books = soup.select("article.product_pod")

    for book in books:
        title = book.h3.a["title"]
        star_rating = book.p["class"][1]
        price = book.select_one("p.price_color").text

        detail_url = base_url + book.h3.a["href"]
        detail_response = requests.get(detail_url)
        detail_soup = BeautifulSoup(detail_response.text, "html.parser")

        description_tag = detail_soup.select_one("#product_description + p")
        description = description_tag.text.strip() if description_tag else "No description"

        availability_tag = detail_soup.select_one("p.instock.availability")
        availability_text = availability_tag.text.strip() if availability_tag else "Unknown"

        if "In stock" in availability_text and "(" in availability_text:
            stock_qty = int(availability_text.split("(")[-1].split()[0])
        else:
            stock_qty = 0

        book_data.append([title, star_rating, price, availability_text, stock_qty, description])

with open("books.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Title", "Rating", "Price", "Availability", "Stock Quantity", "Description"])
    writer.writerows(book_data)

print("Veriler 'books.csv' dosyasına kaydedildi.")
