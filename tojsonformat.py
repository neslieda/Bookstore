import pandas as pd

file_path = "books.csv"

df = pd.read_csv(file_path)

df.head()

books_json_updated = df[['Title', 'Rating', 'Price', 'Stock Quantity', 'Availability', 'Description']].to_json(orient='records', force_ascii=False)

json_file_path_updated = "books_updated.json"
with open(json_file_path_updated, "w", encoding="utf-8") as json_file:
    json_file.write(books_json_updated)

json_file_path_updated
