from urllib.request import urlopen as uOpen
from urllib.parse import urljoin
from bs4 import BeautifulSoup as bSoup
import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

# store url in variable urlFiction and urlNonFiction
urlFiction = "https://books.toscrape.com/catalogue/category/books/fiction_10/index.html"
urlNonFiction = "https://books.toscrape.com/catalogue/category/books/nonfiction_13/index.html"

# read and close HTMLs
def get_soup(url):
    page = uOpen(url)
    html = page.read()
    page.close()
    return bSoup(html, "html.parser")

# call BeautifulSoup for parsing
def get_category_pages(start_url):
    soups = []
    page_url = start_url
 
    while True:
        soup = get_soup(page_url)
        soups.append(soup)
 
        next_li = soup.find("li", {"class": "next"})
        if next_li is None:
            break
 
        # next_li's <a href="..."> is relative to the current page
        page_url = urljoin(page_url, next_li.a["href"])
 
    return soups


# open specific book page and pull the description
def get_book_details(book_url):
    soup = get_soup(book_url)

    # description
    description = ""
    desc_heading = soup.find("div", id="product_description")
    if desc_heading is not None:
        desc_paragraph = desc_heading.find_next_sibling("p")
        if desc_paragraph: 
            description = desc_paragraph.text.strip()    

    # stock UPC and rating
    table = soup.find("table", {"class": "table table-striped"})
    rows = {}
    for tr in table.find_all("tr"):
        rows[tr.th.text.strip()] = tr.td.text.strip()

    upc = rows["UPC"]
    stock = int(re.search(r"\d+", rows["Availability"]).group())

    words = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    star_tag = soup.find("p", class_="star-rating")
    rating = words[star_tag["class"][1]]

    return {
        "description": description,
        "upc": upc,
        "rating": rating,
        "stock": stock
    }
#image downloader 
def download_image(img_url, upc):
    relative_path = Path("data") / "images" / f"{upc}.jpg"
    img_path = PROJECT_DIR / relative_path
    img_path.parent.mkdir(parents=True, exist_ok=True)

    if img_path.exists():
        return relative_path.as_posix()
    
    try:
        data = uOpen(img_url).read()
        with img_path.open("wb") as f:
            f.write(data)
    except Exception as e:
        print(f"Image failed for {upc}: {e}")
        return ""

    return relative_path.as_posix()

# scrape every book across every page of one category
# returns {book_title: {img, description, price} }
def scrape_category(start_url, category):
    records = []
 
    for soup in get_category_pages(start_url):
        listings = soup.findAll("li", {"class": "col-xs-6 col-sm-4 col-md-3 col-lg-3"})
 
        for book in listings:
            title = book.h3.a["title"]
            price = book.find("p", {"class": "price_color"}).text.strip()
 
            img_relative = book.find("img")["src"]
            img_url = urljoin(start_url, img_relative)
 
            book_relative_link = book.h3.a["href"]
            book_url = urljoin(start_url, book_relative_link)
            details = get_book_details(book_url)
            img_path = download_image(img_url, details["upc"])
 
            print(f"Scraped: {title}")
 
            records.append({
                "title": title,
                "category": category,
                "img": img_url,
                "img_path": img_path,
                "description": details["description"],
                "price": price,
                "upc": details ["upc"],
                "rating": details["rating"],
                "stock": details["stock"]
            })
           
    return records


def main():
    print("Scraping Fiction...")
    fiction = scrape_category(urlFiction, "Fiction")
 
    print("Scraping Nonfiction...")
    nonFiction = scrape_category(urlNonFiction, "Nonfiction")
 
 
    with (PROJECT_DIR / "books_data_raw.json").open("w", encoding="utf-8") as f:
        records = fiction + nonFiction
        json.dump(records, f, indent=2, ensure_ascii=False)
 

if __name__ == "__main__":
    main()