from urllib.request import urlopen as uOpen
from urllib.parse import urljoin
from bs4 import BeautifulSoup as bSoup
import json
import re

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
 
            print(f"Scraped: {title}")
 
            records.append({
                "title": title,
                "category": category,
                "img": img_url,
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
 
 
    with open("books_data_raw.json", "w", encoding="utf-8") as f:
        records = fiction + nonFiction
        json.dump(records, f, indent=2, ensure_ascii=False)
 

if __name__ == "__main__":
    main()