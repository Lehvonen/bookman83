from urllib.request import urlopen as uOpen
from urllib.parse import urljoin
from bs4 import BeautifulSoup as bSoup
import json

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
def get_book_description(book_url):
    soup = get_soup(book_url)
    desc_heading = soup.find("div", id="product_description")
    if desc_heading is None:
        return ""
    desc_paragraph = desc_heading.find_next_sibling("p")
    return desc_paragraph.text.strip() if desc_paragraph else ""


# scrape every book across every page of one category
# returns {book_title: {img, description, price} }
def scrape_category(start_url):
    books_data = {}
 
    for soup in get_category_pages(start_url):
        listings = soup.findAll("li", {"class": "col-xs-6 col-sm-4 col-md-3 col-lg-3"})
 
        for book in listings:
            title = book.h3.a["title"]
            price = book.find("p", {"class": "price_color"}).text.strip()
 
            img_relative = book.find("img")["src"]
            img_url = urljoin(start_url, img_relative)
 
            book_relative_link = book.h3.a["href"]
            book_url = urljoin(start_url, book_relative_link)
            description = get_book_description(book_url)
 
            print(f"Scraped: {title}")
 
            books_data[title] = {
                "img": img_url,
                "description": description,
                "price": price
            }
           
    return books_data


def main():
    print("Scraping Fiction...")
    fiction = scrape_category(urlFiction)
 
    print("Scraping Nonfiction...")
    nonFiction = scrape_category(urlNonFiction)
 
    output = {
        "fiction": fiction,
        "nonFiction": nonFiction
    }
 
    with open("books_data_raw.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
 

if __name__ == "__main__":
    main()