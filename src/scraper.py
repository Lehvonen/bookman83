from urllib.request import urlopen as uOpen
from bs4 import BeautifulSoup as bSoup

# grab website and store in variable urlFiction and urlNonFiction
urlFiction = uOpen("https://books.toscrape.com/catalogue/category/books/fiction_10/index.html")
urlNonFiction = uOpen("https://books.toscrape.com/catalogue/category/books/nonfiction_13/index.html")

# read and close both HTMLs
FictionPageHtml = urlFiction.read()
urlFiction.close()

NonFictionPageHtml = urlNonFiction.read()
urlNonFiction.close()

# call BeautifulSoup for parsing
fictionSoup = bSoup(FictionPageHtml, "html.parser")
nonFictionSoup = bSoup(NonFictionPageHtml, "html.parser")

# grabs all books from both fiction and nonfiction
fiction = fictionSoup.findAll("li", {"class": "col-xs-6 col-sm-4 col-md-3 col-lg-3"})
nonFiction = nonFictionSoup.findAll("li", {"class": "col-xs-6 col-sm-4 col-md-3 col-lg-3"})

# create csv file of all books in seperate files
fictionData = ("fictionData.json")
f = open(fictionData, "w")

nonFictionData = ("nonFictionData.json")
nf = open(nonFictionData, "w")

headers = "Book title, Price\n"

f.write(headers)

for books in fiction:

    # collect title of all books
    book_title = books.h3.a["title"]

    # collect book price of all books
    book_price = books.findAll("p", {"class": "price_color"})
    price = book_price[0].text.strip()

    print("Title of the book :" + book_title)
    print("Price of the book :" + price)

    f.write(book_title + "," + price+"\n")

f.close()


nf.write(headers)

for books in nonFiction:

    # collect title of all books
    book_title = books.h3.a["title"]

    # collect book price of all books
    book_price = books.findAll("p", {"class": "price_color"})
    price = book_price[0].text.strip()

    print("Title of the book :" + book_title)
    print("Price of the book :" + price)

    nf.write(book_title + "," + price+"\n")

nf.close()