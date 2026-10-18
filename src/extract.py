import sys
import pymupdf

PDF_PATH = "data/RM0090.pdf"


def main():
    page_number = int(sys.argv[1])        
    doc = pymupdf.open(PDF_PATH)
    print(f"Pages in document: {doc.page_count}")

    page = doc[page_number - 1]           
    print(page.get_text())


if __name__ == "__main__":
    main()