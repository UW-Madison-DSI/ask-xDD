import requests
from pybtex.plugin import find_plugin
from pybtex.database import parse_string

def json_to_bibtex(docID: str) -> str:
    """
    This function converts a document from the XDD API (given by its docID) into a BibTeX string.
    This function assumes the XDD API provides a JSON document with certain fields.

    Args:
        docID (str): The document ID to convert to BibTeX.

    Returns:
        str: The BibTeX string representing the document.
    """

    url = "https://xdd.wisc.edu/api/articles?docid={}".format(docID)
    response = requests.get(url)
    data = response.json()

    if not data['success']:
        print("Request failed")
        return None

    article = data['success']['data'][0]

    if 'author' not in article or 'title' not in article:
        print("Required field missing")
        return None

    authors = ' and '.join([author['name'] for author in article['author']])

    bibtex = "@article{{{id},\n".format(id=article['_gddid'])
    bibtex += "    title = {{{title}}},\n".format(title=article['title'])
    bibtex += "    author = {{{author}}},\n".format(author=authors)

    optional_fields = ['year', 'volume', 'journal', 'pages', 'number', 'publisher', 'abstract']

    for field in optional_fields:
        if field in article and article[field]:
            bibtex += "    {field} = {{{value}}},\n".format(field=field, value=article[field])

    if 'link' in article and article['link']:
        bibtex += "    url = {{{url}}},\n".format(url=article['link'][0]['url'])

    bibtex += "}"

    return bibtex

def bibtex_to_apa(bibtex_str: str) -> str:
    """
    This function converts a BibTeX string into an APA citation string using the pybtex library.

    Args:
        bibtex_str (str): The BibTeX string to convert to APA.

    Returns:
        str: The APA citation string.
    """

    ...

    bib_data = parse_string(bibtex_str, 'bibtex')
    apa_style = find_plugin('pybtex.style.formatting', 'apa')()
    bibliography = apa_style.format_bibliography(bib_data)
    for entry in bibliography:
        # only one entry in bibliography
        return entry.text.render_as('text')

def json_to_apa(docID: str) -> str:
    """
    This function converts a document from the XDD API (given by its docID) into an APA citation string.
    This function uses the json_to_bibtex and bibtex_to_apa functions.

    Args:
        docID (str): The document ID to convert to APA.

    Returns:
        str: The APA citation string representing the document.
    """
    bibtex_str = json_to_bibtex(docID)
    apa_str = bibtex_to_apa(bibtex_str)
    return apa_str

if __name__ == "__main__":
    bibtex_str = json_to_bibtex("5a46f33acf58f18e3365e42b")
    print(bibtex_str)
    apa_str = bibtex_to_apa(bibtex_str)
    print(apa_str)