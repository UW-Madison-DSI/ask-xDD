import requests
from pybtex.plugin import find_plugin
from pybtex.database import parse_string


def to_bibtex(doc_id: str) -> str:
    """
    This function converts a document from the XDD API (given by its doc_id) into a BibTeX string.
    This function assumes the XDD API provides a JSON document with certain fields.

    Args:
        doc_id (str): The document ID to convert to BibTeX.

    Returns:
        str: The BibTeX string representing the document.
    """

    url = f"https://xdd.wisc.edu/api/articles?docid={doc_id}"

    response = requests.get(url, timeout=2)
    data = response.json()
    article = data["success"]["data"][0]

    authors = " and ".join([author["name"] for author in article["author"]])

    bibtex = "@article{{{id},\n".format(id=article["_gddid"])
    bibtex += "    title = {{{title}}},\n".format(title=article["title"])
    bibtex += "    author = {{{author}}},\n".format(author=authors)

    optional_fields = [
        "year",
        "volume",
        "journal",
        "pages",
        "number",
        "publisher",
        "abstract",
    ]

    for field in optional_fields:
        if field in article and article[field]:
            bibtex += "    {field} = {{{value}}},\n".format(
                field=field, value=article[field]
            )

    if "link" in article and article["link"]:
        bibtex += "    url = {{{url}}},\n".format(url=article["link"][0]["url"])

    bibtex += "}"
    return bibtex


def bibtex_to_apa(bibtex: str) -> str:
    """
    This function converts a BibTeX string into an APA citation string using the pybtex library.

    Args:
        bibtex (str): The BibTeX string to convert to APA.

    Returns:
        str: The APA citation string.
    """

    bib_data = parse_string(bibtex, "bibtex")
    apa_style = find_plugin("pybtex.style.formatting", "apa")()
    bibliography = apa_style.format_bibliography(bib_data)
    for entry in bibliography:
        # only one entry in bibliography
        return entry.text.render_as("text")


def to_apa(doc_id: str) -> str:
    """
    This function converts a document from the XDD API (given by its doc_id) into an APA citation string.
    This function uses the json_to_bibtex and bibtex_to_apa functions.

    Args:
        doc_id (str): The document ID to convert to APA.

    Returns:
        str: The APA citation string representing the document.
    """
    bibtex = to_bibtex(doc_id)
    citation = bibtex_to_apa(bibtex)
    return citation
