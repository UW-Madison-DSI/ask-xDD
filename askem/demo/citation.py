import requests
from pybtex.plugin import find_plugin
from pybtex.database import parse_string
from typing import Dict

def get_attributes(doc_id: str) -> Dict[str, str]:
    """
    This function gets the attributes of a document from the XDD API (given by its doc_id).
    This function assumes the XDD API provides a JSON document with certain fields.

    Args:
        doc_id (str): The document ID to get attributes for.

    Returns:
        dict: The attributes of the document.
    """

    url = f"https://xdd.wisc.edu/api/articles?docid={doc_id}"

    response = requests.get(url, timeout=2)
    data = response.json()
    article = data["success"]["data"][0]

    attributes = {
        "id": article["_gddid"]
    }

    if "author" in article and article["author"]:
        authors = " and ".join([author["name"] for author in article["author"]])
        attributes["author"] = authors

    other_fields = [
        "title",
        "year",
        "volume",
        "journal",
        "pages",
        "number",
        "publisher",
    ]

    for field in other_fields:
        if field in article and article[field]:
            attributes[field] = article[field]


    if "link" in article and article["link"]:
        attributes["url"] = article["link"][0]["url"]

    return attributes

def to_bibtex(attrs: Dict) -> str:
    """
    This function formats a dictionary of document attributes into a BibTeX string.

    Args:
        attrs (dict): A dictionary containing document attributes.

    Returns:
        str: The BibTeX string representing the document.
    """

    bibtex = "@article{{{id},\n".format(id=attrs["id"])
    fields = [
        "author",
        "title",
        "year",
        "volume",
        "journal",
        "pages",
        "number",
        "publisher",
        "url",
    ]

    for field in fields:
        if field in attrs and attrs[field]:
            bibtex += f"    {field} = {{{attrs[field]}}},\n"
    bibtex += "}"
    return bibtex

def to_citation(bibtex: str, in_text: bool = False) -> str:
    """
    This function converts a BibTeX string into an APA citation string using the pybtex library.
    It can either generate a full citation or an in-text citation based on the in_text parameter.

    Args:
        bibtex (str): The BibTeX string to convert to APA.
        in_text (bool, optional): Whether to return an in-text citation or not. Defaults to False.

    Returns:
        str: The APA citation string.
    """

    bib_data = parse_string(bibtex, "bibtex")
    apa_style = find_plugin("pybtex.style.formatting", "apa")()
    bibliography = apa_style.format_bibliography(bib_data)
    for entry in bibliography:
        citation = entry.text.render_as("html")
        if not in_text:
            return citation

        author_year = citation.split(")")[0]
        return author_year + ")"

def format_citation(attrs: dict, in_text=False):
    """
    This function formats a dictionary of document attributes into an APA citation.
    It manually formats the citation when the conversion from BibTeX to APA fails, adding robustness to the code.

    Args:
        attrs (dict): A dictionary containing document attributes.
        in_text (bool, optional): Whether to return an in-text citation or not. Defaults to False.

    Returns:
        str: The APA citation string.
    """

    citation = ""
    if attrs.get('author'):
        citation += f"{attrs['author']}"

    if attrs.get('year'):
        citation += f"\n({attrs['year']})."
    if attrs.get('title'):
        citation += f"\n{attrs['title']}."

    if attrs.get('journal'):
        citation += f"\n<em>{attrs['journal']}</em>"
        if attrs.get('volume'):
            citation += f", <em>{attrs['volume']}</em>"
        if attrs.get('number'):
            citation += f"({attrs['number']})"

    if attrs.get('pages'):
        citation += f".\npp. {attrs['pages']}"

    if attrs.get('url'):
        citation += f"\nURL: <a href=\"{attrs['url']}\">{attrs['url']}</a>"
    
    if in_text:
        if attrs.get('year'):
            citation = f"({attrs['year']})"
        else:
            citation = ""
    citation = citation.strip()
    return citation

def to_apa(doc_id: str, in_text: bool = False) -> str:
    """
    This function converts a document from the XDD API into an APA citation.
    It first gets the document attributes using get_attributes and then tries to generate the citation using bibtex_to_apa.
    If an error occurs during this process, it falls back to manual formatting using format_citation.

    Args:
        doc_id (str): The document ID to convert to APA.
        in_text (bool, optional): Whether to return an in-text citation or not. Defaults to False.

    Returns:
        str: The APA citation string representing the document.
    """

    attrs = get_attributes(doc_id)
    try:
        bibtex = to_bibtex(attrs)
        citation = to_citation(bibtex, in_text=in_text)
    except Exception as e:
        citation = format_citation(attrs, in_text=in_text)

    return citation
