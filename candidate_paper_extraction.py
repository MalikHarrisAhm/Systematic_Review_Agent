import pandas as pd
from Bio import Entrez, Medline
from habanero import Crossref
from altmetric import Altmetric
import requests
import math
import numpy as np
from requests.exceptions import HTTPError
from pyaltmetric import Altmetric
import os
from dotenv import load_dotenv
import time
import json
from pathlib import Path

# Load environment variables
load_dotenv()

class PubMedSearch:
    """
    A class for conducting PubMed searches and retrieving document details.
    """

    def __init__(self, email, api_key):
        """
        Initialize the PubMedSearch object with the required API credentials.

        :param email: The email address associated with the NCBI API key.
        :type email: str
        :param api_key: The NCBI API key.
        :type api_key: str
        """
        Entrez.email = email
        Entrez.api_key = api_key

    def search(self, query, retmax=10000):
        """
        Conduct a PubMed search using the provided query.

        :param query: The PubMed search query.
        :type query: str
        :param retmax: The maximum number of records to retrieve (default: 10000).
        :type retmax: int

        :return: A pandas DataFrame containing the search results.
        :rtype: pd.DataFrame
        """
        handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
        record = Entrez.read(handle)
        handle.close()

        ids = record["IdList"]
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
        records = Medline.parse(handle)

        return pd.DataFrame(list(records))


class CrossrefCitationData:
    """
    A class for retrieving citation data from Crossref.
    """

    def __init__(self):
        """
        Initialize the CrossrefCitationData object.
        """
        self.cr = Crossref()

    def fetch_data(self, doi):
        """
        Fetch citation data for the given DOI.

        :param doi: The DOI (Digital Object Identifier) of the data to fetch.
        :type doi: str

        :return: The fetched data, if successful. None otherwise.
        :rtype: dict or None
        """
        try:
            return self.cr.works(ids=doi)
        except HTTPError as e:
            print(f"Unable to fetch data for DOI: {doi}. Error: {e}")
            return None


def get_altmetrics(doi):
    """
    Retrieve Altmetric data for the given DOI.

    :param doi: The DOI (Digital Object Identifier) of the data to fetch.
    :type doi: str

    :return: A pandas Series containing the Altmetric data, if available. An empty Series otherwise.
    :rtype: pd.Series
    """
    try:
        data = Altmetric().doi(doi)
        return pd.Series(data)
    except:
        return pd.Series()


def main():
    # Get API key from environment variable
    api_key = os.getenv('ENTREZ_API_KEY')
    if not api_key:
        raise ValueError("ENTREZ_API_KEY environment variable not set")
    
    # Initialize PubMed search
    pubmed_search = PubMedSearch(Entrez.email, api_key)

    # Write query
    query = '(COVID-19*[Title/Abstract] OR SARS-CoV-2*[Title/Abstract]) AND symptoms*[Title/Abstract] AND (persistent*[Title/Abstract] OR "long COVID"[Title/Abstract] OR "post-COVID syndrome"[Title/Abstract]) AND english[LA] AND "Journal Article"[PT] AND 2020[DP]'

    # Conduct PubMed search
    df = pubmed_search.search(query)

    urls = []
    dois = []

    for doi in df["AID"]:
        if isinstance(doi, list):
            if "[doi]" in doi[-1]:
                sub_url = doi[-1].replace(" [doi]", "")
                url = "https://doi.org/" + sub_url
                dois.append(sub_url)
                urls.append(url)
        elif isinstance(doi, float):
            if math.isnan(doi):
                print("The variable is a float and its value is NaN.")
            else:
                print("The variable is a float.")
        elif isinstance(doi, np.float) and np.isnan(doi):
            print("The variable is a float (np.float) and its value is NaN.")
        else:
            print("The variable is neither a list nor a float.")


    # Save bibliography
    with open("bibliography.bib", "w") as file:
        for url in urls[:10]:  # Only doing the first 10 here to save time on the demo
            response = requests.get(url, headers={"accept": "application/x-bibtex"})
            file.write(response.text)

    # Query Crossref for citation data
    crossref_citation_data = CrossrefCitationData()
    cites_data = [crossref_citation_data.fetch_data(doi) for doi in dois if doi]
    cites_data = list(filter(None, cites_data))
    cites_df = pd.DataFrame(cites_data)
    cites_df = cites_df.rename(columns={"count": "citations_count"})

    # Retrieve Altmetric data
    alt_df = [get_altmetrics(doi) for doi in dois]
    alt_df = pd.DataFrame(alt_df)  # Convert to DataFrame
    alt_df = alt_df.drop(
        columns=[col for col in alt_df.columns if col.startswith("authors")])  # Now drop function can be used

    # Join dataframes
    df = pd.concat([df, alt_df], axis=1)

    # Impute the title
    df["TI"] = df["TI"].fillna('')
    df["title"] = df["TI"].apply(lambda x: x.capitalize())

    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # Summary of top Altmetric papers
    top_altmetric = df.nlargest(10, "score")[["title", "JT", "AID", "AU", "score"]]

    print(top_altmetric)


if __name__ == "__main__":
    main()
