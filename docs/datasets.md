# Datasets Overview

Weekly updates are performed from the XDD ElasticSearch Service to our Weaviate document store.

## Data Selection

We process specific ElasticSearch tags:

- "xdd-covid-19": COVID-19 related data
- "climate-change-modeling": Climate change modeling data
- "dolomites": Dolomites data
- "criticalmaas": Data for criticalMAAS project (This is a super set of "geoarchive")
- "geoarchive": GeoArchive data

Refer to `SET_NAMES` in [askem.elastic](../askem/elastic.py) for the latest list.

## Data Cleaning

Prior to storage in Weaviate, data undergoes sanitization for consistency, covering:

- `DocType` (e.g., "paragraph", "figure", "table", etc.)
- `Topic`, mirroring the name from the original data source (e.g., xdd-covid-19, criticalmaas, etc.)

See [data_models](../askem/retriever/data_models.py) for details.

## Update Schedule

Ingestion occurs every Friday at 7pm, executed via the [ingest.sh](../scripts/ingest.sh) script.
