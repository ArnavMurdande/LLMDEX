# Contributing

We welcome contributions!

1. Fork the repository.
2. Create a feature branch.
3. Submit a Pull Request.

## Adding a New Scraper

1. Create `scraper/scrape_newsource.py`.
2. Ensure it returns a list of dictionaries matching the schema.
3. Import it in `pipeline/run_pipeline.py`.
