# -*- coding: utf-8 -*-
#
# Standalone check: can we get 5 usable LinkedIn job pages?
# A page counts as usable when it downloads (HTTP 200) and contains the
# description block. Run: venv/bin/python test_linkedin.py
#

import time

import requests
from ddgs import DDGS

from job_sources import LINKEDIN_HEADERS, LINKEDIN_QUERY, clean_linkedin_html, page_text

WANTED_PAGES = 5
SEARCH_ATTEMPTS = 4
DELAY_SECONDS = 3


def search_urls() -> list[str]:
    urls = []
    for attempt in range(SEARCH_ATTEMPTS):
        try:
            results = DDGS().text(LINKEDIN_QUERY, max_results=10, backend="auto")
        except Exception as error:
            print(f"search attempt {attempt + 1}: {error}")
            results = []
        for result in results:
            if result["href"] not in urls:
                urls.append(result["href"])
        print(f"search attempt {attempt + 1}: {len(urls)} distinct urls so far")
        if len(urls) >= 10:
            break
        time.sleep(DELAY_SECONDS)
    return urls


def main() -> None:
    urls = search_urls()
    good_pages = 0
    for url in urls:
        response = requests.get(url, headers=LINKEDIN_HEADERS, timeout=30)
        description = clean_linkedin_html(response.text) if response.status_code == 200 else ""
        title = page_text(response.text, "top-card-layout__title") if description else ""
        status = "OK " if description else "BAD"
        print(f"{status} {response.status_code} {title[:40]:40} {len(description):5} chars  {url[:70]}")
        if description:
            good_pages += 1
        if good_pages >= WANTED_PAGES:
            break
        time.sleep(DELAY_SECONDS)
    print(f"\n{good_pages}/{WANTED_PAGES} usable pages ->", "PASS" if good_pages >= WANTED_PAGES else "FAIL")


if __name__ == "__main__":
    main()
