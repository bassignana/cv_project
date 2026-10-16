# -*- coding: utf-8 -*-
#
# Job sources for cv_project.py. Each source downloads its listings and
# returns them in one common shape, so the rest of the pipeline never sees
# a backend's own field names:
#
#   title, company, location, url   strings
#   description                     plain text (HTML already removed)
#   tags                            list of strings
#   published                       ISO date string (sorts by date as text)
#   source                          display name, for crediting the backend
#
# To add a backend: write a fetch_<name>_jobs() that returns that shape and
# register it in SOURCES.
#

import html
import json
import re
import time
from pathlib import Path

import requests

CACHE_MAX_AGE_HOURS = 12
LINKEDIN_QUERY = 'site:linkedin.com/jobs/view "data analyst" "Italy"'
LINKEDIN_MAX_RESULTS = 10
LINKEDIN_PAGES_WANTED = 5
LINKEDIN_PAGE_DELAY_SECONDS = 3
LINKEDIN_SEARCH_ATTEMPTS = 4
LINKEDIN_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                    "Accept-Language": "en-US,en;q=0.9"}
NEWLINE_TAG_NAMES = ("br", "/p", "/li", "/div", "/h1", "/h2", "/h3", "/h4", "/h5", "/h6")


def clean_remotive_html(raw: str) -> str:
    #
    # Tags are found with str.find: line-break tags become newlines, <li>
    # becomes "- " and every other tag becomes a space. A "<" with no closing
    # ">" is kept as plain text.
    #
    parts = []
    position = 0
    while position < len(raw):
        tag_start = raw.find("<", position)
        tag_end = raw.find(">", tag_start)
        if tag_start == -1 or tag_end == -1:
            parts.append(raw[position:])
            break
        parts.append(raw[position:tag_start])
        tag_words = raw[tag_start + 1:tag_end].lower().split()
        tag_name = tag_words[0].rstrip("/") if tag_words else ""
        if tag_name in NEWLINE_TAG_NAMES:
            parts.append("\n")
        elif tag_name == "li":
            parts.append("- ")
        else:
            parts.append(" ")
        position = tag_end + 1
    lines = []
    for line in html.unescape("".join(parts)).split("\n"):
        collapsed_line = " ".join(line.split())  # also folds tabs and non-breaking spaces
        if collapsed_line:
            lines.append(collapsed_line)
    return "\n".join(lines)


def cached_download(source_name: str, download) -> list[dict]:
    #
    # Caches a source's raw listings in <source_name>_cache.json, so reruns
    # don't hit its API. download() is only called when the cache is stale.
    #
    cache_path = Path(f"{source_name}_cache.json")
    fresh = (cache_path.exists()
             and time.time() - cache_path.stat().st_mtime < CACHE_MAX_AGE_HOURS * 3600)
    if fresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    raw_jobs = download()
    if not raw_jobs:  # don't keep an empty answer for 12 hours
        return raw_jobs
    cache_path.write_text(json.dumps(raw_jobs), encoding="utf-8")
    return raw_jobs

# ---------------------------------------------------------------------
# REMOTIVE - public API, no key (https://remotive.com/api/remote-jobs)
#   Remotive's terms: if you show listings publicly, link back to Remotive
#   and credit it as the source. Listings are ~24h delayed.
# ---------------------------------------------------------------------
def download_remotive() -> list[dict]:
    response = requests.get("https://remotive.com/api/remote-jobs", timeout=30)
    response.raise_for_status()
    return response.json().get("jobs", [])


def normalize_remotive(raw_job: dict) -> dict:
    return {
        "title": raw_job.get("title", ""),
        "company": raw_job.get("company_name", ""),
        "location": raw_job.get("candidate_required_location", ""),
        "url": raw_job.get("url", ""),
        "description": clean_remotive_html(raw_job.get("description", "")),
        "tags": raw_job.get("tags") or [],
        "published": raw_job.get("publication_date", ""),
        "source": "Remotive",
    }


def fetch_remotive_jobs() -> list[dict]:
    return [normalize_remotive(raw_job) for raw_job in cached_download("remotive", download_remotive)]

# ---------------------------------------------------------------------
# LINKEDIN - no API: a web search (ddgs) finds public job pages, which are
#   then downloaded one by one until LINKEDIN_PAGES_WANTED usable pages are
#   collected (search results come from LINKEDIN_QUERY). Public pages only, no login; listings link
#   back to LinkedIn.
# ---------------------------------------------------------------------
def clean_linkedin_html(raw_page: str) -> str:
    #
    # Keeps only the description block of a LinkedIn job page. <br> and block
    # ends become newlines, <li> becomes "- ", other tags are dropped.
    #
    block_start = raw_page.find("show-more-less-html__markup")
    if block_start == -1:
        return ""
    block_start = raw_page.find(">", block_start) + 1
    block_end = raw_page.find("</div>", block_start)
    block = raw_page[block_start:block_end]
    block = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</ul>|</ol>", "\n", block)
    block = re.sub(r"(?i)<li[^>]*>", "- ", block)
    block = re.sub(r"<[^>]+>", "", block)
    lines = [" ".join(line.split()) for line in html.unescape(block).split("\n")]
    return "\n".join(line for line in lines if line)


def page_text(raw_page: str, marker: str) -> str:
    #
    # Text of the first tag whose attributes contain marker, e.g. a class name.
    #
    marker_at = raw_page.find(marker)
    if marker_at == -1:
        return ""
    text_start = raw_page.find(">", marker_at) + 1
    text_end = raw_page.find("</", text_start)
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", raw_page[text_start:text_end])).split())


def search_linkedin_urls() -> list[str]:
    #
    # backend="auto" on purpose: ddgs's duckduckgo and brave backends alone
    # often answer "No results" for this dork, "auto" tries them all. The
    # search is flaky (few or no results), so it is retried and the distinct
    # urls of all attempts are collected.
    #
    from ddgs import DDGS
    urls = []
    for attempt in range(LINKEDIN_SEARCH_ATTEMPTS):
        try:
            results = DDGS().text(LINKEDIN_QUERY, max_results=LINKEDIN_MAX_RESULTS, backend="auto")
        except Exception:  # ddgs raises when a search finds nothing
            results = []
        for result in results:
            if result["href"] not in urls:
                urls.append(result["href"])
        if len(urls) >= LINKEDIN_MAX_RESULTS:
            break
        time.sleep(LINKEDIN_PAGE_DELAY_SECONDS * 2)
    return urls


def download_linkedin() -> list[dict]:
    #
    # Goes through the search results until LINKEDIN_PAGES_WANTED pages have
    # downloaded with a description block. Pages that answer 429 or come back
    # without a description are skipped.
    #
    raw_jobs = []
    for url in search_linkedin_urls():
        response = requests.get(url, headers=LINKEDIN_HEADERS, timeout=30)
        if response.status_code == 200 and "show-more-less-html__markup" in response.text:
            raw_jobs.append({"url": url, "page": response.text, "fetched": time.strftime("%Y-%m-%d")})
        if len(raw_jobs) >= LINKEDIN_PAGES_WANTED:
            break
        time.sleep(LINKEDIN_PAGE_DELAY_SECONDS)
    return raw_jobs


def normalize_linkedin(raw_job: dict) -> dict:
    page = raw_job["page"]
    return {
        "title": page_text(page, "top-card-layout__title"),
        "company": page_text(page, "topcard__org-name-link"),
        "location": page_text(page, "topcard__flavor topcard__flavor--bullet"),
        "url": raw_job["url"],
        "description": clean_linkedin_html(page),
        "tags": [],
        "published": raw_job["fetched"],  # the page only says "2 days ago"
        "source": "LinkedIn",
    }


def fetch_linkedin_jobs() -> list[dict]:
    return [normalize_linkedin(raw_job) for raw_job in cached_download("linkedin", download_linkedin)]

# ---------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------
SOURCES = {
    "remotive": fetch_remotive_jobs,
    "linkedin": fetch_linkedin_jobs,
}


def fetch_jobs(source_names: list[str] = ("linkedin",)) -> list[dict]:
    jobs = []
    for source_name in source_names:
        jobs.extend(SOURCES[source_name]())
    return jobs
