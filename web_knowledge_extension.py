# web_knowledge_extension.py
import asyncio
import aiohttp
import time
import json
import re
import logging
import random
import hashlib
import os
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Union, Any
import numpy as np
import torch
import torch.nn.functional as F

# Compatibility with SAM's consciousness system
from pathlib import Path
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SAM.WebKnowledge")


@dataclass
class WebCrawlConfig:
    """Configuration for SAM's Web Knowledge Acquisition"""
    # Core settings
    max_pages_per_session: int = 50
    max_pages_per_domain: int = 5
    crawl_delay: float = 2.0  # Seconds between requests to same domain
    request_timeout: int = 30
    max_depth: int = 3
    follow_redirects: bool = True

    # Knowledge integration settings
    concepts_per_page: int = 50
    min_concept_frequency: int = 2
    knowledge_decay_time: int = 7 * 24 * 3600  # 7 days in seconds

    # Crawler behavior
    user_agent: str = "SAM-WebKnowledge/1.0 (github.com/your-repo/SAM; Synergistic Autonomous Machine)"
    respect_robots_txt: bool = True
    respect_nofollow: bool = True
    verify_ssl: bool = True

    # Storage
    knowledge_db_path: str = "web_knowledge.db"
    cache_dir: str = "web_cache"

    # Session management
    session_timeout: int = 3600  # 1 hour
    max_concurrent_requests: int = 5

    # Priority domains and topics
    priority_domains: List[str] = field(default_factory=list)
    priority_topics: List[str] = field(default_factory=list)

    # Excluded domains and patterns
    excluded_domains: List[str] = field(default_factory=lambda: [
        "facebook.com", "twitter.com", "instagram.com", "youtube.com",
        "pinterest.com", "reddit.com", "tiktok.com", "linkedin.com"
    ])
    excluded_patterns: List[str] = field(default_factory=lambda: [
        r"\.pdf$", r"\.zip$", r"\.rar$", r"\.exe$", r"\.dmg$", r"\.apk$",
        r"login", r"signin", r"signup", r"register", r"checkout", r"cart"
    ])


class WebKnowledgeExtension:
    """Web Knowledge Acquisition System for SAM"""

    def __init__(self, sam_model, config: Optional[WebCrawlConfig] = None):
        """Initialize the Web Knowledge Extension for SAM

        Args:
            sam_model: The SAM model instance to extend
            config: Optional configuration override
        """
        self.sam = sam_model
        self.config = config or WebCrawlConfig()

        # Initialize storage
        os.makedirs(self.config.cache_dir, exist_ok=True)

        # Internal state
        self.active_crawl_task = None
        self.crawl_queue = asyncio.Queue()
        self.visited_urls = set()
        self.domain_last_visit = {}
        self.domain_visit_count = {}
        self.current_depth_map = {}
        self.content_hashes = set()

        # Knowledge statistics
        self.knowledge_stats = {
            "pages_processed": 0,
            "concepts_added": 0,
            "domains_visited": set(),
            "last_crawl_time": None,
            "total_crawl_sessions": 0,
            "total_content_bytes": 0
        }

        # Robots.txt cache
        self.robots_cache = {}

        # Setup page processors
        self.page_processors = [
            TextContentProcessor(self),
            SemanticStructureProcessor(self),
            CodeExtractionProcessor(self),
            DataTableProcessor(self)
        ]

        # Task scheduler
        self.scheduler = WebKnowledgeScheduler(self)

        # Initialize connection to SAM systems
        self._initialize_sam_integration()

        logger.info("Web Knowledge Extension initialized")

    def _initialize_sam_integration(self):
        """Initialize integration with SAM's core systems"""
        if not hasattr(self.sam, 'concept_bank'):
            logger.error("SAM model lacks required concept_bank component")
            raise ValueError("SAM model must have a concept_bank component")

        # Ensure we have an experience manager
        if hasattr(self.sam, 'experience_manager'):
            self.experience_manager = self.sam.experience_manager
        else:
            logger.warning("Creating new ExperienceManager for web knowledge")
            from sam import ExperienceManager
            self.experience_manager = ExperienceManager(self.sam.config)
            self.sam.experience_manager = self.experience_manager

        # Register knowledge extension with SAM
        if hasattr(self.sam, 'extensions'):
            self.sam.extensions['web_knowledge'] = self
        else:
            self.sam.extensions = {'web_knowledge': self}

        # Register concepts for web knowledge domain
        self._register_web_knowledge_concepts()

    def _register_web_knowledge_concepts(self):
        """Register fundamental concepts for web knowledge domain"""
        web_concepts = [
            "website", "webpage", "url", "hyperlink", "crawl", "domain",
            "http", "https", "html", "web", "internet", "browser", "download",
            "scrape", "extract", "data", "information", "knowledge", "learn",
            "update", "fetch", "query", "search", "discover", "explore"
        ]

        # Add concepts to SAM's concept bank
        concept_bank = self.sam.concept_bank
        for concept in web_concepts:
            if not concept_bank.find_concept_by_source(concept):
                concept_bank.add_character_concept(concept, modality="text")
                logger.debug(f"Added web concept: {concept}")

    async def start(self):
        """Start the Web Knowledge Extension services"""
        # Start scheduler
        await self.scheduler.start()
        logger.info("Web Knowledge Extension started")

    async def stop(self):
        """Stop the Web Knowledge Extension services"""
        await self.scheduler.stop()
        if self.active_crawl_task and not self.active_crawl_task.done():
            self.active_crawl_task.cancel()
            try:
                await self.active_crawl_task
            except asyncio.CancelledError:
                pass
        logger.info("Web Knowledge Extension stopped")

    async def crawl(self, seed_urls=None, topics=None, depth=None, max_pages=None):
        """Start a crawl session with the given parameters

        Args:
            seed_urls: List of URLs to start crawling from
            topics: List of topics to focus on
            depth: Maximum crawl depth
            max_pages: Maximum pages to crawl

        Returns:
            A dictionary with crawl statistics
        """
        # Cancel any active crawl
        if self.active_crawl_task and not self.active_crawl_task.done():
            logger.info("Cancelling active crawl to start new one")
            self.active_crawl_task.cancel()
            try:
                await self.active_crawl_task
            except asyncio.CancelledError:
                pass

        # Use default seed URLs if none provided
        if seed_urls is None:
            # Default to popular news and knowledge sites
            seed_urls = [
                "https://en.wikipedia.org/wiki/Main_Page",
                "https://news.ycombinator.com/",
                "https://arxiv.org/",
                "https://www.nature.com/",
                "https://www.sciencedaily.com/",
                "https://techcrunch.com/",
                "https://www.technologyreview.com/"
            ]

            # Add domain-specific seeds based on topics
            if topics:
                topic_seeds = self._get_topic_seeds(topics)
                seed_urls.extend(topic_seeds)

        # Set crawl parameters
        effective_depth = depth if depth is not None else self.config.max_depth
        effective_max_pages = max_pages if max_pages is not None else self.config.max_pages_per_session

        # Initialize crawl state
        self.visited_urls = set()
        self.domain_visit_count = {}
        self.current_depth_map = {}

        # Queue initial URLs
        for url in seed_urls:
            await self.crawl_queue.put((url, 0))  # (url, depth)
            self.current_depth_map[url] = 0

        # Start crawl task
        self.knowledge_stats["last_crawl_time"] = time.time()
        self.knowledge_stats["total_crawl_sessions"] += 1

        self.active_crawl_task = asyncio.create_task(
            self._crawl_worker(effective_depth, effective_max_pages, topics)
        )

        # Return reference to the task
        return self.active_crawl_task

    async def _crawl_worker(self, max_depth, max_pages, topics=None):
        """Worker process for crawling pages

        Args:
            max_depth: Maximum crawl depth
            max_pages: Maximum pages to crawl
            topics: Optional list of topics to focus on
        """
        logger.info(f"Starting crawl: max_depth={max_depth}, max_pages={max_pages}")
        pages_crawled = 0
        start_time = time.time()

        # Create a session for all requests
        async with aiohttp.ClientSession(
            headers={"User-Agent": self.config.user_agent},
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
        ) as session:
            # Process queue until empty or limits reached
            try:
                semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
                tasks = []

                while not self.crawl_queue.empty() and pages_crawled < max_pages:
                    url, depth = await self.crawl_queue.get()

                    # Skip if already visited or if depth exceeds max
                    if url in self.visited_urls or depth > max_depth:
                        self.crawl_queue.task_done()
                        continue

                    # Skip if we've reached the per-domain limit
                    domain = self._get_domain(url)
                    if self.domain_visit_count.get(domain, 0) >= self.config.max_pages_per_domain:
                        self.crawl_queue.task_done()
                        continue

                    # Add to visited set
                    self.visited_urls.add(url)

                    # Process with rate limiting
                    task = asyncio.create_task(
                        self._rate_limited_process_url(
                            semaphore, session, url, depth, topics
                        )
                    )
                    tasks.append(task)
                    pages_crawled += 1

                    # Wait for some tasks to complete if we have too many
                    if len(tasks) >= self.config.max_concurrent_requests * 2:
                        completed, tasks = await asyncio.wait(
                            tasks, return_when=asyncio.FIRST_COMPLETED
                        )

                # Wait for remaining tasks
                if tasks:
                    await asyncio.gather(*tasks)

            except asyncio.CancelledError:
                logger.info("Crawl task cancelled")
                raise
            except Exception as e:
                logger.error(f"Error in crawl worker: {e}")
                logger.error(traceback.format_exc())

        duration = time.time() - start_time
        logger.info(f"Crawl completed: {pages_crawled} pages in {duration:.2f}s")

        # Update statistics
        self.knowledge_stats["pages_processed"] += pages_crawled

        # Trigger knowledge integration with SAM
        await self._integrate_crawled_knowledge()

        return {
            "pages_crawled": pages_crawled,
            "duration_seconds": duration,
            "domains_visited": len(self.domain_visit_count),
            "concepts_added": self.knowledge_stats["concepts_added"]
        }

    async def _rate_limited_process_url(self, semaphore, session, url, depth, topics):
        """Process URL with rate limiting for each domain

        Args:
            semaphore: Semaphore for limiting concurrent requests
            session: HTTP session
            url: URL to process
            depth: Current crawl depth
            topics: Optional list of topics to focus on
        """
        domain = self._get_domain(url)

        # Apply domain-specific rate limiting
        last_visit = self.domain_last_visit.get(domain, 0)
        time_since_last = time.time() - last_visit
        if time_since_last < self.config.crawl_delay:
            await asyncio.sleep(self.config.crawl_delay - time_since_last)

        # Mark this domain as visited now
        self.domain_last_visit[domain] = time.time()
        self.domain_visit_count[domain] = self.domain_visit_count.get(domain, 0) + 1
        self.knowledge_stats["domains_visited"].add(domain)

        # Process the URL with concurrency limit
        async with semaphore:
            try:
                # Check robots.txt
                if self.config.respect_robots_txt and not await self._check_robots_permission(session, url):
                    logger.debug(f"Robots.txt disallows: {url}")
                    return

                # Fetch and process the page
                await self._fetch_and_process_page(session, url, depth, topics)

            except Exception as e:
                logger.warning(f"Error processing {url}: {e}")

    async def _check_robots_permission(self, session, url):
        """Check if robots.txt allows crawling this URL

        Args:
            session: HTTP session
            url: URL to check

        Returns:
            True if crawling is allowed, False otherwise
        """
        if not self.config.respect_robots_txt:
            return True

        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        # Check cache first
        if domain in self.robots_cache:
            robots_rules = self.robots_cache[domain]
            # Use urllib.robotparser to check if url is allowed
            return robots_rules.can_fetch(self.config.user_agent, url)

        # Fetch robots.txt
        robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
        try:
            async with session.get(robots_url, timeout=5) as response:
                if response.status == 200:
                    robots_content = await response.text()

                    # Parse robots.txt content
                    from urllib.robotparser import RobotFileParser
                    robots = RobotFileParser()
                    # RobotFileParser needs a file-like interface, so we use StringIO
                    from io import StringIO
                    robots.parse(StringIO(robots_content).readlines())

                    # Cache the rules
                    self.robots_cache[domain] = robots

                    # Check permission
                    return robots.can_fetch(self.config.user_agent, url)
                else:
                    # If robots.txt doesn't exist or can't be fetched, allow access
                    self.robots_cache[domain] = RobotFileParser()  # Empty rules
                    return True
        except Exception as e:
            logger.warning(f"Error fetching robots.txt for {domain}: {e}")
            return True  # Default to allowing access on error

    async def _fetch_and_process_page(self, session, url, depth, topics):
        """Fetch and process a single web page

        Args:
            session: HTTP session
            url: URL to fetch
            depth: Current crawl depth
            topics: Optional list of topics to focus on
        """
        logger.debug(f"Fetching {url} (depth {depth})")

        try:
            async with session.get(
                url,
                allow_redirects=self.config.follow_redirects,
                ssl=self.config.verify_ssl
            ) as response:
                if response.status != 200:
                    logger.debug(f"Non-200 status for {url}: {response.status}")
                    return

                content_type = response.headers.get('Content-Type', '').lower()

                # Only process HTML content
                if 'text/html' not in content_type:
                    logger.debug(f"Skipping non-HTML content: {url} ({content_type})")
                    return

                # Get page content
                html_content = await response.text()

                # Check content hash to avoid processing duplicate content
                content_hash = hashlib.sha256(html_content.encode()).hexdigest()
                if content_hash in self.content_hashes:
                    logger.debug(f"Skipping duplicate content: {url}")
                    return

                self.content_hashes.add(content_hash)
                self.knowledge_stats["total_content_bytes"] += len(html_content)

                # Process the page
                await self._process_page_content(url, html_content, depth, topics)

        except asyncio.TimeoutError:
            logger.debug(f"Timeout fetching {url}")
        except Exception as e:
            logger.warning(f"Error fetching {url}: {e}")

    async def _process_page_content(self, url, html_content, depth, topics):
        """Process page content to extract knowledge and links

        Args:
            url: URL of the page
            html_content: HTML content of the page
            depth: Current crawl depth
            topics: Optional list of topics to focus on
        """
        try:
            # Parse the HTML
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract page title and meta description
            title = soup.title.string if soup.title else ""
            meta_desc = ""
            meta_tag = soup.find("meta", attrs={"name": "description"})
            if meta_tag and "content" in meta_tag.attrs:
                meta_desc = meta_tag["content"]

            # Create page context
            page_context = {
                "url": url,
                "title": title,
                "description": meta_desc,
                "crawl_time": time.time(),
                "depth": depth,
                "domain": self._get_domain(url)
            }

            # Run all page processors
            extracted_knowledge = []
            for processor in self.page_processors:
                try:
                    processor_results = await processor.process(soup, page_context)
                    if processor_results:
                        extracted_knowledge.extend(processor_results)
                except Exception as e:
                    logger.warning(f"Error in {processor.__class__.__name__}: {e}")

            # Store extracted knowledge
            if extracted_knowledge:
                await self._store_extracted_knowledge(extracted_knowledge, page_context)

            # Extract links for further crawling if depth allows
            if depth < self.config.max_depth:
                await self._extract_and_queue_links(soup, url, depth)

        except Exception as e:
            logger.error(f"Error processing content for {url}: {e}")

    async def _extract_and_queue_links(self, soup, base_url, current_depth):
        """Extract links from the page and queue them for crawling

        Args:
            soup: BeautifulSoup object for the page
            base_url: URL of the current page
            current_depth: Current crawl depth
        """
        next_depth = current_depth + 1

        # Find all links
        for link in soup.find_all('a', href=True):
            href = link['href']

            # Skip if nofollow and we respect it
            if self.config.respect_nofollow:
                if link.get('rel') and 'nofollow' in link.get('rel'):
                    continue

            # Resolve relative URLs
            full_url = urljoin(base_url, href)

            # Validate URL (must be http or https)
            parsed = urlparse(full_url)
            if parsed.scheme not in ('http', 'https'):
                continue

            # Skip fragment identifiers on the same page
            if parsed.fragment and parsed.path == urlparse(base_url).path:
                continue

            # Skip excluded domains
            domain = parsed.netloc
            if any(excluded in domain for excluded in self.config.excluded_domains):
                continue

            # Skip excluded patterns
            if any(re.search(pattern, full_url) for pattern in self.config.excluded_patterns):
                continue

            # Check if already visited or queued
            if full_url in self.visited_urls or full_url in self.current_depth_map:
                continue

            # Queue this URL for crawling
            await self.crawl_queue.put((full_url, next_depth))
            self.current_depth_map[full_url] = next_depth

    async def _store_extracted_knowledge(self, knowledge_items, page_context):
        """Store extracted knowledge for later integration

        Args:
            knowledge_items: List of extracted knowledge items
            page_context: Context information about the source page
        """
        # Create a knowledge entry in the page cache
        cache_path = os.path.join(
            self.config.cache_dir,
            f"page_{hashlib.md5(page_context['url'].encode()).hexdigest()}.json"
        )

        # Store the knowledge data
        with open(cache_path, 'w') as f:
            json.dump({
                "context": page_context,
                "knowledge": knowledge_items,
                "timestamp": time.time()
            }, f)

    async def _integrate_crawled_knowledge(self):
        """Integrate all crawled knowledge into SAM's systems"""
        logger.info("Integrating crawled knowledge into SAM")

        knowledge_files = [f for f in os.listdir(self.config.cache_dir)
                         if f.startswith("page_") and f.endswith(".json")]

        # Process files by creation time (oldest first)
        knowledge_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.config.cache_dir, f)))

        concepts_added = 0
        experiences_added = 0

        for filename in knowledge_files:
            file_path = os.path.join(self.config.cache_dir, filename)

            # Skip files older than knowledge decay time
            if time.time() - os.path.getmtime(file_path) > self.config.knowledge_decay_time:
                os.remove(file_path)
                continue

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                context = data["context"]
                knowledge_items = data["knowledge"]

                # Integrate each knowledge item
                for item in knowledge_items:
                    item_type = item.get("type", "unknown")

                    if item_type == "concept":
                        # Add to concept bank
                        added = await self._integrate_concept(item, context)
                        concepts_added += added

                    elif item_type == "experience":
                        # Add to experience manager
                        added = await self._integrate_experience(item, context)
                        experiences_added += added

                # Delete processed file
                os.remove(file_path)

            except Exception as e:
                logger.error(f"Error integrating knowledge from {filename}: {e}")

        # Update knowledge statistics
        self.knowledge_stats["concepts_added"] += concepts_added

        logger.info(f"Knowledge integration complete: {concepts_added} concepts, {experiences_added} experiences")

        # Trigger SAM's dreaming cycle to assimilate new knowledge
        if hasattr(self.sam, 'conceptual_dreaming'):
            dream_task = asyncio.create_task(self._trigger_sam_dreaming())

    async def _integrate_concept(self, concept_item, context):
        """Integrate a concept into SAM's concept bank

        Args:
            concept_item: Concept item to integrate
            context: Source context information

        Returns:
            Number of concepts added (0 or 1)
        """
        concept_text = concept_item.get("text", "")
        if not concept_text or len(concept_text) < 3:
            return 0

        # Skip if concept already exists
        if self.sam.concept_bank.find_concept_by_source(concept_text):
            # Update concept usage
            concept_id = self.sam.concept_bank.find_concept_by_source(concept_text)
            self.sam.concept_bank.update_concept_usage(concept_id, context=f"web:{context['domain']}")
            return 0

        # Add to concept bank
        modality = concept_item.get("modality", "text")
        importance = concept_item.get("importance", 1.0)

        concept_id = self.sam.concept_bank.add_character_concept(
            concept_text,
            modality=modality,
            importance=importance
        )

        # If concept has a meaning vector, update it
        if "meaning_vector" in concept_item and hasattr(self.sam.concept_bank, "meaning_vectors"):
            meaning_vector = torch.tensor(
                concept_item["meaning_vector"],
                device=self.sam.concept_bank.meaning_vectors.device
            )
            with torch.no_grad():
                self.sam.concept_bank.meaning_vectors[concept_id] = F.normalize(meaning_vector, dim=0)

        return 1

    async def _integrate_experience(self, experience_item, context):
        """Integrate an experience into SAM's experience manager

        Args:
            experience_item: Experience item to integrate
            context: Source context information

        Returns:
            Number of experiences added (0 or 1)
        """
        experience_type = experience_item.get("experience_type", "web_knowledge")
        content = experience_item.get("content", "")

        if not content:
            return 0

        # Create metadata
        metadata = {
            "source_url": context["url"],
            "source_domain": context["domain"],
            "crawl_time": context["crawl_time"],
            "title": context.get("title", ""),
        }

        # Add any additional metadata from the experience item
        if "metadata" in experience_item:
            metadata.update(experience_item["metadata"])

        # Record the experience
        experience_id = self.experience_manager.record_experience(
            experience_type=experience_type,
            content=content,
            metadata=metadata,
            private=False,
            modality=experience_item.get("modality", "text")
        )

        return 1

    async def _trigger_sam_dreaming(self):
        """Trigger SAM's dreaming cycle to assimilate new knowledge"""
        if hasattr(self.sam, 'conceptual_dreaming'):
            logger.info("Triggering SAM dream cycle to assimilate web knowledge")
            try:
                dream_result = await self.sam.conceptual_dreaming.dream_cycle(duration_minutes=1)
                logger.info(f"Dream cycle completed: {dream_result}")
            except Exception as e:
                logger.error(f"Error in dream cycle: {e}")

    def _get_domain(self, url):
        """Extract domain from URL"""
        return urlparse(url).netloc

    def _get_topic_seeds(self, topics):
        """Get seed URLs for specific topics

        Args:
            topics: List of topics to find seeds for

        Returns:
            List of seed URLs for the specified topics
        """
        topic_seeds = []

        # Topic-specific seed URLs
        topic_url_map = {
            "technology": [
                "https://www.wired.com/",
                "https://techcrunch.com/",
                "https://arstechnica.com/",
                "https://www.theverge.com/"
            ],
            "science": [
                "https://www.sciencemag.org/",
                "https://www.sciencedaily.com/",
                "https://phys.org/",
                "https://www.scientificamerican.com/"
            ],
            "medicine": [
                "https://www.nejm.org/",
                "https://www.medicalnewstoday.com/",
                "https://www.mayoclinic.org/",
                "https://www.nih.gov/"
            ],
            "finance": [
                "https://www.bloomberg.com/",
                "https://www.ft.com/",
                "https://www.wsj.com/",
                "https://www.investopedia.com/"
            ],
            "programming": [
                "https://stackoverflow.com/",
                "https://dev.to/",
                "https://github.com/topics/",
                "https://www.geeksforgeeks.org/"
            ],
            "ai": [
                "https://arxiv.org/list/cs.AI/recent",
                "https://distill.pub/",
                "https://ai.googleblog.com/",
                "https://openai.com/blog/"
            ]
        }

        # Add URLs for each topic
        for topic in topics:
            if topic.lower() in topic_url_map:
                topic_seeds.extend(topic_url_map[topic.lower()])

        return topic_seeds

    def get_knowledge_stats(self):
        """Get statistics about the web knowledge acquisition

        Returns:
            Dictionary of statistics
        """
        stats = self.knowledge_stats.copy()
        stats["domains_visited"] = len(stats["domains_visited"])
        return stats


class WebKnowledgeScheduler:
    """Scheduler for periodic web knowledge acquisition"""

    def __init__(self, extension):
        """Initialize the scheduler

        Args:
            extension: WebKnowledgeExtension instance
        """
        self.extension = extension
        self.running = False
        self.task = None

        # Default schedule: once per day at a random time
        self.schedule = {
            "enabled": True,
            "interval_hours": 24,
            "jitter_minutes": 120,  # 2 hour random jitter
            "last_run": 0,
            "topics_rotation": [
                ["technology", "science"],
                ["programming", "ai"],
                ["medicine", "finance"]
            ],
            "topic_index": 0
        }

    async def start(self):
        """Start the scheduler"""
        if self.running:
            return

        self.running = True
        self.task = asyncio.create_task(self._scheduler_loop())

    async def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                now = time.time()

                # Check if it's time to run
                if self.schedule["enabled"] and now - self.schedule["last_run"] >= self.schedule["interval_hours"] * 3600:
                    # Add jitter to prevent all SAM instances from crawling at the same time
                    jitter = random.randint(0, self.schedule["jitter_minutes"] * 60)
                    await asyncio.sleep(jitter)

                    # Run the crawl
                    await self._scheduled_crawl()

                    # Update last run time
                    self.schedule["last_run"] = time.time()

                    # Rotate topics
                    self.schedule["topic_index"] = (self.schedule["topic_index"] + 1) % len(self.schedule["topics_rotation"])

                # Sleep for a while before checking again (1 hour)
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(3600)  # Sleep and retry on error

    async def _scheduled_crawl(self):
        """Run a scheduled crawl"""
        # Get current topics
        topics = self.schedule["topics_rotation"][self.schedule["topic_index"]]

        logger.info(f"Running scheduled crawl with topics: {topics}")

        try:
            # Run the crawl
            await self.extension.crawl(topics=topics)
        except Exception as e:
            logger.error(f"Error in scheduled crawl: {e}")


class BasePageProcessor:
    """Base class for page content processors"""

    def __init__(self, extension):
        """Initialize the processor

        Args:
            extension: WebKnowledgeExtension instance
        """
        self.extension = extension

    async def process(self, soup, context):
        """Process page content

        Args:
            soup: BeautifulSoup object for the page
            context: Context information about the page

        Returns:
            List of extracted knowledge items
        """
        raise NotImplementedError("Subclasses must implement process()")


class TextContentProcessor(BasePageProcessor):
    """Process textual content from web pages"""

    async def process(self, soup, context):
        """Extract text content and concepts

        Args:
            soup: BeautifulSoup object for the page
            context: Context information about the page

        Returns:
            List of extracted knowledge items
        """
        knowledge_items = []

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract()

        # Extract main content
        main_content = ""
        main_elements = soup.find_all(["article", "main", "div"], class_=["content", "main", "article", "post"])

        if main_elements:
            for element in main_elements:
                main_content += element.get_text(separator=" ", strip=True) + " "
        else:
            # Fallback to body content
            main_content = soup.body.get_text(separator=" ", strip=True) if soup.body else ""

        # Clean text
        main_content = re.sub(r'\s+', ' ', main_content).strip()

        if not main_content:
            return []

        # Record the entire content as an experience
        knowledge_items.append({
            "type": "experience",
            "experience_type": "web_content",
            "content": main_content[:10000],  # Limit to 10K chars
            "modality": "text",
            "metadata": {
                "domain": context["domain"],
                "title": context["title"],
                "url": context["url"]
            }
        })

        # Extract important paragraphs
        paragraphs = soup.find_all("p")
        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 100:  # Meaningful paragraphs only
                knowledge_items.append({
                    "type": "experience",
                    "experience_type": "web_paragraph",
                    "content": text,
                    "modality": "text"
                })

        # Extract concepts (phrases and terms)
        extracted_concepts = self._extract_concepts(main_content)
        knowledge_items.extend(extracted_concepts)

        return knowledge_items

    def _extract_concepts(self, text):
        """Extract concepts from text

        Args:
            text: Text to extract concepts from

        Returns:
            List of concept items
        """
        concepts = []

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Extract phrases using patterns
        concept_patterns = [
            # Quoted phrases
            r'"([^"]{3,50})"',
            # Capitalized phrases (3+ words)
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){2,5})',
            # Technical terms
            r'\b([a-z]+(?:[A-Z][a-z]+)+)\b',  # camelCase or pascalCase
            r'\b([a-z]+(?:_[a-z]+)+)\b',      # snake_case
            # Parenthesized terms
            r'\(([^)]{3,50})\)'
        ]

        all_matches = []
        for pattern in concept_patterns:
            for sentence in sentences:
                matches = re.findall(pattern, sentence)
                all_matches.extend(matches)

        # Deduplicate and clean
        unique_concepts = set()
        for match in all_matches:
            if isinstance(match, tuple):
                match = match[0]  # Get first capture group

            concept = match.strip()
            if len(concept) >= 3 and concept not in unique_concepts:
                unique_concepts.add(concept)

                # Add to concepts list
                concepts.append({
                    "type": "concept",
                    "text": concept,
                    "modality": "text",
                    "importance": 1.0
                })

        return concepts


class SemanticStructureProcessor(BasePageProcessor):
    """Process semantic structure from web pages"""

    async def process(self, soup, context):
        """Extract semantic structure

        Args:
            soup: BeautifulSoup object for the page
            context: Context information about the page

        Returns:
            List of extracted knowledge items
        """
        knowledge_items = []

        # Extract headings and their content
        for heading_level in range(1, 7):
            headings = soup.find_all(f'h{heading_level}')
            for heading in headings:
                heading_text = heading.get_text(strip=True)
                if not heading_text or len(heading_text) < 3:
                    continue

                # Extract content under this heading
                content = []
                for sibling in heading.next_siblings:
                    # Stop at next heading of same or higher level
                    if sibling.name and sibling.name.lower() in [f'h{i}' for i in range(1, heading_level+1)]:
                        break

                    if sibling.name == 'p':
                        content.append(sibling.get_text(strip=True))

                # Create structured heading item
                if content:
                    knowledge_items.append({
                        "type": "experience",
                        "experience_type": "web_section",
                        "content": {
                            "heading": heading_text,
                            "level": heading_level,
                            "content": "\n".join(content)
                        },
                        "modality": "text",
                        "metadata": {
                            "heading_level": heading_level,
                            "url": context["url"],
                            "domain": context["domain"]
                        }
                    })

                # Add heading as concept
                if len(heading_text) > 3 and len(heading_text) < 100:
                    knowledge_items.append({
                        "type": "concept",
                        "text": heading_text,
                        "modality": "text",
                        "importance": 1.2  # Headings are more important
                    })

        # Extract lists
        lists = soup.find_all(['ul', 'ol'])
        for list_elem in lists:
            items = list_elem.find_all('li')
            if items:
                list_items = [item.get_text(strip=True) for item in items]
                list_items = [item for item in list_items if item]  # Remove empty items

                if list_items:
                    knowledge_items.append({
                        "type": "experience",
                        "experience_type": "web_list",
                        "content": {
                            "items": list_items,
                            "ordered": list_elem.name == 'ol'
                        },
                        "modality": "text"
                    })

                    # Add list items as concepts if appropriate
                    for item in list_items:
                        if 3 < len(item) < 50 and not item.endswith(('.', '?', '!')):
                            knowledge_items.append({
                                "type": "concept",
                                "text": item,
                                "modality": "text",
                                "importance": 1.0
                            })

        return knowledge_items


class CodeExtractionProcessor(BasePageProcessor):
    """Extract code blocks from web pages"""

    async def process(self, soup, context):
        """Extract code blocks

        Args:
            soup: BeautifulSoup object for the page
            context: Context information about the page

        Returns:
            List of extracted knowledge items
        """
        knowledge_items = []

        # Find code elements
        code_blocks = soup.find_all(['pre', 'code'])
        code_blocks.extend(soup.find_all(class_=["code", "highlight", "syntax", "prettyprint"]))

        for block in code_blocks:
            code_text = block.get_text(strip=True)

            # Skip empty or very short blocks
            if not code_text or len(code_text) < 20:
                continue

            # Try to detect language
            language = "unknown"
            classes = block.get('class', [])

            # Look for language hints in class names
            lang_prefixes = ["language-", "lang-", "brush:", "syntax-"]
            for cls in classes:
                for prefix in lang_prefixes:
                    if isinstance(cls, str) and cls.startswith(prefix):
                        language = cls[len(prefix):]
                        break

                # Common language class names
                if cls in ["python", "javascript", "java", "c", "cpp", "csharp", "ruby", "php", "html", "css"]:
                    language = cls
                    break

            # Store as experience
            knowledge_items.append({
                "type": "experience",
                "experience_type": "code_block",
                "content": {
                    "code": code_text,
                    "language": language
                },
                "modality": "text",
                "metadata": {
                    "language": language,
                    "url": context["url"],
                    "domain": context["domain"]
                }
            })

            # Extract function and class names as concepts
            if language in ["python", "javascript", "java", "c", "cpp", "csharp"]:
                # Extract function definitions
                function_pattern = r"\b(def|function|class|interface)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
                matches = re.findall(function_pattern, code_text)

                for match in matches:
                    concept = match[1]  # Function/class name
                    if concept and len(concept) > 2:
                        knowledge_items.append({
                            "type": "concept",
                            "text": concept,
                            "modality": "text",
                            "importance": 1.1,
                            "metadata": {
                                "code_type": match[0],  # def, function, class, etc.
                                "language": language
                            }
                        })

        return knowledge_items


class DataTableProcessor(BasePageProcessor):
    """Extract data tables from web pages"""

    async def process(self, soup, context):
        """Extract data from tables

        Args:
            soup: BeautifulSoup object for the page
            context: Context information about the page

        Returns:
            List of extracted knowledge items
        """
        knowledge_items = []

        # Find tables
        tables = soup.find_all('table')

        for table_idx, table in enumerate(tables):
            # Skip layout tables and tiny tables
            rows = table.find_all('tr')
            if len(rows) < 3:
                continue

            # Find header row
            header_cells = rows[0].find_all(['th'])
            if not header_cells:
                header_cells = rows[0].find_all(['td'])

            # Skip tables without headers
            if not header_cells:
                continue

            # Extract headers
            headers = [cell.get_text(strip=True) for cell in header_cells]
            headers = [h for h in headers if h]  # Remove empty headers

            # Skip tables with empty headers
            if not headers:
                continue

            # Extract data rows
            data_rows = []
            for row in rows[1:]:  # Skip header row
                cells = row.find_all(['td'])
                if cells:
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    # Only include rows with same length as headers
                    if len(row_data) == len(headers):
                        data_rows.append(row_data)

            # Skip tables with no data
            if not data_rows:
                continue

            # Store as experience
            table_data = {
                "headers": headers,
                "rows": data_rows
            }

            knowledge_items.append({
                "type": "experience",
                "experience_type": "data_table",
                "content": table_data,
                "modality": "text",
                "metadata": {
                    "url": context["url"],
                    "domain": context["domain"],
                    "table_index": table_idx
                }
            })

            # Extract headers as concepts
            for header in headers:
                if 3 < len(header) < 50:
                    knowledge_items.append({
                        "type": "concept",
                        "text": header,
                        "modality": "text",
                        "importance": 1.1
                    })

        return knowledge_items


# Integration helpers
def integrate_web_knowledge_extension(sam_model, config=None):
    """Integrate the Web Knowledge Extension with a SAM model

    Args:
        sam_model: SAM model instance to extend
        config: Optional WebCrawlConfig instance

    Returns:
        The initialized WebKnowledgeExtension instance
    """
    extension = WebKnowledgeExtension(sam_model, config)

    # Register commands
    if hasattr(sam_model, 'register_command'):
        sam_model.register_command('crawl', extension.crawl)
        sam_model.register_command('web_stats', extension.get_knowledge_stats)

    # Start extension services
    asyncio.create_task(extension.start())

    return extension
