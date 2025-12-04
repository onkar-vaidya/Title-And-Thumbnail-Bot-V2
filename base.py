import os
import logging
import json
import time
import requests
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
import re
import shutil
import html
from bs4 import BeautifulSoup
import requests
from youtube_transcript_api import YouTubeTranscriptApi

# Configuration




# Ensure Homebrew bin is in PATH for node and ffmpeg
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("video-processor")

# Gemini Client (Replaced with Local Ollama)
try:
    from openai import OpenAI
except ImportError:
    raise RuntimeError("openai package required. pip install openai")

MODEL = os.getenv("MODEL", "local-model")
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio", # required but unused
)

SUGGEST_URL = "https://suggestqueries.google.com/complete/search?client=firefox&ds=yt&gl=IN&hl=en-IN&q={}"

# --- Rate Limiter & Retry Logic ---

class RateLimiter:
    """Enforce a minimum interval between calls to avoid hitting rate limits."""
    def __init__(self, calls_per_minute=15):
        self.interval = 60.0 / calls_per_minute
        self.last_call = 0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()

# Global Rate Limiter
# Local model is the bottleneck, so we don't strictly need to rate limit for API costs.
# But we can keep a small delay to let the system breathe if needed.
# Setting to 60 RPM (1 per second) effectively disables it for most local inference speeds.
rate_limiter = RateLimiter(calls_per_minute=60)

def retry_with_backoff(retries=5, backoff_in_seconds=5):
    """Decorator to retry functions on exception with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    # Enforce rate limit before calling
                    rate_limiter.wait()
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(f"Failed after {retries} retries: {e}")
                        raise
                    
                    # Check if it's a 429 or similar
                    msg = str(e).lower()
                    if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
                        sleep = (backoff_in_seconds * 2 ** x) + random.uniform(0, 1)
                        logger.warning(f"Rate limit hit. Retrying in {sleep:.1f}s...")
                        time.sleep(sleep)
                    else:
                        # For other errors, maybe don't retry or retry less? 
                        # For now, we retry everything to be robust.
                        logger.warning(f"Error: {e}. Retrying...")
                        time.sleep(backoff_in_seconds)
                    x += 1
        return wrapper
    return decorator

# --- Core Functions ---

@retry_with_backoff(retries=3, backoff_in_seconds=2)
def get_suggestions(keyword: str):
    """Return list of suggestions from suggestqueries endpoint."""
    try:
        url = SUGGEST_URL.format(requests.utils.requote_uri(keyword))
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data[1] if len(data) > 1 else []
    except Exception as e:
        # logger.warning(f"Suggestion fetch failed: {e}") # Optional: reduce log noise
        return []

def repair_json(text):
    """Attempt to repair broken JSON."""
    text = text.strip()
    # Fix trailing commas
    text = re.sub(r',\s*([\]}])', r'\1', text)
    # Fix unquoted keys (simple case)
    # text = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
    return text

def clean_json_text(text):
    """Attempt to clean JSON text from LLM output."""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Find the first { and last }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end != -1:
        text = text[start:end]
    
    text = repair_json(text)
    text = repair_json(text)
    return text

def call_llm_with_json_retry(system_prompt, user_prompt, model=MODEL, max_retries=2, validation_callback=None):
    """
    Call LLM and retry with error message if JSON parsing fails OR if validation fails.
    validation_callback: function(dict) -> bool. Returns True if JSON is semantically valid.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                timeout=300,
            )
            text = response.choices[0].message.content
            cleaned_text = clean_json_text(text)
            
            # Try parsing
            try:
                data = json.loads(cleaned_text)
                
                # Optional Semantic Validation
                if validation_callback:
                    if not validation_callback(data):
                        raise ValueError("JSON is valid but failed semantic validation (missing keys?).")
                
                return data
                
            except (json.JSONDecodeError, ValueError) as e:
                # If it's the last attempt, raise or return empty
                if attempt == max_retries:
                    logger.warning(f"Final JSON parse/validation failed: {e}")
                    logger.warning(f"Failed text was: {text[:1000]}...") # Log the text for debugging
                    raise
                
                # Otherwise, feed back the error
                logger.warning(f"JSON parse/validation failed (Attempt {attempt+1}/{max_retries+1}). Asking LLM to fix...")
                error_feedback = f"Your previous response was invalid or incomplete. Error: {e}. \nResponse was: {text[:500]}...\n\nPlease fix it and regenerate the COMPLETE response as valid JSON. Do not return a partial fix."
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": error_feedback})
                
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            if attempt == max_retries:
                raise
            time.sleep(2)
            
    return {}

@retry_with_backoff(retries=3, backoff_in_seconds=5)
def validate_video_availability(url):
    """Check if video is available/accessible using requests."""
    try:
        # Allow redirects to catch "Sign in" pages which mean private/restricted
        response = requests.head(url, allow_redirects=True, timeout=5)
        
        if response.status_code != 200:
            return False
            
        # Check if redirected to a login page or something that isn't a video
        if "google.com/accounts" in response.url or "youtube.com/login" in response.url:
            return False
            
        return True
    except Exception as e:
        logger.warning(f"Video validation failed for {url}: {e}")
        return False

@retry_with_backoff(retries=3, backoff_in_seconds=5)
def fetch_video_data_bs4(video_id: str):
    """Fetch video metadata using BeautifulSoup (Scraping)."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to fetch video page: {response.status_code}")
            return None, None, 0
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 1. Title
        title = ""
        meta_title = soup.find("meta", property="og:title")
        if meta_title:
            title = meta_title["content"]
        else:
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.text.replace(" - YouTube", "")
        
        # 2. Description
        description = ""
        meta_desc = soup.find("meta", property="og:description")
        if meta_desc:
            description = meta_desc["content"]
        
        # 3. Duration (Tricky with BS4, usually in a script tag)
        # We'll try to find it in the meta tag 'duration' if available, or regex the script
        duration = 0
        # Try to find duration in meta itemprop="duration" content="PT..."
        meta_duration = soup.find("meta", itemprop="duration")
        if meta_duration:
            duration_iso = meta_duration["content"]
            # Parse ISO duration manually since we removed isodate
            # Format: PT1H2M10S
            import re
            match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_iso)
            if match:
                h = int(match.group(1) or 0)
                m = int(match.group(2) or 0)
                s = int(match.group(3) or 0)
                duration = h * 3600 + m * 60 + s
        
        return description, title, duration

    except Exception as e:
        logger.error(f"BS4 Scraping failed: {e}")
        return None, None, 0

@retry_with_backoff(retries=3, backoff_in_seconds=5)
def fetch_transcript_api(video_id: str):
    """Fetch transcript using youtube-transcript-api with robust fallback."""
    try:
        # 1. List all available transcripts (Using .list() as verified in tests)
        # Note: In this version, it seems to be an instance method named 'list'
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        target_transcript = None
        
        # 2. Try to find Manually Created Hindi or English
        try:
            target_transcript = transcript_list.find_manually_created_transcript(['hi', 'en'])
        except:
            pass
            
        # 3. Try to find Generated Hindi or English
        if not target_transcript:
            try:
                target_transcript = transcript_list.find_generated_transcript(['hi', 'en'])
            except:
                pass
        
        # 4. Fallback: Take ANY available transcript (first one)
        if not target_transcript:
            # transcript_list is iterable
            for t in transcript_list:
                target_transcript = t
                break
        
        if target_transcript:
            # Fetch the actual data
            fetched_transcript = target_transcript.fetch()
            # Concatenate text
            full_text = " ".join([entry.text for entry in fetched_transcript.snippets])
            return full_text
            
        return ""

    except Exception as e:
        logger.warning(f"Failed to fetch transcript via API for {video_id}: {e}")
        return ""

def fetch_video_data(video_id: str):
    """Fetch video description and transcript using BS4 (metadata) and youtube-transcript-api (transcript)."""
    
    # Rate limit is handled by decorators on called functions
    # time.sleep(random.uniform(0.8, 1.5))
    
    description = ""
    title = ""
    duration = 0
    transcript_text = ""
    
    url = f"https://www.youtube.com/watch?v={video_id}"
    url = html.unescape(url)
    
    # 0. Validate first
    if not validate_video_availability(url):
        logger.error(f"Video {video_id} is unavailable or private.")
        return "", "", "", 0

    # 1. Fetch Metadata (BS4)
    bs4_desc, bs4_title, bs4_duration = fetch_video_data_bs4(video_id)
    if bs4_title:
        description = bs4_desc
        title = bs4_title
        duration = bs4_duration
        logger.info(f"Fetched metadata via BeautifulSoup for {video_id}")
    else:
        # Fallback if BS4 fails
        logger.warning("BS4 failed to fetch metadata.")

    # 2. Fetch Transcript (youtube-transcript-api)
    transcript_text = fetch_transcript_api(video_id)
    
    if not transcript_text:
        logger.warning(f"No transcript found for {video_id}. Proceeding without it.")

    return description, transcript_text, title, duration

def extract_links(description: str):
    """Extract links from description."""
    # Simple regex for URLs
    urls = re.findall(r'(https?://\S+)', description)
    return list(set(urls))

@retry_with_backoff(retries=5, backoff_in_seconds=10)
def analyze_title_intent(title: str, transcript: str = ""):
    """Analyze title intent using Gemini/Ollama."""
    transcript_context = ""
    if transcript:
        # Truncate transcript to avoid context limit issues (e.g. first 10000 chars - approx 2500 tokens)
        transcript_context = f"\nVideo Transcript (excerpt): {transcript[:10000]}...\n"

    system_prompt = (
        "You are an expert YouTube Strategist. "
        "Your task is to analyze an 'old' YouTube title to guide a new content creation process. "
        "Return ONLY a valid JSON object."
    )
    user_prompt = f"""
    Old YouTube Title: "{title}"
    {transcript_context}

    Your task is to:
    1. Infer the most likely topic of the video based on the TRANSCRIPT content.
    2. Identify the primary keyword that best represents the actual content.
    3. List 10–15 semantic/LSI keywords found within the transcript or highly relevant to the discussed topics.
    4. Predict the full content structure of the video (what sections it likely contains)
    5. Classify the intent (Educational, Entertainment, Transactional)
    6. Write a clean, short context summary that will help a Python script decide which keywords to scrape from YouTube.

    DO NOT generate a new title or description here.
    Only return the context and keyword predictions.

    Output Format (JSON ONLY):
    {{
      "topic": "...",
      "primary_keyword": "...",
      "lsi_keywords": ["keyword1", "keyword2", ...],
      "content_structure": ["Section 1", "Section 2", ...],
      "intent": "...",
      "context_summary": "..."
    }}
    """
    
    try:
        return call_llm_with_json_retry(
            system_prompt, 
            user_prompt, 
            validation_callback=lambda x: "primary_keyword" in x
        )
    except Exception as e:
        logger.warning(f"LLM analysis failed with full context: {e}. Retrying with truncated context...")
        # Fallback to smaller context
        if transcript:
            transcript_context = f"\nVideo Transcript (excerpt): {transcript[:5000]}...\n"
            user_prompt = f"""
            Old YouTube Title: "{title}"
            {transcript_context}

            Your task is to:
            1. Infer the most likely topic of the video based on the TRANSCRIPT content.
            2. Identify the primary keyword that best represents the actual content.
            3. List 10–15 semantic/LSI keywords found within the transcript or highly relevant to the discussed topics.
            4. Predict the full content structure of the video (what sections it likely contains)
            5. Classify the intent (Educational, Entertainment, Transactional)
            6. Write a clean, short context summary that will help a Python script decide which keywords to scrape from YouTube.

            DO NOT generate a new title or description here.
            Only return the context and keyword predictions.

            Output Format (JSON ONLY):
            {{
              "topic": "...",
              "primary_keyword": "...",
              "lsi_keywords": ["keyword1", "keyword2", ...],
              "content_structure": ["Section 1", "Section 2", ...],
              "intent": "...",
              "context_summary": "..."
            }}
            """
        
        try:
            return call_llm_with_json_retry(
                system_prompt, 
                user_prompt,
                validation_callback=lambda x: "primary_keyword" in x
            )
        except Exception as final_e:
            logger.error(f"All LLM attempts failed: {final_e}")
            return {"primary_keyword": title, "lsi_keywords": [title], "context_summary": "Analysis failed."}

def deep_expand(seeds: list, target_count=50):
    """Recursively expand keywords."""
    collected = set(s.lower().strip() for s in seeds)
    queue = list(collected)
    searched = set()

    while len(collected) < target_count and queue:
        current_seed = queue.pop(0)
        if current_seed in searched:
            continue
        searched.add(current_seed)
        
        suggestions = get_suggestions(current_seed)
        for s in suggestions:
            s_clean = s.lower().strip()
            if len(s_clean) < 3: continue
            if s_clean not in collected:
                collected.add(s_clean)
                queue.append(s_clean)
                if len(collected) >= target_count: break
        time.sleep(0.05)
    # Deduplicate while preserving order
    seen = set()
    final_list = []
    for item in collected:
        if item not in seen:
            seen.add(item)
            final_list.append(item)
    return final_list

@retry_with_backoff(retries=5, backoff_in_seconds=10)
def generate_final_metadata(primary_keyword: str, lsi_keywords: list, scraped_keywords: list, context_summary: str, old_links: list, transcript: str, duration: int):
    """Generate final metadata using Gemini/Ollama."""
    
    links_str = "\n".join(old_links)
    transcript_context = ""
    if transcript:
        # Provide more transcript for chapters, maybe 10000 chars or full if model handles it.
        # Local models often have 4k context, so keep it safe.
        transcript_context = f"\nFull Video Transcript for Chapter Generation:\n{transcript[:10000]}\n"

    system_prompt = "You are an expert YouTube Strategist. Return ONLY a valid JSON object. Do not include any conversational text, markdown formatting, or explanations."
    user_prompt = f"""
    I will give you the following inputs:
    Primary keyword: {primary_keyword}
    LSI keywords: {', '.join(lsi_keywords)}
    Scraped YouTube/Google keyword suggestions: {', '.join(scraped_keywords)}
    Original context summary: {context_summary}
    Links to preserve:
    {links_str}
    
    {transcript_context}

    Using all inputs, generate final optimized YouTube metadata by following these steps:

    1. Create 3 Highly-Optimized Titles
    SEO-focused title (front-load keyword, under 60 chars)
    CTR/viral title (curiosity gap, emotional trigger, still accurate)
    Balanced title (CTR + SEO combined)

    2. Write a Fully Optimized Description
    Must include:
    2-line SEO hook with primary keyword
    200–250 word mini-blog body using semantic keywords
    IMPORTANT: You MUST include the provided "Links to preserve" at the end of the description or where appropriate.
    Strong CTA
    
    3. Generate Chapters/Timestamps (MANDATORY)
    Video Duration: {duration} seconds.
    
    Rules:
    - You MUST generate chapters.
    - If Duration < 60 seconds (Shorts): Generate exactly one timestamp "0:00 - [Engaging Hook]".
    - If Duration >= 60 seconds: Generate 3+ chapters (Intro, Key Points, Conclusion).
    - If Transcript is missing: INFER chapters based on the Title, Description, and Context. Do NOT skip this step.
    - Format: 00:00 - Chapter Title


    4. Generate Tags & Hashtags
    15–20 tags (mix of broad, niche, long-tail, misspellings)
    Exactly 3–5 hashtags

    5. Thumbnail Direction
    Create a visual concept that matches the strongest title.

    Output Format (JSON ONLY):
    {{
      "titles": [
        {{"type": "SEO", "text": "..."}},
        {{"type": "Viral", "text": "..."}},
        {{"type": "Balanced", "text": "..."}}
      ],
      "description": "...",
      "chapters": "00:00 - Intro...",
      "tags": ["tag1", "tag2", ...],
      "hashtags": ["#tag1", "#tag2", ...],
      "thumbnail_direction": "..."
    }}
    """
    
    try:
        return call_llm_with_json_retry(
            system_prompt, 
            user_prompt,
            validation_callback=lambda x: "titles" in x and isinstance(x["titles"], list)
        )
    except Exception as e:
        logger.warning(f"LLM generation failed with full context: {e}. Retrying with truncated context...")
        # Fallback
        if transcript:
             transcript_context = f"\nFull Video Transcript for Chapter Generation:\n{transcript[:5000]}\n"
             # Re-construct user prompt with smaller transcript
             user_prompt = f"""
    I will give you the following inputs:
    Primary keyword: {primary_keyword}
    LSI keywords: {', '.join(lsi_keywords)}
    Scraped YouTube/Google keyword suggestions: {', '.join(scraped_keywords)}
    Original context summary: {context_summary}
    Links to preserve:
    {links_str}
    
    {transcript_context}

    Using all inputs, generate final optimized YouTube metadata by following these steps:

    1. Create 3 Highly-Optimized Titles
    SEO-focused title (front-load keyword, under 60 chars)
    CTR/viral title (curiosity gap, emotional trigger, still accurate)
    Balanced title (CTR + SEO combined)

    2. Write a Fully Optimized Description
    Must include:
    2-line SEO hook with primary keyword
    200–250 word mini-blog body using semantic keywords
    IMPORTANT: You MUST include the provided "Links to preserve" at the end of the description or where appropriate.
    Strong CTA
    
    3. Generate Chapters/Timestamps (MANDATORY)
    Video Duration: {duration} seconds.
    
    Rules:
    - You MUST generate chapters.
    - If Duration < 60 seconds (Shorts): Generate exactly one timestamp "0:00 - [Engaging Hook]".
    - If Duration >= 60 seconds: Generate 3+ chapters (Intro, Key Points, Conclusion).
    - If Transcript is missing: INFER chapters based on the Title, Description, and Context. Do NOT skip this step.
    - Format: 00:00 - Chapter Title


    4. Generate Tags & Hashtags
    15–20 tags (mix of broad, niche, long-tail, misspellings)
    Exactly 3–5 hashtags

    5. Thumbnail Direction
    Create a visual concept that matches the strongest title.

    Output Format (JSON ONLY):
    {{
      "titles": [
        {{"type": "SEO", "text": "..."}},
        {{"type": "Viral", "text": "..."}},
        {{"type": "Balanced", "text": "..."}}
      ],
      "description": "...",
      "chapters": "00:00 - Intro...",
      "tags": ["tag1", "tag2", ...],
      "hashtags": ["#tag1", "#tag2", ...],
      "thumbnail_direction": "..."
    }}
    """

        try:
            return call_llm_with_json_retry(
                system_prompt, 
                user_prompt,
                validation_callback=lambda x: "titles" in x and isinstance(x["titles"], list)
            )
        except Exception as final_e:
            logger.error(f"All LLM attempts failed: {final_e}")
            return {}

# --- Worker Function ---

def process_single_row(row, index, total):
    """Process a single row and return the updated row dictionary."""
    try:
        title = str(row.get("Video Title", "")).strip()
        # if not title or title.lower() == "nan":
        #     return row

        # 1. Fetch Data
        old_desc, transcript, fetched_title, duration = fetch_video_data(str(row.get("Video ID", "")))
        
        # Use fetched title if input title is missing
        if not title or title.lower() == "nan":
            title = fetched_title
            # Update row so it appears in output
            row["Video Title"] = title
            
        if not title:
            logger.warning(f"Could not find title for video {row.get('Video ID', '')}")
            return row

        old_links = extract_links(old_desc)

        # 2. Analyze
        analysis = analyze_title_intent(title, transcript)
        primary_kw = analysis.get("primary_keyword", title)
        lsi_kws = analysis.get("lsi_keywords", [])
        context = analysis.get("context_summary", "")

        # 3. Scrape
        seeds = [primary_kw] + lsi_kws
        scraped = deep_expand(seeds, target_count=40)

        # 4. Generate
        metadata = generate_final_metadata(primary_kw, lsi_kws, scraped, context, old_links, transcript, duration)

        # 4. Fill Data
        titles = metadata.get("titles", [])
        seo_title = next((t["text"] for t in titles if t.get("type") == "SEO"), "")
        viral_title = next((t["text"] for t in titles if t.get("type") == "Viral"), "")
        balanced_title = next((t["text"] for t in titles if t.get("type") == "Balanced"), "")

        # Update Row
        updated_row = row.copy()
        updated_row["Optimized Title (SEO)"] = seo_title
        updated_row["Optimized Title (Viral)"] = viral_title
        updated_row["Optimized Title (Balanced)"] = balanced_title
        
        # Combine Description and Chapters
        final_description = metadata.get("description", "")
        chapters = metadata.get("chapters", "")
        
        # Normalize chapters if list
        if isinstance(chapters, list):
            chapters = "\n".join(chapters)
            
        if chapters:
            final_description += f"\n\nTimestamps:\n{chapters}"
            
        # Sanitize tags and hashtags (ensure list of strings)
        def sanitize_list(lst):
            if not isinstance(lst, list): return []
            clean = []
            for item in lst:
                if isinstance(item, str):
                    clean.append(item)
                elif isinstance(item, dict):
                    # Extract values if it's a dict like {"tag": "value"} or just take the first value
                    clean.extend([str(v) for v in item.values() if isinstance(v, (str, int, float))])
                else:
                    clean.append(str(item))
            return clean

        updated_row["New Description"] = final_description
        updated_row["New Tags"] = ", ".join(sanitize_list(metadata.get("tags", [])))
        updated_row["New Hashtags"] = ", ".join(sanitize_list(metadata.get("hashtags", [])))
        updated_row["Thumbnail Direction"] = metadata.get("thumbnail_direction", "")
        updated_row["Primary Keyword"] = primary_kw
        updated_row["Context Summary"] = context
        
        # Only mark success if we actually got an optimized title
        if seo_title:
            updated_row["Status"] = "Success"
        else:
            updated_row["Status"] = "Failed"
            logger.warning(f"Row {index} marked Failed: No SEO title generated.")
        
        return updated_row
    except Exception as e:
        logger.error(f"Error processing row {index}: {e}")
        row["Status"] = "Failed"
        return row

# --- Main Processing Logic ---

def process_file(input_file, output_file, max_workers=1):
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"Reading from {input_file}...")
    
    # Read Input
    ext = os.path.splitext(input_file)[1].lower()
    try:
        if ext == '.xlsx':
            df_input = pd.read_excel(input_file)
        elif ext == '.csv':
            df_input = pd.read_csv(input_file)
        else:
            logger.error(f"Unsupported file extension: {ext}")
            return
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return

    # Check for existing output to resume
    processed_ids = set()
    df_output = pd.DataFrame()
    
    if os.path.exists(output_file):
        logger.info(f"Found existing output file {output_file}. Resuming...")
        try:
            out_ext = os.path.splitext(output_file)[1].lower()
            if out_ext == '.xlsx':
                df_output = pd.read_excel(output_file)
            else:
                df_output = pd.read_csv(output_file)
            
            # Check which IDs are *successfully* processed
            if "Video ID" in df_output.columns and "Status" in df_output.columns:
                for index, row in df_output.iterrows():
                    vid_id = str(row.get("Video ID", ""))
                    status = str(row.get("Status", ""))
                    
                    if status == "Success":
                        processed_ids.add(vid_id)
                
                logger.info(f"Already successfully processed {len(processed_ids)} videos.")
            elif "Video ID" in df_output.columns:
                 # Legacy check for older files without Status column
                 for index, row in df_output.iterrows():
                    vid_id = str(row.get("Video ID", ""))
                    seo_title = str(row.get("Optimized Title (SEO)", ""))
                    if not pd.isna(seo_title) and seo_title != "" and seo_title.lower() != "nan":
                         processed_ids.add(vid_id)
        except Exception as e:
            logger.warning(f"Could not read existing output file: {e}. Starting fresh.")

    # Filter rows to process
    all_rows = df_input.to_dict('records')
    rows_to_process = []
    
    for row in all_rows:
        vid_id = str(row.get("Video ID", ""))
        if vid_id not in processed_ids:
            rows_to_process.append(row)
    
    total = len(rows_to_process)
    if total == 0:
        logger.info("All videos already successfully processed!")
        return

    logger.info(f"Processing {total} remaining videos with {max_workers} threads...")

    # Initialize results with existing rows (including failed ones, which we will overwrite/append)
    # Actually, if we want to *fix* failed ones, we should probably rebuild the list or update in place.
    # Simpler approach: Load existing valid rows, and append new ones. 
    # But we need to preserve the order or at least not duplicate.
    # Let's just append new results to the file. If the file has "Error" rows, we will just append correct ones at the end?
    # No, that duplicates IDs. 
    # Better: Read the file, keep ONLY the successful ones, and then start processing the rest.
    
    final_results = []
    if not df_output.empty:
        # Keep only successful rows in the starting buffer
        for index, row in df_output.iterrows():
            vid_id = str(row.get("Video ID", ""))
            if vid_id in processed_ids:
                final_results.append(row.to_dict())
    
    # Now we have a clean slate of successful rows. We will process the rest.
    # Note: This effectively "deletes" the error rows from the output file when we first save.

    
    completed_count = 0
    lock = threading.Lock()

    # Use ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_single_row, row, i, total): i 
            for i, row in enumerate(rows_to_process)
        }
        
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                result_row = future.result()
                
                with lock:
                    final_results.append(result_row)
                    completed_count += 1
                    
                    # Real-time saving: Save every 5 rows
                    if completed_count % 5 == 0:
                        logger.info(f"Progress: {completed_count}/{total} - Saving batch...")
                        temp_df = pd.DataFrame(final_results)
                        out_ext = os.path.splitext(output_file)[1].lower()
                        temp_output_file = output_file + ".tmp"
                        
                        if out_ext == '.xlsx':
                            # Prefer CSV for intermediate if possible, but stick to requested format
                            temp_df.to_excel(temp_output_file, index=False)
                        else:
                            temp_df.to_csv(temp_output_file, index=False)
                        
                        # Atomic move
                        shutil.move(temp_output_file, output_file)

            except Exception as e:
                logger.error(f"Row {i} failed: {e}")
                
        # Final save for any remaining rows
        if completed_count % 5 != 0:
             logger.info(f"Final Save... ({completed_count}/{total})")
             with lock:
                temp_df = pd.DataFrame(final_results)
                out_ext = os.path.splitext(output_file)[1].lower()
                temp_output_file = output_file + ".tmp"
                
                if out_ext == '.xlsx':
                    temp_df.to_excel(temp_output_file, index=False)
                else:
                    temp_df.to_csv(temp_output_file, index=False)
                
                shutil.move(temp_output_file, output_file)

    logger.info("All tasks completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process YouTube videos for metadata optimization.")
    parser.add_argument("input_file", help="Path to input file (CSV or Excel)")
    parser.add_argument("output_file", help="Path to output file (CSV or Excel)")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads (default: 1)")
    
    args = parser.parse_args()
    
    # Force threads to 1 or low number to respect rate limits, unless user overrides
    # But even with high threads, the RateLimiter will bottleneck it.
    process_file(args.input_file, args.output_file, max_workers=args.threads)
