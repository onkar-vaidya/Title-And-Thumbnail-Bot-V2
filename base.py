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
import yt_dlp
import shutil
import html




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

MODEL = os.getenv("MODEL", "llama3.1")
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama", # required but unused
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

def get_suggestions(keyword: str):
    """Return list of suggestions from suggestqueries endpoint."""
    try:
        url = SUGGEST_URL.format(requests.utils.requote_uri(keyword))
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data[1] if len(data) > 1 else []
    except Exception as e:
        return []

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
    
    # Attempt to fix common issues like unescaped newlines in strings
    # This is tricky without a proper parser, but let's try a simple pass
    # or just rely on the model being better if we prompt it better?
    # For now, let's just return the extracted block.
    return text

def validate_video_availability(url):
    """Check if video is available/accessible using yt-dlp simulation."""
    ydl_opts = {
        'quiet': True,
        'simulate': True,
        'skip_download': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=False)
        return True
    except Exception as e:
        logger.warning(f"Video validation failed for {url}: {e}")
        return False

def fetch_video_data(video_id: str):
    """Fetch video description and transcript using yt-dlp with retries and fallback."""
    
    # Rate limit delay
    time.sleep(random.uniform(0.8, 1.5))
    
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Decode HTML entities in URL just in case
    url = html.unescape(url)
    
    # 0. Validate first
    if not validate_video_availability(url):
        logger.error(f"Video {video_id} is unavailable or private.")
        return "", "", "", 0

    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'sleep_interval': 2,
        'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    }
    
    description = ""
    transcript_text = ""
    title = ""
    duration = 0
    
    # Retry logic for metadata fetching
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                description = info.get('description', '')
                title = info.get('title', '')
                duration = info.get('duration', 0)
                
                subtitles = info.get('subtitles', {})
                automatic_captions = info.get('automatic_captions', {})
                
                # Determine best language
                available_langs = set(subtitles.keys()) | set(automatic_captions.keys())
                selected_lang = 'en' # Default
                
                if 'hi' in available_langs:
                    selected_lang = 'hi'
                elif 'en' in available_langs:
                    selected_lang = 'en'
                elif available_langs:
                    selected_lang = list(available_langs)[0]
                    
                logger.info(f"Selected language for transcript: {selected_lang}")
                break # Success
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Metadata fetch failed for {video_id} (Attempt {attempt+1}/{max_retries}). Retrying in {wait_time:.1f}s... Error: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Error fetching metadata for {video_id} after retries: {e}")
                return description, "", title, 0

    # 2. Download transcript in selected language with fallback
    try:
        ydl_opts['skip_download'] = True
        ydl_opts['writeautomaticsub'] = True
        ydl_opts['writesubtitles'] = True
        ydl_opts['subtitleslangs'] = [selected_lang]
        ydl_opts['outtmpl'] = f'temp_{video_id}'
        
        # Retry logic for transcript download
        for attempt in range(max_retries):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Transcript download failed (Attempt {attempt+1}). Retrying... Error: {e}")
                    time.sleep(wait_time)
                else:
                    raise e

        # Find the file
        found_file = False
        for ext in ['vtt', 'srv3', 'ttml']:
            fname = f'temp_{video_id}.{selected_lang}.{ext}'
            if os.path.exists(fname):
                with open(fname, 'r', encoding='utf-8') as f:
                    transcript_text = f.read()
                os.remove(fname)
                found_file = True
                break
            # Fallback naming check
            fname_alt = f'temp_{video_id}.{ext}' 
            if os.path.exists(fname_alt):
                 with open(fname_alt, 'r', encoding='utf-8') as f:
                    transcript_text = f.read()
                 os.remove(fname_alt)
                 found_file = True
                 break
        
        if not found_file:
            logger.warning(f"Transcript file not found for {video_id} after download.")

        # Simple cleanup
        if transcript_text:
            transcript_text = re.sub(r'<[^>]+>', '', transcript_text)
            transcript_text = re.sub(r'WEBVTT', '', transcript_text)
            transcript_text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}', '', transcript_text)
            transcript_text = re.sub(r'align:start position:0%', '', transcript_text)
            transcript_text = "\n".join([line.strip() for line in transcript_text.split('\n') if line.strip()])
        
    except Exception as e:
        logger.warning(f"Could not download transcript for {video_id}: {e}. Proceeding without transcript.")

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
        # Truncate transcript to avoid context limit issues (e.g. first 25000 chars)
        transcript_context = f"\nVideo Transcript (excerpt): {transcript[:25000]}...\n"

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
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        timeout=60,
    )
    text = response.choices[0].message.content
    cleaned_text = clean_json_text(text)
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        # Try to use a more lenient parser if available, or just log and fail
        # Let's try to strip control characters
        cleaned_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_text)
        try:
            return json.loads(cleaned_text)
        except:
            logger.warning(f"Failed to parse JSON: {text[:100]}...")
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
    return list(collected)

@retry_with_backoff(retries=5, backoff_in_seconds=10)
def generate_final_metadata(primary_keyword: str, lsi_keywords: list, scraped_keywords: list, context_summary: str, old_links: list, transcript: str, duration: int):
    """Generate final metadata using Gemini/Ollama."""
    
    links_str = "\n".join(old_links)
    transcript_context = ""
    if transcript:
        # Provide more transcript for chapters, maybe 30000 chars or full if model handles it.
        # Llama 3.1 8B has 128k context, so we can pass a lot.
        transcript_context = f"\nFull Video Transcript for Chapter Generation:\n{transcript[:30000]}\n"

    system_prompt = "You are an expert YouTube Strategist. Return ONLY a valid JSON object."
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
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        timeout=90,
    )
    text = response.choices[0].message.content
    cleaned_text = clean_json_text(text)
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        cleaned_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_text)
        try:
            return json.loads(cleaned_text)
        except:
            logger.warning(f"Failed to parse JSON: {text[:100]}...")
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
        if chapters:
            final_description += f"\n\nTimestamps:\n{chapters}"
            
        updated_row["New Description"] = final_description
        updated_row["New Tags"] = ", ".join(metadata.get("tags", []))
        updated_row["New Hashtags"] = ", ".join(metadata.get("hashtags", []))
        updated_row["Thumbnail Direction"] = metadata.get("thumbnail_direction", "")
        updated_row["Primary Keyword"] = primary_kw
        updated_row["Context Summary"] = context
        
        return updated_row
    except Exception as e:
        logger.error(f"Error processing row {index}: {e}")
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
            if "Video ID" in df_output.columns:
                for index, row in df_output.iterrows():
                    vid_id = str(row.get("Video ID", ""))
                    # Check if it was an error or empty
                    context = str(row.get("Context Summary", ""))
                    seo_title = str(row.get("Optimized Title (SEO)", ""))
                    
                    if context != "Error." and seo_title != "nan" and seo_title != "":
                        processed_ids.add(vid_id)
                
                logger.info(f"Already successfully processed {len(processed_ids)} videos.")
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
                    
                    # Real-time saving: Save every 1 row (since user asked for real-time)
                    # For performance on large files, maybe every 5 is better, but let's do 1 for safety.
                    if completed_count % 1 == 0:
                        logger.info(f"Progress: {completed_count}/{total} - Saving...")
                        temp_df = pd.DataFrame(final_results)
                        out_ext = os.path.splitext(output_file)[1].lower()
                        temp_output_file = output_file + ".tmp"
                        
                        if out_ext == '.xlsx':
                            temp_df.to_excel(temp_output_file, index=False)
                        else:
                            temp_df.to_csv(temp_output_file, index=False)
                        
                        # Atomic move
                        shutil.move(temp_output_file, output_file)

            except Exception as e:
                logger.error(f"Row {i} failed: {e}")

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
