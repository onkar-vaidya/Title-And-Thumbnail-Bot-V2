# 🎥 YouTube Metadata Optimizer Bot V2

**Automate your YouTube growth with AI-powered metadata optimization.**

This tool uses a local LLM (via **Ollama**) to analyze your videos and generate high-performing Titles, Descriptions, Chapters, Tags, and Thumbnail concepts. It goes beyond simple keywords by "watching" your video (analyzing the full transcript) to understand the true context and intent.

## 🚀 Key Features

-   **🧠 Deep Content Analysis**: Reads up to **25,000 characters** of your video's transcript to understand specific topics, camera models, or techniques mentioned.
-   **🌍 Multi-Language Support**: Automatically detects and prioritizes **Hindi** transcripts, falling back to English or auto-captions.
-   **🛡️ Robust & Resilient**:
    -   **Smart Retries**: Handles network glitches and 429 errors with exponential backoff.
    -   **Subtitle Fallback**: If transcripts are unavailable, it infers chapters and context from the title/description so your pipeline never breaks.
    -   **Atomic Writes**: Saves progress safely to prevent data corruption.
-   **⚡️ Automated Workflow**:
    -   Fetches **Video Title** automatically if missing.
    -   Preserves your existing **Affiliate/Social Links**.
    -   Generates **Mandatory Chapters** (even for Shorts!).
-   **🔒 Privacy-First**: Runs locally using **Ollama** (Llama 3.1 recommended)—no API costs, no data leaks.

## 🛠️ Prerequisites

1.  **Python 3.8+**
2.  **Ollama**: [Download & Install](https://ollama.com/)
    -   Pull the model: `ollama pull llama3.1`
3.  **System Dependencies** (Recommended for best performance):
    -   `ffmpeg` (for stable media processing)
    -   `node` (for yt-dlp JavaScript execution)
    -   *Install via Homebrew:* `brew install ffmpeg node`

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/onkar-vaidya/Title-And-Thumbnail-Bot-V2.git
    cd Title-And-Thumbnail-Bot-V2
    ```

2.  **Install Python dependencies**:
    ```bash
    pip install pandas openai yt-dlp openpyxl requests
    ```

## 🚦 Usage

1.  **Prepare your Input File** (`input.csv` or `.xlsx`):
    Create a file with a single column named `Video ID`.
    ```csv
    Video ID
    dQw4w9WgXcQ
    W2LJQUmMaWI
    ...
    ```
    *(You can optionally add a "Video Title" column, but the bot fetches it automatically!)*

2.  **Run the Bot**:
    ```bash
    python3 base.py input.csv output.csv --threads 1
    ```

3.  **Get Optimized Results**:
    The script will generate `output.csv` containing:
    -   **3 Optimized Titles** (SEO, Viral, Balanced)
    -   **New Description** (with Hook, Mini-blog, Links, and **Timestamps**)
    -   **Tags & Hashtags**
    -   **Thumbnail Direction** (AI visual concept)
    -   **Context Summary** & **Primary Keyword**

## ⚙️ Configuration

-   **Model Selection**: Default is `llama3.1`. To use a different model:
    ```bash
    export MODEL=mistral
    python3 base.py ...
    ```
-   **Threads**: Use `--threads N` for parallel processing (Note: Local LLMs are resource-intensive, so keep this low).

## 📝 License

This project is open-source and available for personal and commercial use.
