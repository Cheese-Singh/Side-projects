# Side Projects Index

A collection of technical explorations and utility scripts.

## Contents

### [MLX-Series](./MLX%20Series/)
* **Description:** Progressive exploration of the MLX framework for running and understanding machine learning models on Apple Silicon.
* **Focus:** From tensor operations and manual embeddings to a fully voice-driven agentic assistant with on-device inference, macOS system control, and real-time audio I/O.
* **Files:**
  - `requirements.txt` → Python dependencies required for MLX inference, MLX-VLM, Whisper transcription, and audio recording
  - `A01_mlx_basics.py` → Tensor operations, vectorization, manual tokenization, and embedding layer implementation
  - `A02_mlx_llm_inference.py` → Running a local Qwen3.5-9B model using MLX with configurable inference parameters
  - `A03_mlx_chat_with_history.py` → Running a local Qwen3.5-9B model through MLX while maintaining chat history
  - `A04_voice_qwen_assistant.py` → Voice-to-LLM assistant pipeline: records audio, transcribes speech using Whisper, and generates responses using Qwen3.5-9B through MLX-VLM
  - `A05_mlx_with_tools.py` → Lightweight tool-use with structured JSON tool-call prompting, CSV-backed task storage, and sandboxed file operations.
  - `A06_mlx_tts.py` → Voice assistant with Qwen3.5-9B and Kokoro-82M TTS; maintains chat history, strips think blocks, and streams audio sentence-by-sentence for low latency.
  - `A07_mlx_whisper.py` → VAD-based voice pipeline: silence-aware recording, Whisper transcription, and Qwen3.5-9B response generation.
  - `A08_mlx_voice_loop.py` → Full duplex voice assistant: VAD recording, Whisper transcription, Qwen3.5-9B generation, and Kokoro-82M sentence-streamed TTS output.
  - `A09_mlx_system_agent.py` → Agentic macOS control: dynamic tool registration, AppleScript-driven app and music control, and web search with context-augmented LLM responses.
  - `A10_mlx_voice_agent.py` → Complete voice-driven system agent: VAD recording, MLX-Whisper transcription, Qwen3.5-9B tool-calling, macOS control, and Kokoro-82M TTS output. No keyboard input required.
  - `A11_Gwen.py` → a Siri-esque agentic voice assistant with advanced capabilities. Uses Qwen3.5-9B model for complex tasks and Qwen2.5-1.5B for smaller tasks for quicker responses. Includes SQLite for global user state + latency logs + web search cache with semantic dedup + session summaries, and ChromaDB for semantic memory store + retrieval. Versioned auto-update logic for global state.
* **Stack:** `mlx`, `mlx-vlm`, `mlx-audio`, `mlx-whisper`, `transformers`, `torch`, `pyaudio`, `python`, `whisper`, `sounddevice`, `beautifulsoup4`, `requests`, `ddgs`

---

### [Ollama-Series](./Ollama%20Series/)
* **Description:** Progressive exploration of local LLM inference and interaction patterns using the Ollama runtime.
* **Focus:** From basic generation to multimodal input, stateful conversation, and agentic tool-use with function calling.
* **Files:**
  - `B1_ollama_basic_generate.py` → Single-turn text generation using `ollama.chat()` with streaming output
  - `B2_ollama_image_input.py` → Multimodal input handling with image encoding and vision-language models
  - `B3_ollama_chat_with_history.py` → Multi-turn chat with persistent history, command handling (clear/history/exit), think-block stripping, and repetition loop prevention
  - `B4_ollama_with_tools.py` → Agentic tool-use loop with function calling; manages a CSV-backed inventory (check, add, update, save) with error handling and multi-step reasoning
  - `B5_ollama_cli.py` → Command-line interface for Modelfile lifecycle management and offline LLM evaluation; includes model validation, automated creation, interactive chat loop, and structured prompt benchmarking with report generation
* **Stack:** `ollama`, `pandas`, `python`

---

### [Research ML](./Research%20ML/)
* **Description:** Research-oriented machine learning and analytics workflows for materials science and scientific literature analysis.
* **Focus:** Small-dataset regression, supercapacitor capacitance prediction, LOOCV evaluation, diagnostic plotting, and OpenAlex-based publication trend analysis.
* **Files:**
  - `catboost_loocv_regression.py` → Trains a CatBoost regression model to predict supercapacitor capacitance (F/g) using material, synthesis, and electrolyte features; evaluates performance with Leave-One-Out Cross-Validation and generates diagnostic plots
  - `research-trend-analyzer.py` → Queries the OpenAlex API to track yearly publication volume for a selected research topic and visualizes the trend using a bar chart
  - `datasets/` → Stores the capacitance dataset used for regression modeling
  - `plots/` → Stores generated evaluation plots such as actual-vs-predicted, residual distribution, error histogram, and feature importance
* **Stack:** `python`, `pandas`, `numpy`, `catboost`, `scikit-learn`, `matplotlib`, `requests`, `OpenAlex API`

---

### [Mini](./Mini/)
* **Description:** Small, self-contained utility applications and programs exploring on-device ML pipelines and native macOS/iOS interfaces.
* **Focus:** Rapid prototyping of end-to-end AI-powered tools with real hardware integration, local inference backends, some with native Swift frontends.
* **Files:**
  - `VoiceSummariser/` → macOS SwiftUI app with a FastAPI backend; records microphone audio, transcribes speech using MLX-Whisper (distil-large-v3), and generates a concise summary using Qwen3.5-9B via MLX-VLM. Full on-device inference with no external API calls.
  - `Translate/` → Records microphone audio, feeds the recordings to Gemma-4B for translation, and transcribes speech using MLX-Whisper in 3 Languages : Spanish, French, and Italian. Currently holds only the backend.
* **Stack:** `swift`, `swiftui`, `avfoundation`, `fastapi`, `mlx-whisper`, `mlx-vlm`, `ffmpeg`, `python`, `pyaudio`, `sounddevice`, mlx-lm`, `mlx-audio`

---

## Technical Objectives

This repository documents:
- Hands-on exploration of new AI/ML frameworks such as MLX for Apple Silicon inference
- Progression from low-level numerical operations to applied local LLM and voice-assistant systems
- Development of research-oriented ML workflows for small scientific datasets
- Use of external APIs for data-driven research trend analysis
- Emphasis on reproducible, modular scripts with clear inputs, outputs, and generated plots

Each module represents a step toward building more complex AI systems, combining practical implementation with an understanding of the underlying mechanics rather than relying solely on high-level abstractions.
