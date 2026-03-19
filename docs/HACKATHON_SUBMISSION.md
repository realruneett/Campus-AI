# Campus-AI: Hackathon Submission Answers

## What is your Idea about?

Campus-AI is a Generative AI platform specifically tailored for Indian colleges and universities that automates the creation of highly professional, culturally-aware event posters. It empowers anyone, regardless of design expertise, to generate stunning marketing materials for any campus occasion—from Tech Hackathons and Robotics Seminars to Cultural Fests like Diwali, Garba, and Freshers' Parties—in just 10-15 seconds.

Unlike generic image generators, our platform uses a custom-trained Stable Diffusion XL model (fine-tuned on over 71,000 curated Indian campus posters) paired with an intelligent LLM-powered prompt engine. It features a complete 6-mode creation suite including Text-to-Poster generation, Reference Image style transfer (IP-Adapter), HD Upscaling (Real-ESRGAN), and a specialized typography compositor to ensure text is always clean and professional.

## What problem are you trying to solve?

Creating high-quality promotional materials is a massive bottleneck for student organizations, clubs, and councils across thousands of universities. It typically requires expensive software, specialized graphic design skills, and hours of manual labor per poster.

Furthermore, existing generic GenAI tools (like Midjourney or DALL-E 3) fail at this task for two main reasons:

1. **Lack of Cultural Context:** Generic models fundamentally lack training data on Indian festival visual languages (like rangoli, diyas, torans), regional cultural diversity, and campus-specific poster conventions.
2. **The Text Rendering Problem:** Diffusion models notoriously struggle to render clean, readable typography, making raw AI generations unusable for real-world event marketing.

Campus-AI solves this by providing the first domain-specific AI solution that natively understands Indian collegiate aesthetics and utilizes a hybrid pipeline (AI background generation + programmatic typography compositing) to guarantee flawless text rendering.

## Technology Stack being used

* **Core Base Model:** Stable Diffusion XL 1.0 (2.6B parameters) via HuggingFace `diffusers`.
* **Fine-Tuning:** LoRA (Low-Rank Adaptation) trained using `ai-toolkit` with a novel dual-phase curriculum and Prodigy optimizer.
* **Dataset Engineering:** Headless Selenium scrapers with SQLite-backed O(1) Perceptual Hash (pHash) caching for absolute deduplication, and PyTorch for GPU-accelerated Laplacian variance quality filtering.
* **Vision-Language Model:** Microsoft `Florence-2-Large` for dense, multi-modal dataset captioning.
* **Intelligent Prompt Engine:** Llama 3.3 70B (accessed via **Groq API** for extreme low-latency inference) to automatically expand basic user inputs into 10 optimized visual styles.
* **Advanced Pipelines:** `IP-Adapter` for structural style transfer and `Real-ESRGAN` for 4x HD upscaling.
* **User Interface & Deployment:** **Gradio** web application integrated with Python `PIL` (Pillow) for dynamic, localized typography compositing.

## Impact of your solution

Campus-AI democratizes state-of-the-art graphic design for the underserved market of over 40,000 Indian colleges and universities. By reducing the time required to create a professional poster from hours down to seconds—for zero cost—we eliminate the design bottleneck for student bodies, academic departments, and event organizers.

This enables significantly higher quality campus marketing, amplifies student engagement, and allows event organizers to focus on the actual execution of their programs rather than struggling with design software. Furthermore, our highly optimized LoRA+ training pipeline proves that domain-specific generative AI solutions can be trained and deployed efficiently on consumer-grade hardware, making the underlying technology highly scalable and accessible.
