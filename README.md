# Transkrypcja_Polska
Benchmarking Polish Speech Recognition: Whisper against commercial ASR

# Benchmarking Polish Speech Recognition Systems

Automatic speech transcription has become increasingly common with the rise of audio models. However, when it comes to less-represented languages such as **Polish**, the task becomes far more complex.  
This project compares several **speech recognition solutions** for Polish, under **real-world, large-scale conditions**.

---

## 🎯 Project Objective

Evaluate the performance of different transcription solutions for Polish:

1. **Whisper (OpenAI)**  
   - Model: `whisper-large-v3-turbo`  
   - Open-source, downloadable via Hugging Face for local use  
   - Low recurring costs & full data control (no API dependency)  
   - Reference standard for multilingual ASR  

2. **ElevenLabs**  
   - Founded in 2022, first known for ultra-realistic speech synthesis  
   - Proprietary transcription model: *Scribe*  
   - Free: 2h30/month  
   - Paid tiers: from **$5/month → 12h30 (~$0.40/h)**  

3. **AssemblyAI**  
   - Popular ASR provider in the U.S.  
   - Proprietary engine: *Universal-2*  
   - Free: $50 credits (185h non-streaming / 333h streaming)  
   - Costs: **$0.27/h async** | **$0.15/h streaming**  

4. **Speechmatics**  
   - UK company, pioneer in ASR (founded 2006)  
   - Specialized in multilingual transcription  
   - Free: 8h/month  
   - Costs: from **$0.24/h**  

---

## 🚀 Achievements

### 1. Interactive Testing App
- Built a **Gradio app** (Hugging Face Space, ZeroGPU)  
- Records audio from microphone  
- Sends input simultaneously to **Whisper, ElevenLabs, AssemblyAI, Speechmatics**  
- Displays instant transcription results  

### 2. Large-Scale Benchmark on Common Voice
- **Dataset**: [mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) (Polish, test set, 9,230 recordings)  
- **Metrics**:  
  - Word Error Rate (WER)  
  - Processing time  
  - Processing stability (error rate)  

**Optimizations for runtime**:  
- GPU (T4, 16GB VRAM)  
- **Multithreading** → parallel I/O-bound tasks  
- **Async programming** → non-blocking API calls  
- **Asencio (Python)** → combines async + semaphore for API throttling control  
- **Batch processing (100 files/batch)** → checkpointing, memory cleanup, resumable runs  

---

## 📊 Evaluation

### 🔹 Word Error Rate (WER)
Standard metric for ASR:  
\[
WER = \frac{S + D + I}{N}
\]  
- *S* = substitutions, *D* = deletions, *I* = insertions, *N* = total reference words  

**Example**:  
Reference: *"Dzień dobry wszystkim"*  
Prediction: *"Dzień dobry wszystkim ludziom"*  
Result: 1 insertion → **WER = 1/3 = 33%**  

---

## ✅ Results

### 1. Overall Runtime
- **Whisper + ElevenLabs + AssemblyAI**: ~3h for 9,230 files  
- **Speechmatics**: ~4h10 (avg. ~10s/file, even with concurrency=10)  

### 2. Runtime per Model
- **Whisper**: fastest – 94% of files <1s  
- **ElevenLabs**: ~80% <1s  
- **AssemblyAI**: ~6% >5s (problematic for real-time)  
- **Speechmatics**: slowest (~10s/file)  

👉 *Whisper* & *ElevenLabs* are best for interactive use cases (chatbots, live captions).  

### 3. Accuracy (WER)
- **Speechmatics**: 98.9%  
- **ElevenLabs**: 98.6%  
- **Whisper**: 95.6%  
- **AssemblyAI**: 95.7%  

👉 *Speechmatics* & *ElevenLabs* are most consistent; *Whisper* trades some accuracy for speed.  

### 4. Error Analysis
- **55%**: perfectly transcribed by all models (WER=0)  
- **10%**: failed across all (noisy/low-quality audio) → require preprocessing  
- **17%**: errors only in a single model  

### 5. Accuracy vs. Runtime Trade-off
- Speed (<1.5s) & Accuracy (WER ≤5%):  
  - **Whisper**: 92% of files  
  - **ElevenLabs**: 95% of files  
- *Speechmatics*: highly accurate, too slow  
- *AssemblyAI*: weaker in both  

---

## 🏆 Conclusion

- **Whisper** → ⚡ Best for **real-time**, ultra-fast with solid accuracy  
- **ElevenLabs** → 🔥 Best **compromise**: speed + stability  
- **Speechmatics** → 🎯 Most **accurate**, but too slow in batch mode  
- **AssemblyAI** → ❌ Less competitive (slower + less stable)  

---

## 📌 Tech Highlights
- **Frameworks**: Python, Gradio, Hugging Face Spaces  
- **Optimizations**: async I/O, multithreading, semaphore-based job control  
- **Dataset**: Mozilla Common Voice v17 (Polish)  
- **Hardware**: T4 GPU (16GB VRAM)  
