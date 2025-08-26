### Imports
import os
import shutil
import math
import time
import logging
import asyncio
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import librosa
import soundfile as sf
import pandas as pd
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from datasets import load_dataset, load_from_disk, Audio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import assemblyai
from elevenlabs.client import ElevenLabs

# Contexts of transcriptions class
class TranscriptionContext:
    def __init__(self, hf_model, hf_processor, ass_config, el_client, device, logger):
        self.hf_model = hf_model
        self.hf_processor = hf_processor
        self.ass_config = ass_config
        self.el_client = el_client
        self.device = device
        self.logger = logger

### Functions

def setup_logger(log_file: str = "transcriptions.log") -> logging.Logger:
    """
    Configure and return a logger with both file and console output.

    This logger writes logs to a specified file.
    If the log file directory does not exist, it will be created automatically.

    Args:
        log_file (str, optional): Path to the log file. Defaults to "transcriptions.log".

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger("transcription")
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if the logger is reused
    if not logger.handlers:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # File handler
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)

        # Log format
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(fh)

    return logger
##

## Whisper, ElevenLabs, & AssemblyAI
# ---- BLOCKING PARTS IN THREAD ----
def run_librosa_and_hf(hf_audio_filepath: str, ctx: TranscriptionContext) -> str:
    """
    Transcribe an audio file using Hugging Face Whisper model.

    Args:
        hf_audio_filepath (str): Path to the audio file.
        ctx (TranscriptionContext): Context object containing all shared resources and configurations, including:
            - ctx.hf_model (AutoModelForSpeechSeq2Seq): Pretrained Hugging Face Whisper model.
            - ctx.hf_processor (AutoProcessor): Processor for the Hugging Face model.
            - ctx.ass_config (assemblyai.TranscriptionConfig): Configuration for AssemblyAI transcription.
            - ctx.el_client (ElevenLabs): ElevenLabs API client.
            - ctx.device (str): Device for model inference ("cpu" or "cuda").
            - ctx.logger (logging.Logger) : Log file for the process.

    Returns:
        str: Transcribed text, or an error message if transcription fails.
    """
    try:
        audio, sr = librosa.load(hf_audio_filepath, sr=16000)
        input_features = ctx.hf_processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = input_features.input_features.to(ctx.device)
        attention_mask = torch.ones_like(input_features).to(ctx.device)

        predicted_ids = ctx.hf_model.generate(input_features, language="pl",
                                              attention_mask=attention_mask)
        transcription = ctx.hf_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    except Exception as e:
        transcription = f"Error transcription: {e}"

    return transcription


def run_assemblyai(ass_audio_filepath: str, ctx: TranscriptionContext) -> str:
    """
    Transcribe an audio file using AssemblyAI API.

    Args:
        ass_audio_filepath (str): Path to the audio file.
        ctx (TranscriptionContext): Context object containing all shared resources and configurations, including:
            - ctx.hf_model (AutoModelForSpeechSeq2Seq): Pretrained Hugging Face Whisper model.
            - ctx.hf_processor (AutoProcessor): Processor for the Hugging Face model.
            - ctx.ass_config (assemblyai.TranscriptionConfig): Configuration for AssemblyAI transcription.
            - ctx.el_client (ElevenLabs): ElevenLabs API client.
            - ctx.device (str): Device for model inference ("cpu" or "cuda").
            - ctx.logger (logging.Logger) : Log file for the process.

    Returns:
        str: Transcribed text, or an error message if transcription fails.
    """
    try:
        transcriber = assemblyai.Transcriber()
        transcript = transcriber.transcribe(ass_audio_filepath, ctx.ass_config)
        if transcript.status == assemblyai.TranscriptStatus.error:
            transcription = f"Transcription failed: {transcript.error}"
        else:
            transcription = transcript.text
    except Exception as e:
        transcription = f"Error transcription: {e}"

    return transcription


def run_elevenlabs(el_audio_filepath: str, ctx: TranscriptionContext) -> str:
    """
    Transcribe an audio file using ElevenLabs API.

    Args:
        el_audio_filepath (str): Path to the audio file.
        ctx (TranscriptionContext): Context object containing all shared resources and configurations, including:
            - ctx.hf_model (AutoModelForSpeechSeq2Seq): Pretrained Hugging Face Whisper model.
            - ctx.hf_processor (AutoProcessor): Processor for the Hugging Face model.
            - ctx.ass_config (assemblyai.TranscriptionConfig): Configuration for AssemblyAI transcription.
            - ctx.el_client (ElevenLabs): ElevenLabs API client.
            - ctx.device (str): Device for model inference ("cpu" or "cuda").
            - ctx.logger (logging.Logger) : Log file for the process.

    Returns:
        str: Transcribed text, or an error message if transcription fails.
    """
    try:
        audio, sr = librosa.load(el_audio_filepath, sr=16000)
        tmp_buffer = BytesIO()
        sf.write(tmp_buffer, audio, sr, format="WAV")
        tmp_buffer.seek(0)

        el_clt = ctx.el_client
        resp = el_clt.speech_to_text.convert(
            file=tmp_buffer,
            model_id="scribe_v1",
            language_code="pol",
            diarize=False,
            timestamps_granularity="word",
            tag_audio_events=False
        )
        transcription = resp.text  # Access text using .text attribute
    except Exception as e:
        transcription = f"Error transcription: {e}"

    return transcription


# ---- MAIN ASYNC FUNCTION ----
async def transcribe_audio_async(audio_filepath: str, executor: ThreadPoolExecutor,
                                 ctx: TranscriptionContext) -> dict:
    """
    Transcribe an audio file using Whisper (HF), AssemblyAI, and ElevenLabs concurrently.

    Args:
        audio_filepath (str): Path to the audio file.
        executor (ThreadPoolExecutor): Executor to run blocking tasks.
        ctx (TranscriptionContext): Context object containing all shared resources and configurations, including:
            - ctx.hf_model (AutoModelForSpeechSeq2Seq): Pretrained Hugging Face Whisper model.
            - ctx.hf_processor (AutoProcessor): Processor for the Hugging Face model.
            - ctx.ass_config (assemblyai.TranscriptionConfig): Configuration for AssemblyAI transcription.
            - ctx.el_client (ElevenLabs): ElevenLabs API client.
            - ctx.device (str): Device for model inference ("cpu" or "cuda").
            - ctx.logger (logging.Logger) : Log file for the process.

    Returns:
        dict: Transcriptions and elapsed time for each service.
    """
    loop = asyncio.get_event_loop()

    def run_with_timer(func, fp, label, ctx):
        start_time = time.perf_counter()
        transcription = func(fp, ctx)
        elapsed = time.perf_counter() - start_time
        ctx.logger.info(f"[{label}] Done transcription: {fp} (took {elapsed:.2f}s)")
        return transcription, elapsed

    # Launch tasks concurrently
    hf_future = loop.run_in_executor(executor, run_with_timer, run_librosa_and_hf,
                                     audio_filepath, "Whisper", ctx)
    ass_future = loop.run_in_executor(executor, run_with_timer, run_assemblyai,
                                      audio_filepath, "AssemblyAI", ctx)
    el_future = loop.run_in_executor(executor, run_with_timer, run_elevenlabs,
                                     audio_filepath, "ElevenLabs", ctx)

    hf_transcription, hf_time = await hf_future
    ass_transcription, ass_time = await ass_future
    el_transcription, el_time = await el_future

    return {
        "file_path": audio_filepath,
        "hf_transcription": hf_transcription,
        "hf_time": hf_time,
        "ass_transcription": ass_transcription,
        "ass_time": ass_time,
        "el_transcription": el_transcription,
        "el_time": el_time
    }


async def process_all_with_progress(list_file_paths: list[str], ctx: TranscriptionContext,
                                    max_workers: int = 5) -> list[dict]:
    """
    Transcribe multiple audio files concurrently with progress tracking.

    Args:
        list_file_paths (list[str]): List of audio file paths.
        ctx (TranscriptionContext): Context object containing all shared resources and configurations, including:
            - ctx.hf_model (AutoModelForSpeechSeq2Seq): Pretrained Hugging Face Whisper model.
            - ctx.hf_processor (AutoProcessor): Processor for the Hugging Face model.
            - ctx.ass_config (assemblyai.TranscriptionConfig): Configuration for AssemblyAI transcription.
            - ctx.el_client (ElevenLabs): ElevenLabs API client.
            - ctx.device (str): Device for model inference ("cpu" or "cuda").
            - ctx.logger (logging.Logger) : Log file for the process.
        max_workers (int, optional): Maximum number of worker threads. Defaults to 5.

    Returns:
        list[dict]: List of transcription results for each audio file.
    """
    start_time_total = time.perf_counter()
    results = [None] * len(list_file_paths)  # Pre-allocation

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        async def indexed_task(idx, fp):
            start_time_file = time.perf_counter()
            res = await transcribe_audio_async(fp, executor, ctx)
            elapsed_file = time.perf_counter() - start_time_file
            ctx.logger.info(f"[TOTAL] Finished {fp} in {elapsed_file:.2f}s")
            return idx, res

        tasks = [indexed_task(i, fp) for i, fp in enumerate(list_file_paths)]

        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Transcriptions"):
            idx, res = await fut
            results[idx] = res

    elapsed_total = time.perf_counter() - start_time_total
    ctx.logger.info(f"[TOTAL] All files processed: {len(list_file_paths)} files in {elapsed_total:.2f}s")
    ctx.logger.info(f"[TOTAL] Average per file: {elapsed_total / len(list_file_paths):.2f}s")

    return results
##

# The async main function to run the processing
async def main(logger: logging.Logger):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"*** Device: {device}")

    ### Configurations
    load_dotenv()

    el_client = ElevenLabs(api_key=os.getenv('elevenlabs_api_key'))

    assemblyai.settings.api_key = os.getenv('assemblyai_api_key')
    ass_config = assemblyai.TranscriptionConfig(language_code="pl",
                                                speech_model=assemblyai.SpeechModel.best)

    model_id = "openai/whisper-large-v3-turbo"
    hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, low_cpu_mem_usage=True, use_safetensors=True, token=os.getenv("HF_TOKEN")
    ).to(device)
    hf_processor = AutoProcessor.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))


    ctx = TranscriptionContext(hf_model, hf_processor, ass_config, el_client, device, logger)

    # Define the path to the results directory
    directory_path = os.getenv("RESULTS_DIR", "./results")
    os.makedirs(directory_path, exist_ok=True)

    ### Load data

    # Define the path to the directory in the Drive
    data_path = os.getenv("DATA_PATH", "./data/common_voice_pl_test")
    # Convert to Path object for convenience
    data_path = Path(data_path)
    # Ensure parent directory exists
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if the path exists
    if not data_path.exists():
        # Create from Hugging Face datasets
        cv_pl_test = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            "pl",
            split="test",
        token=os.getenv("HF_TOKEN"),
        trust_remote_code=True # added
        )
        cv_pl_test.save_to_disk(data_path)
    else:
        cv_pl_test = load_from_disk(data_path)

    cv_pl_test = cv_pl_test.cast_column("audio", Audio(decode=False))

    # Batch configuration
    num_rows = len(cv_pl_test)
    batch_size = 100
    num_batches = math.ceil(num_rows / batch_size)

    start_batch = 0

    ### Process for Whisper, ElevenLabs, AssemblyAI

    for i in range(start_batch, num_batches):
        start_idx = i * batch_size
        end_idx = min(((i + 1) * batch_size), num_rows)

        # Get the audio datas and save them locally
        mp3_folder = Path(f"./mp3_batch_{i}")
        mp3_folder.mkdir(parents=True, exist_ok=True)

        list_path = []
        list_sentence = []

        # Process the batch from dataset
        for ex in cv_pl_test.select(range(start_idx, end_idx)):
            try:
                audio_path = ex['audio']['path']
                sentence = ex['sentence']
                save_path = mp3_folder / Path(audio_path).name
                with open(save_path, "wb") as f:
                    f.write(ex['audio']['bytes'])
                list_path.append(save_path)
                list_sentence.append(sentence)
            except Exception as e:
                logger.error(f"Error processing file {ex}: {e}") # Log the error and continue

        logger.info("%s loaded", mp3_folder)

        df = pd.DataFrame({
            'file_path': list_path,
            'sentence': list_sentence
        })

        # Process the transcriptions
        results = await process_all_with_progress(list_path, ctx, max_workers=5,)

        df_results = pd.DataFrame(results)
        df_results["total_time"] = df_results[["hf_time", "ass_time", "el_time"]].sum(axis=1)

        # Merge results with batch dataframe
        df_batch = df.merge(df_results, on='file_path', how='left')

        # Save the batch results in a csv file
        csv_filename = Path(directory_path) / f"batch_{i+1}.csv"
        df_batch.to_csv(csv_filename, sep=";", index=False)

        logger.info(f"Line {start_idx} to {end_idx} saved in {csv_filename}")

        # Remove mp3 folder
        if mp3_folder.exists() and mp3_folder.is_dir():
            shutil.rmtree(mp3_folder)


### MAIN PROCESS

if __name__ == "__main__":
    logger = setup_logger(os.getenv("LOG_FILE", "transcriptions.log"))
    asyncio.run(main(logger))
