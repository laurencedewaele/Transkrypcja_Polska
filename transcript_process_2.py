## Imports
import os
import shutil
import math
import time
import logging
import asyncio
from pathlib import Path

import torch
import pandas as pd
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from datasets import load_dataset, load_from_disk, Audio
from speechmatics.batch import AsyncClient, JobConfig, JobType, TranscriptionConfig

# Contexts of transcriptions class
class TranscriptionContext:
    def __init__(self, spm_client, spm_config, device, logger):
        self.spm_client = spm_client
        self.spm_config = spm_config
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

# ---- WRAPPER WITH LOGS ----
async def transcribe_audio(audio_filepath: str, ctx: TranscriptionContext,
                           semaphore: asyncio.Semaphore | None = None) -> dict:
    """
    Wrapper for Speechmatics transcription with optional concurrency control.

    Args:
        audio_filepath (str): Path to the audio file.
        ctx (TranscriptionContext): Context object containing all shared resources and configurations, including:
            - ctx.spm_client (AsyncClient): Speechmatics async client.
            - ctx.device (str): Device for model inference ("cpu" or "cuda").
            - ctx.logger (logging.Logger) : Log file for the process.
        semaphore (asyncio.Semaphore | None, optional): Semaphore for concurrency limiting.

    Returns:
        dict: Transcription result and elapsed time.
    """
# ---- SINGLE FILE TRANSCRIPTION ----
    async def run_spm(spm_audio_filepath: str) -> str:
        """
        Transcribe an audio file using Speechmatics API.

        Args:
            spm_audio_filepath (str): Path to the audio file.

        Returns:
            str: Transcribed text, or an error message if transcription fails.
        """
        spm_transcription = ""
        try:
            # ⚡️ async calls to Speechmatics client
            spm_clt = ctx.spm_client
            job = await spm_clt.submit_job(spm_audio_filepath, config=ctx.spm_config)
            result = await spm_clt.wait_for_completion(job.id)

            transcription = result.transcript_text.removeprefix("SPEAKER UU: ")
            spm_transcription = transcription

        except Exception as e:
            spm_transcription = f"Error transcription: {e}"

        return spm_transcription


    async def run_with_timer(fp):
        start_time = time.perf_counter()
        transcription = await run_spm(fp)
        elapsed = time.perf_counter() - start_time
        ctx.logger.info(f"[SPM] Done transcription: {fp} (took {elapsed:.2f}s)")
        return transcription, elapsed

    if semaphore:
        async with semaphore:
            spm_transcription, spm_time = await run_with_timer(audio_filepath)
    else:
        spm_transcription, spm_time = await run_with_timer(audio_filepath)

    return {
        "file_path": audio_filepath,
        "spm_transcription": spm_transcription,
        "spm_time": spm_time
    }


# ---- BATCH PROCESSING ----
async def process_spm(list_file_paths: list[str], ctx: TranscriptionContext,
                      max_concurrent: int = 5) -> list[dict]:
    """
    Transcribe multiple audio files concurrently using Speechmatics.

    Args:
        list_file_paths (list[str]): List of audio file paths.
        ctx (TranscriptionContext): Context object containing all shared resources and configurations, including:
            - ctx.spm_client (AsyncClient): Speechmatics async client.
            - ctx.device (str): Device for model inference ("cpu" or "cuda").
            - ctx.logger (logging.Logger) : Log file for the process.
        max_concurrent (int, optional): Maximum number of concurrent Speechmatics requests. Defaults to 5.

    Returns:
        list[dict]: List of transcription results for each audio file.
    """
    start_time_total = time.perf_counter()
    results = [None] * len(list_file_paths)  # Pre-allocation
    semaphore = asyncio.Semaphore(max_concurrent)

    async def indexed_task(idx, fp):
        res = await transcribe_audio(fp, ctx, semaphore=semaphore)
        return idx, res

    # ✅ create tasks (not just coroutines)
    tasks = [asyncio.create_task(indexed_task(i, fp)) for i, fp in enumerate(list_file_paths)]

    # Parallel execution with progress bar
    async for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Transcriptions"):
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

    spm_client = AsyncClient(api_key=os.getenv('speechmatics_api_key'))
    spm_config = JobConfig(
        type=JobType.TRANSCRIPTION,
        transcription_config=TranscriptionConfig(language="pl", diarization=None))

    ctx = TranscriptionContext(spm_client, spm_config, device, logger)

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

    ### Process for SpeechMatics

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
        results = await process_spm(list_path, ctx, 10)

        df_results = pd.DataFrame(results)

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
