import argparse
import os
from helpers import *
from faster_whisper import WhisperModel
import whisperx
import torch
import librosa
import soundfile
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
import logging
import time

start_time = time.time()

mtypes = {'cpu': 'int8', 'cuda': 'float16'}

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio",
    dest="audio",
    required=True,
    type=str,
    help="Name of the target audio file.",
)
parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=True,
    help="Disables source separation."
    "This helps with long files that don't contain a lot of music.",
)
parser.add_argument(
    "-m", "--whisper-model",
    dest="model_name",
    type=str,
    default="large-v2",
    help="Name of the Whisper model to use."
    "Select from this list: 'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large'.",
)
parser.add_argument(
    "--device",
    dest="device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="If you have a GPU use 'cuda', otherwise use 'cpu'.",
)
parser.add_argument(
    "-s","--speakers",
    dest="num_speakers",
    type=int,
    default=None,
    help="Enter the number of speakers.",
)
parser.add_argument(
    "-l", "--language",
    dest="language",
    type=str,
    default=None,
    help="Enter the language (in an ISO 639-1 Code).",
)
parser.add_argument(
    "-bs", "--beam-size",
    dest="beam_size",
    type=int,
    default=1,
    help="Enter the desired beam size."
    "The higher the beam size, the more accurate the transcription, but the slower the process and the higher the likelihood of running out of memory.",
)
parser.add_argument(
    "-dt", "--domain-type",
    dest="domain_type",
    type=str,
    default="telephonic", # Can be "general", "meeting" or "telephonic" based on domain type of the audio file (see https://github.com/NVIDIA/NeMo/tree/main/examples/speaker_tasks/diarization/conf/inference)
    help="Enter the desired domain type."
    "general: optimized to show balanced performances on various types of domains."
    "meeting: suitable for 3~5 speakers participating in a meeting and may not show the best performance on other types of dialogues."
    "telephonic: suitable for telephone recordings involving 2~8 speakers in a session and may not show the best performance on the other types of acoustic conditions or dialogues.",
)
parser.add_argument(
    "-sd", "--speaker-duration",
    dest="speaker_duration",
    type=float,
    default=None,
    help="Duration (in seconds) to split long segments of a single speaker talking. (The script will cut a little bit over the duration.)"
)

args = parser.parse_args()

if args.audio.endswith('.webm'):
    new_filename = args.audio.replace('.webm', '.wav')
    os.system(f'ffmpeg -i {args.audio} {new_filename}')

    audio = new_filename
else:
    audio = args.audio

if args.stemming:
    # Isolate vocals from the rest of the audio
    return_code = os.system(
        f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{audio}" -o "temp_outputs"'
    )

    if return_code != 0:
        logging.warning(
            "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
        )
        vocal_target = audio
    else:
        vocal_target = os.path.join(
            "temp_outputs", "htdemucs", os.path.basename(audio[:-4]), "vocals.wav"
        )
else:
    vocal_target = audio


# Run on GPU with FP16
whisper_model = WhisperModel(
    args.model_name, device=args.device, compute_type=mtypes[args.device])

# or run on GPU with INT8
# whisper_model = WhisperModel(args.model_name, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# whisper_model = WhisperModel(args.model_name, device="cpu", compute_type="int8")

transcribe_args = {
    "beam_size": args.beam_size,
    "word_timestamps": True
}
if args.language:
    transcribe_args["language"] = args.language

segments, info = whisper_model.transcribe(vocal_target, **transcribe_args)

whisper_results = []
for segment in segments:
    whisper_results.append(segment._asdict())

# clear gpu vram
del whisper_model
torch.cuda.empty_cache()

language_to_check = args.language if args.language else info.language
if language_to_check in wav2vec2_langs:
    alignment_model, metadata = whisperx.load_align_model(
        language_code=language_to_check, device=args.device
    )
    result_aligned = whisperx.align(
        whisper_results, alignment_model, metadata, vocal_target, args.device
    )
    word_timestamps = result_aligned["word_segments"]
    # clear gpu vram
    del alignment_model
    torch.cuda.empty_cache()
else:
    word_timestamps = []
    for segment in whisper_results:
        for word in segment["words"]:
            word_timestamps.append({"text": word[2], "start": word[0], "end": word[1]})


# convert audio to mono for NeMo combatibility
signal, sample_rate = librosa.load(vocal_target, sr=None)
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
soundfile.write(os.path.join(temp_path, "mono_file.wav"), signal, sample_rate, "PCM_24")

# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path, args.domain_type, args.num_speakers)).to(args.device)
msdd_model.diarize()

del msdd_model
torch.cuda.empty_cache()

# Reading timestamps <> Speaker Labels mapping


speaker_ts = []
with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

if info.language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences
    punct_model = PunctuationModel(model="kredor/punctuate-all")

    words_list = list(map(lambda x: x["word"], wsm))

    labled_words = punct_model.predict(words_list)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"

    # We don't want to punctuate U.S.A. with a period. Right?
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

    for word_dict, labeled_tuple in zip(wsm, labled_words):
        word = word_dict["word"]
        if (
            word
            and labeled_tuple[1] in ending_puncts
            and (word[-1] not in model_puncts or is_acronym(word))
        ):
            word += labeled_tuple[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
else:
    logging.warning(
        f'Punctuation restoration is not available for {language_to_check} language.'
    )

if args.speaker_duration is not None: # then split periods where a single speaker talks into chunks
    speaker_duration_ms = args.speaker_duration*1000
    split_segments = split_word_segments_by_duration(wsm, speaker_duration_ms)

    with open(f"{audio[:-4]}.txt", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(split_segments, f)

    with open(f"{audio[:-4]}.srt", "w", encoding="utf-8-sig") as srt:
        write_srt(split_segments, srt)
else: # no splitting
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    with open(f"{audio[:-4]}.txt", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(f"{audio[:-4]}.srt", "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

cleanup(temp_path)
if new_filename:
    os.remove(new_filename)

# Time calculation
time_taken_seconds = time.time() - start_time # Calculate the time taken in seconds
time_taken_formatted = "{:.3f}".format(time_taken_seconds) # Format the time taken to three decimal places

# Calculate hours, minutes, and seconds
hours = int(time_taken_seconds // 3600)
minutes = int((time_taken_seconds % 3600) // 60)
seconds = int(time_taken_seconds % 60)

# Print the time taken in hours, minutes, and seconds format with three decimal places
print(
    "Time taken: {} hours, {} minutes, {} seconds".format(
        hours, minutes, time_taken_formatted
    )
)
