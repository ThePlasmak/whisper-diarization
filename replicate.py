import replicate

output = replicate.run(
    "thomasmol/whisper-diarization:3ff22700b10e9c888e72235131e10c0a8427cd79e750bc42e4c035be2121796b",
    input={
        "file": "/mnt/e/10_OUTGOING/01_INTERNAL/Master V0.4_audio.wma",
        "prompt": "",
        "file_url": "",
        "num_speakers": 2,
        "group_segments": True,
        "offset_seconds": 0,
        "transcript_output_format": "both"
    }
)
print(output)