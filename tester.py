test_name = "tester.mp4"
speaker_duration = 2 # in seconds
speaker_ts = [[460, 4360, 1], [5500, 9240, 2], [10060, 10280, 0], [11900, 12840, 0], [13100, 14280, 0], [14620, 14920, 0]]
wsm = [{'word': 'Now', 'start_time': 582, 'end_time': 783, 'speaker': 1}, {'word': 'listen', 'start_time': 883, 'end_time': 1225, 'speaker': 1}, {'word': 'to', 'start_time': 1265, 'end_time': 1406, 'speaker': 1}, {'word': 'a', 'start_time': 1446, 'end_time': 1466, 'speaker': 1}, {'word': 'conversation', 'start_time': 1546, 'end_time': 2310, 'speaker': 1}, {'word': 'between', 'start_time': 2370, 'end_time': 2691, 'speaker': 1}, {'word': 'a', 'start_time': 2731, 'end_time': 2772, 'speaker': 1}, {'word': 'student', 'start_time': 2832, 'end_time': 3274, 'speaker': 1}, {'word': 'and', 'start_time': 3475, 'end_time': 3595, 'speaker': 1}, {'word': 'a', 'start_time': 3635, 'end_time': 3655, 'speaker': 1}, {'word': 'professor.', 'start_time': 3716, 'end_time': 4258, 'speaker': 1}, {'word': 'So,', 'start_time': 5680, 'end_time': 5820, 'speaker': 2}, {'word': 'Erin,', 'start_time': 5901, 'end_time': 6182, 'speaker': 2}, {'word': 'in', 'start_time': 6444, 'end_time': 6524, 'speaker': 2}, {'word': 'your', 'start_time': 6544, 'end_time': 6705, 'speaker': 2}, {'word': 'email', 'start_time': 6806, 'end_time': 7147, 'speaker': 2}, {'word': 'you', 'start_time': 7188, 'end_time': 7288, 'speaker': 2}, {'word': 'said', 'start_time': 7328, 'end_time': 7489, 'speaker': 2}, {'word': 'you', 'start_time': 7530, 'end_time': 7650, 'speaker': 2}, {'word': 'wanted', 'start_time': 7690, 'end_time': 7932, 'speaker': 2}, {'word': 'to', 'start_time': 7972, 'end_time': 8032, 'speaker': 2}, {'word': 'talk', 'start_time': 8093, 'end_time': 8314, 'speaker': 2}, {'word': 'about', 'start_time': 8374, 'end_time': 8595, 'speaker': 2}, {'word': 'the', 'start_time': 8636, 'end_time': 8736, 'speaker': 2}, {'word': 'exam.', 'start_time': 8796, 'end_time': 9199, 'speaker': 2}, {'word': 'Yeah,', 'start_time': 10180, 'end_time': 11003, 'speaker': 0}, {'word': "I've", 'start_time': 11506, 'end_time': 11646, 'speaker': 0}, {'word': 'just', 'start_time': 11666, 'end_time': 11867, 'speaker': 0}, {'word': 'never', 'start_time': 11988, 'end_time': 12249, 'speaker': 0}, {'word': 'taken', 'start_time': 12390, 'end_time': 12751, 'speaker': 0}, {'word': 'a', 'start_time': 12791, 'end_time': 12831, 'speaker': 0}, {'word': 'class', 'start_time': 12892, 'end_time': 13233, 'speaker': 0}, {'word': 'with', 'start_time': 13274, 'end_time': 13434, 'speaker': 0}, {'word': 'so', 'start_time': 13495, 'end_time': 13675, 'speaker': 0}, {'word': 'many', 'start_time': 13716, 'end_time': 13896, 'speaker': 0}, {'word': 'different', 'start_time': 13937, 'end_time': 14238, 'speaker': 0}, {'word': 'readings.', 'start_time': 14298, 'end_time': 14600, 'speaker': 0}]

def split_word_segments_by_duration(wsm, max_duration):
    segments = []
    segment_start_time = wsm[0]['start_time']
    segment_duration = 0
    segment_text = []
    current_speaker = wsm[0]['speaker']

    for word in wsm:
        word_duration = word['end_time'] - word['start_time']

        # If adding the current word exceeds the max_duration or the speaker changes
        if segment_duration + word_duration > max_duration or word['speaker'] != current_speaker:
            segments.append({
                'speaker': f"Speaker {current_speaker}",
                'start_time': segment_start_time,
                'end_time': word['start_time'],
                'text': ' '.join(segment_text)
            })
            segment_duration = 0
            segment_text = []
            segment_start_time = word['start_time']
            current_speaker = word['speaker']

        segment_duration += word_duration
        segment_text.append(word['word'])

    # Add the last segment
    if segment_text:
        segments.append({
            'speaker': f"Speaker {current_speaker}",
            'start_time': segment_start_time,
            'end_time': wsm[-1]['end_time'],
            'text': ' '.join(segment_text)
        })

    return segments


def get_speaker_aware_transcript(sentences_speaker_mapping, f):
    for sentence_dict in sentences_speaker_mapping:
        sp = sentence_dict["speaker"]
        text = sentence_dict["text"]
        f.write(f"\n\n{sp}: {text}")

def get_sentences_speaker_mapping(word_speaker_mapping, spk_ts):
    s, e, spk = spk_ts[0]
    prev_spk = spk

    snts = []
    snt = {"speaker": f"Speaker {spk}", "start_time": s, "end_time": e, "text": ""}

    for wrd_dict in word_speaker_mapping:
        wrd, spk = wrd_dict["word"], wrd_dict["speaker"]
        s, e = wrd_dict["start_time"], wrd_dict["end_time"]
        if spk != prev_spk:
            snts.append(snt)
            snt = {
                "speaker": f"Speaker {spk}",
                "start_time": s,
                "end_time": e,
                "text": "",
            }
        else:
            snt["end_time"] = e
        snt["text"] += wrd + " "
        prev_spk = spk

    snts.append(snt)
    return snts

def write_srt(transcript, file):
    """
    Write a transcript to a file in SRT format.

    """
    for i, segment in enumerate(transcript, start=1):
        # write srt lines
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start_time'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment['end_time'], always_include_hours=True, decimal_marker=',')}\n"
            f"{segment['speaker']}: {segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )

def format_timestamp(milliseconds, always_include_hours=False, decimal_marker='.'):
    total_seconds = milliseconds / 1000
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds * 1000) % 1000)

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"

if speaker_duration is not None: # then split periods where a single speaker talks into chunks
    speaker_duration_ms = speaker_duration*1000
    split_segments = split_word_segments_by_duration(wsm, speaker_duration_ms)

    for segment in split_segments:
        start_time = segment['start_time']
        end_time = segment['end_time']

        # Find the correct speaker from speaker_ts
        for speaker_time in speaker_ts:
            if start_time >= speaker_time[0] and end_time <= speaker_time[1]:
                segment['speaker'] = f"Speaker {speaker_time[2]}"
                break

    with open(f"{test_name[:-4]}.txt", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(split_segments, f)

    with open(f"{test_name[:-4]}.srt", "w", encoding="utf-8-sig") as srt:
        write_srt(split_segments, srt)

ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

with open(f"{test_name[:-4]}_no_split.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f)

with open(f"{test_name[:-4]}_no_split.srt", "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)