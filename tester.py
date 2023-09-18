def write_movie_srt(wsm, srt_file):
    max_words_per_entry = 12  # Maximum number of words per subtitle entry
    current_words = []  # Words for the current subtitle entry
    current_start_time = None  # Start time for the current subtitle entry
    current_speaker = None  # Speaker for the current subtitle entry
    entry_count = 1  # Subtitle entry count
    last_end_time = 0  # Last end time to prevent overlaps

    for entry in wsm:
        word = entry['word']
        start_time = entry['start_time']
        end_time = entry['end_time']
        speaker = entry['speaker']

        # Prevent overlap by adjusting the start_time
        if start_time < last_end_time:
            start_time = last_end_time

        if current_start_time is None:
            current_start_time = start_time

        if current_speaker is None:
            current_speaker = speaker

        # Create a new subtitle entry if the word limit is reached or the speaker changes
        if len(current_words) >= max_words_per_entry or current_speaker != speaker:
            srt_file.write(f"{entry_count}\n")
            srt_file.write(f"{format_time(current_start_time)} --> {format_time(end_time)}\n")
            srt_file.write(f"Speaker {current_speaker}: {' '.join(current_words)}\n\n")
            entry_count += 1
            current_words = []
            current_start_time = start_time
            current_speaker = speaker
            last_end_time = end_time  # Update the last end time

        current_words.append(word)

    # Write the last subtitle entry
    if current_words:
        srt_file.write(f"{entry_count}\n")
        srt_file.write(f"{format_time(current_start_time)} --> {format_time(end_time)}\n")
        srt_file.write(f"Speaker {current_speaker}: {' '.join(current_words)}\n")


# Helper function to format time in SRT format
def format_time(time_in_milliseconds):
    seconds, milliseconds = divmod(time_in_milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


# Example usage
wsm = [{'word': 'animals,', 'start_time': 603923, 'end_time': 604304, 'speaker': 0}, {'word': 'try', 'start_time': 604324, 'end_time': 604565, 'speaker': 0}, {'word': 'to', 'start_time': 604987, 'end_time': 605167, 'speaker': 0}, {'word': 'get', 'start_time': 605187, 'end_time': 605288, 'speaker': 0}, {'word': 'them', 'start_time': 605308, 'end_time': 605428, 'speaker': 0}, {'word': 'to', 'start_time': 605448, 'end_time': 605549, 'speaker': 0}, {'word': 'interact,', 'start_time': 605629, 'end_time': 606472, 'speaker': 0}, {'word': 'better', 'start_time': 606814, 'end_time': 607115, 'speaker': 0}, {'word': 'their', 'start_time': 607135, 'end_time': 607597, 'speaker': 0}, {'word': 'mental', 'start_time': 607617, 'end_time': 607938, 'speaker': 0}, {'word': 'wellbeing.', 'start_time': 607958, 'end_time': 608320, 'speaker': 0}, {'word': 'Similar', 'start_time': 609621, 'end_time': 610045, 'speaker': 0}, {'word': 'to', 'start_time': 610085, 'end_time': 610287, 'speaker': 0}, {'word': 'the', 'start_time': 610327, 'end_time': 610448, 'speaker': 0}, {'word': 'herd', 'start_time': 610468, 'end_time': 610811, 'speaker': 0}, {'word': 'event', 'start_time': 611174, 'end_time': 611476, 'speaker': 0}, {'word': 'that', 'start_time': 611496, 'end_time': 611617, 'speaker': 0}, {'word': 'we', 'start_time': 611637, 'end_time': 611738, 'speaker': 0}, {'word': 'did.', 'start_time': 611758, 'end_time': 611879, 'speaker': 0}, {'word': 'Yeah.', 'start_time': 612900, 'end_time': 613060, 'speaker': 0}, {'word': 'We', 'start_time': 616242, 'end_time': 616544, 'speaker': 0}, {'word': 'are', 'start_time': 616645, 'end_time': 616765, 'speaker': 0}, {'word': 'doing', 'start_time': 617128, 'end_time': 617410, 'speaker': 0}, {'word': 'one', 'start_time': 617510, 'end_time': 617631, 'speaker': 0}, {'word': 'art', 'start_time': 617893, 'end_time': 617974, 'speaker': 0}, {'word': 'jamming', 'start_time': 618014, 'end_time': 618356, 'speaker': 0}, {'word': 'with', 'start_time': 618396, 'end_time': 618517, 'speaker': 0}, {'word': 'cats.', 'start_time': 618557, 'end_time': 618779, 'speaker': 0}, {'word': 'Yeah.', 'start_time': 620816, 'end_time': 620900, 'speaker': 0}, {'word': 'Soon.', 'start_time': 622160, 'end_time': 622284, 'speaker': 0}, {'word': 'When', 'start_time': 623402, 'end_time': 623627, 'speaker': 0}, {'word': 'we', 'start_time': 623668, 'end_time': 623812, 'speaker': 0}, {'word': 'have', 'start_time': 623832, 'end_time': 623935, 'speaker': 0}, {'word': 'time.', 'start_time': 623976, 'end_time': 624119, 'speaker': 0}, {'word': 'Yeah.', 'start_time': 625242, 'end_time': 625377, 'speaker': 0}, {'word': 'Okay,', 'start_time': 628026, 'end_time': 628292, 'speaker': 0}, {'word': 'good.', 'start_time': 628313, 'end_time': 628600, 'speaker': 0}, {'word': 'Hi,', 'start_time': 628760, 'end_time': 629063, 'speaker': 0}, {'word': 'nice', 'start_time': 629083, 'end_time': 629791, 'speaker': 0}, {'word': 'to', 'start_time': 629913, 'end_time': 630115, 'speaker': 0}, {'word': 'meet', 'start_time': 630176, 'end_time': 630358, 'speaker': 0}, {'word': 'you.', 'start_time': 630378, 'end_time': 630439, 'speaker': 0}, {'word': 'Hi,', 'start_time': 631502, 'end_time': 631644, 'speaker': 0}, {'word': 'sorry', 'start_time': 631847, 'end_time': 632050, 'speaker': 0}, {'word': 'to', 'start_time': 632091, 'end_time': 632193, 'speaker': 0}, {'word': 'interrupt.', 'start_time': 632274, 'end_time': 632599, 'speaker': 0}, {'word': 'Nice', 'start_time': 633041, 'end_time': 633485, 'speaker': 0}, {'word': 'to', 'start_time': 633526, 'end_time': 633707, 'speaker': 0}, {'word': 'meet', 'start_time': 633748, 'end_time': 633970, 'speaker': 0}, {'word': 'you.', 'start_time': 634010, 'end_time': 634111, 'speaker': 0}, {'word': 'I', 'start_time': 635485, 'end_time': 635545, 'speaker': 0}, {'word': 'heard', 'start_time': 635586, 'end_time': 635828, 'speaker': 0}, {'word': 'about', 'start_time': 635889, 'end_time': 636192, 'speaker': 0}, {'word': 'the', 'start_time': 636212, 'end_time': 636273, 'speaker': 0}, {'word': 'project.', 'start_time': 636536, 'end_time': 636839, 'speaker': 0}, {'word': 'Okay,', 'start_time': 637221, 'end_time': 637504, 'speaker': 0}, {'word': 'then', 'start_time': 637787, 'end_time': 637969, 'speaker': 0}, {'word': 'this', 'start_time': 638029, 'end_time': 638716, 'speaker': 0}, {'word': 'is', 'start_time': 638797, 'end_time': 638898, 'speaker': 0}, {'word': 'mine.', 'start_time': 638918, 'end_time': 639080, 'speaker': 0}]

srt = "test.srt"
with open(srt, "w", encoding="utf-8-sig") as s:
    write_movie_srt(wsm, s)