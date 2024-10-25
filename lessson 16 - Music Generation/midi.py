"""
File: midi.py
Author: Addy771
Description: 
A script which converts MIDI files to WAV and optionally to MP3 using ffmpeg. 
Works by playing each file and using the stereo mix device to record at the same time
"""


import pyaudio  # audio recording
import wave     # file saving
import pygame   # midi playback
import os       # file listing
from music21 import stream, chord



#### CONFIGURATION ####

do_ffmpeg_convert = True    # Uses FFmpeg to convert WAV files to MP3. Requires ffmpeg.exe in the script folder or PATH
do_wav_cleanup = True       # Deletes WAV files after conversion to MP3
sample_rate = 44100         # Sample rate used for WAV/MP3
channels = 2                # Audio channels (1 = mono, 2 = stereo)
buffer = 1024               # Audio buffer size
mp3_bitrate = 128           # Bitrate to save MP3 with in kbps (CBR)
input_device = 1            # Which recording device to use. On my system Stereo Mix = 1



# Begins playback of a MIDI file
def play_music(music_file):

    try:
        pygame.mixer.music.load(music_file)
        
    except pygame.error:
        print ("Couldn't play %s! (%s)" % (music_file, pygame.get_error()))
        return
        
    pygame.mixer.music.play()



def convert_mid_wav(file_path):
    # Init pygame playback
    bitsize = -16   # unsigned 16 bit
    pygame.mixer.init(sample_rate, bitsize, channels, buffer)

    # optional volume 0 to 1.0
    pygame.mixer.music.set_volume(1.0)

    # Init pyAudio
    format = pyaudio.paInt16
    audio = pyaudio.PyAudio()



    try:
        # Create a filename with a .wav extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        new_file = file_name + '.wav'

        # Open the stream and start recording
        stream = audio.open(format=format, channels=channels, rate=sample_rate, input=True, input_device_index=input_device, frames_per_buffer=buffer)
        
        # Playback the song
        print("Playing " + file_name + ".mid\n")
        play_music(file_path)
        
        frames = []
        
        # Record frames while the song is playing
        while pygame.mixer.music.get_busy():
            frames.append(stream.read(buffer))
            
        # Stop recording
        stream.stop_stream()
        stream.close()

        
        # Configure wave file settings
        wave_file = wave.open(new_file, 'wb')
        wave_file.setnchannels(channels)
        wave_file.setsampwidth(audio.get_sample_size(format))
        wave_file.setframerate(sample_rate)
        
        print("Saving " + new_file)   
        
        # Write the frames to the wave file
        wave_file.writeframes(b''.join(frames))
        wave_file.close()
        
        # Call FFmpeg to handle the MP3 conversion if desired
        if do_ffmpeg_convert:
            os.system('ffmpeg -i ' + new_file + ' -y -f mp3 -ab ' + str(mp3_bitrate) + 'k -ac ' + str(channels) + ' -ar ' + str(sample_rate) + ' -vn ' + file_name + '.mp3')
            
            # Delete the WAV file if desired
            if do_wav_cleanup:        
                os.remove(new_file)
            
        # End PyAudio    
        audio.terminate()    

        return new_file
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        # (works only in console mode)
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit
    
def int_to_note(int_value):
    """
    Function to convert an integer value to a note name.
    """
    # Map integer values to note names (C2 = 36, C3 = 48, ..., A5 = 81)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Calculate octave and note index
    octave = (int_value - 24) // 12
    note_index = (int_value - 24) % 12
    
    # Construct the note name
    note_name = note_names[note_index] + str(octave + 2)
    
    return note_name

def create_music_from_chord_data(chord_data, write_path):
    """
    Function to create music from a list of chords represented as lists of integer notes.
    """
    # Create a new music stream
    music_stream = stream.Stream()

    # Iterate through each chord in the chord data
    for chord_notes in chord_data:
        # Convert each integer note representation to a note name
        notes = [int_to_note(int_note) for int_note in chord_notes]

        # Create a chord object
        new_chord = chord.Chord(notes)

        # Add the chord to the music stream
        music_stream.append(new_chord)

    music_stream.write('midi', fp=write_path)