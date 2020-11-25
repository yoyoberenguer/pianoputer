#!/usr/bin/env python
import numpy
from scipy.io import wavfile
import argparse
import numpy as np
import pygame
import sys
import warnings

from FireEffect import make_palette, fire_surface24


def speedx(snd_array, factor):
    """ Speeds up / slows down a sound, by some factor. """
    indices = np.round(np.arange(0, len(snd_array), factor))
    indices = indices[indices < len(snd_array)].astype(int)
    return snd_array[indices]


def stretch(snd_array, factor, window_size, h):
    """ Stretches/shortens a sound, by some factor. """
    phase = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros(int(len(snd_array) / factor + window_size))

    for i in np.arange(0, len(snd_array) - (window_size + h), h*factor):
        i = int(i)
        # Two potentially overlapping subarrays
        a1 = snd_array[i: i + window_size]
        a2 = snd_array[i + h: i + window_size + h]

        # The spectra of these arrays
        s1 = np.fft.fft(hanning_window * a1)
        s2 = np.fft.fft(hanning_window * a2)

        # Rephase all frequencies
        phase = (phase + np.angle(s2/s1)) % 2*np.pi

        a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase))
        i2 = int(i/factor)
        result[i2: i2 + window_size] += hanning_window*a2_rephased.real

    # normalize (16bit)
    result = ((2**(16-4)) * result/result.max())

    return result.astype('int16')


def pitchshift(snd_array, n, window_size=2**13, h=2**11):
    """ Changes the pitch of a sound by ``n`` semitones. """
    factor = 2**(1.0 * n / 12.0)
    stretched = stretch(snd_array, 1.0/factor, window_size, h)
    return speedx(stretched[window_size:], factor)


def parse_arguments():
    description = ('Use your computer keyboard as a "piano"')

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--wav', '-w',
        metavar='FILE',
        type=argparse.FileType('r'),
        default='bowl.wav',
        help='WAV file (default: bowl.wav)')
    parser.add_argument(
        '--keyboard', '-k',
        metavar='FILE',
        type=argparse.FileType('r'),
        default='typewriter.kb',
        help='keyboard file (default: typewriter.kb)')
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='verbose mode')

    return (parser.parse_args(), parser)


def main():
    # Parse command line arguments
    (args, parser) = parse_arguments()

    # Enable warnings from scipy if requested
    if not args.verbose:
        warnings.simplefilter('ignore')

    fps, sound = wavfile.read(args.wav.name)

    tones = range(-25, 25)
    sys.stdout.write('Transponding sound file... ')
    sys.stdout.flush()
    transposed_sounds = [pitchshift(sound, n) for n in tones]
    print('DONE')

    # So flexible ;)
    if int(pygame.version.ver[0]) >= 2:
        pygame.mixer.init(fps, -16, 1, 2048, allowedchanges=0)
    else:
        pygame.mixer.init(fps, -16, 1, 2048)

    width = 400
    height = 400
    width_2 = width // 2
    height_2 = height // 2
    # GENERATE THE SAMPLING FOR HALF WIDTH & HEIGHT TO SPEED UP THE PROCESS
    palette, surf = make_palette(width_2, height_2 - 150, 4.0, 60, 1.5)
    mask = numpy.full((width_2, height_2), 255, dtype=numpy.uint8)

    # For the focus
    SCREEN = pygame.display.set_mode((width, height))

    keys = args.keyboard.read().split('\n')

    sounds = map(pygame.sndarray.make_sound, transposed_sounds)
    key_sound = dict(zip(keys, sounds))
    is_playing = {k: False for k in keys}

    fire = numpy.zeros((height, width), dtype=numpy.float32)
    empty_x2 = pygame.Surface((width, height)).convert()

    # pygame.event.set_grab(True)
    CLOCK = pygame.time.Clock()
    FRAME = 0
    while True:
        pygame.display.set_caption("PianoComputer %s fps" % round(CLOCK.get_fps(), 2))
        # event = pygame.event.wait()
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type in (pygame.KEYDOWN, pygame.KEYUP):
                key = pygame.key.name(event.key)

            if event.type == pygame.KEYDOWN:
                if (key in key_sound.keys()) and (not is_playing[key]):
                    key_sound[key].play(fade_ms=50)
                    is_playing[key] = True

                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise KeyboardInterrupt

            elif event.type == pygame.KEYUP and key in key_sound.keys():
                # Stops with 50ms fadeout
                key_sound[key].fadeout(50)
                is_playing[key] = False

        s, o = fire_surface24(width_2, height_2, 3.95, palette, mask, fire)
        pygame.transform.scale2x(s, empty_x2)
        SCREEN.blit(empty_x2, (0, 0))
        fire = o

        CLOCK.tick(300)
        pygame.display.flip()
        FRAME += 1


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Goodbye')
