import os
import argparse
import traceback

from sfp import Poller, dummy_file_check
from predict_common import TRANSCRIPTION_FORMATS, TRANSCRIPTION_FORMAT_TEXT, TRANSCRIPTION_FORMAT_JSON, load_model, transcribe

SUPPORTED_EXTS = [".mp3", ".wav"]
""" supported file extensions (lower case). """


def process_audio(fname, output_dir, poller):
    """
    Method for processing an audio file.

    :param fname: the audio file to process
    :type fname: str
    :param output_dir: the directory to write the audio file to
    :type output_dir: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: the list of generated output files
    :rtype: list
    """
    result = []

    try:
        out_format = poller.params.transcription_format
        transcription = transcribe(poller.params.model, fname, beam_size=poller.params.beam_size, transcription_format=out_format)
        if out_format == TRANSCRIPTION_FORMAT_JSON:
            ext = ".json"
        else:
            ext = ".txt"
        fname_out = os.path.join(output_dir, os.path.splitext(os.path.basename(fname))[0] + ext)
        with open(fname_out, "w") as fp:
            fp.write(transcription)
        result.append(fname_out)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process audio file: %s\n%s" % (fname, traceback.format_exc()))
    return result


def predict_on_audio_files(input_dir, model, output_dir, tmp_dir, transcription_format=TRANSCRIPTION_FORMAT_TEXT,
                           poll_wait=1.0, continuous=False, use_watchdog=False, watchdog_check_interval=10.0,
                           delete_input=False, verbose=False, quiet=False, beam_size=5):
    """
    Method for performing predictions on audio files.

    :param input_dir: the directory with the audio files
    :type input_dir: str
    :param model: the whisper model
    :param output_dir: the output directory to move the audio files to and store the predictions
    :type output_dir: str
    :param tmp_dir: the temporary directory to store the predictions until finished, use None if not to use
    :type tmp_dir: str
    :param transcription_format: the format to use for the prediction audio files (grayscale/bluechannel)
    :type transcription_format: str
    :param poll_wait: the amount of seconds between polls when not in watchdog mode
    :type poll_wait: float
    :param continuous: whether to poll continuously
    :type continuous: bool
    :param use_watchdog: whether to react to file creation events rather than use fixed-interval polling
    :type use_watchdog: bool
    :param watchdog_check_interval: the interval for the watchdog process to check for files that were missed due to potential race conditions
    :type watchdog_check_interval: float
    :param delete_input: whether to delete the input audio files rather than moving them to the output directory
    :type delete_input: bool
    :param verbose: whether to output more logging information
    :type verbose: bool
    :param quiet: whether to suppress output
    :type quiet: bool
    :param beam_size: the beam size to use for decoding
    :type beam_size: int
    """

    poller = Poller()
    poller.input_dir = input_dir
    poller.output_dir = output_dir
    poller.tmp_dir = tmp_dir
    poller.extensions = SUPPORTED_EXTS
    poller.delete_input = delete_input
    poller.progress = not quiet
    poller.verbose = verbose
    poller.check_file = dummy_file_check
    poller.process_file = process_audio
    poller.poll_wait = poll_wait
    poller.continuous = continuous
    poller.use_watchdog = use_watchdog
    poller.watchdog_check_interval = watchdog_check_interval
    poller.params.model = model
    poller.params.transcription_format = transcription_format
    poller.params.beam_size = beam_size
    poller.poll()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Faster Whisper - Prediction", prog="fw_predict_poll")
    parser.add_argument("--model_size", type=str, help="The size of the whisper model to use, e.g., 'base' or 'large-v3'", required=False, default="base")
    parser.add_argument("--device", type=str, help="The device to run on, e.g., 'cuda' or 'cpu'", required=False, default="cpu")
    parser.add_argument("--compute_type", type=str, help="The compute type to use, e.g., 'float16' or 'int8'", required=False, default="int8")
    parser.add_argument("--beam_size", type=int, help="The beam size to use for decoding", required=False, default=5)
    parser.add_argument('--transcription_format', default=TRANSCRIPTION_FORMAT_TEXT, choices=TRANSCRIPTION_FORMATS, help='The format of the generated transcription.')
    parser.add_argument('--prediction_in', help='Path to the test audio files', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the output files folder', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary files folder', required=False, default=None)
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test audio files and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input audio files rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)
    parsed = parser.parse_args()

    try:
        model = load_model(parsed.model_size, device=parsed.device, compute_type=parsed.compute_type)

        # Performing the prediction and producing the output files
        predict_on_audio_files(parsed.prediction_in, model, parsed.prediction_out, parsed.prediction_tmp,
                               transcription_format=parsed.transcription_format, continuous=parsed.continuous,
                               use_watchdog=parsed.use_watchdog, watchdog_check_interval=parsed.watchdog_check_interval,
                               delete_input=parsed.delete_input, verbose=parsed.verbose, quiet=parsed.quiet,
                               beam_size=parsed.beam_size)

    except Exception as e:
        print(traceback.format_exc())
