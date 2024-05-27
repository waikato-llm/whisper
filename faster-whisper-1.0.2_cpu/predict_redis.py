from datetime import datetime
import io
import traceback

from predict_common import TRANSCRIPTION_FORMATS, TRANSCRIPTION_FORMAT_TEXT, load_model, transcribe
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log


def process_audio(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()

        data = io.BytesIO(msg_cont.message['data'])
        transcription = transcribe(config.model, data, beam_size=config.beam_size, transcription_format=config.transcription_format)
        msg_cont.params.redis.publish(msg_cont.params.channel_out, transcription)

        if config.verbose:
            log("process_audio - transcription published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_audio - finished processing audio data: %d ms" % processing_time)

    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_audio - failed to process: %s" % traceback.format_exc())


if __name__ == '__main__':
    parser = create_parser("Faster Whisper - Prediction (Redis)", prog="fw_predict_redis", prefix="redis_")
    parser.add_argument("--model_size", type=str, help="The size of the whisper model to use, e.g., 'base' or 'large-v3'", required=False, default="base")
    parser.add_argument("--device", type=str, help="The device to run on, e.g., 'cuda' or 'cpu'", required=False, default="cpu")
    parser.add_argument("--compute_type", type=str, help="The compute type to use, e.g., 'float16' or 'int8'", required=False, default="int8")
    parser.add_argument("--beam_size", type=int, help="The beam size to use for decoding", required=False, default=5)
    parser.add_argument('--transcription_format', default=TRANSCRIPTION_FORMAT_TEXT, choices=TRANSCRIPTION_FORMATS, help='The format of the generated transcription.')
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parsed = parser.parse_args()

    try:
        model = load_model(parsed.model_size, device=parsed.device, compute_type=parsed.compute_type)

        config = Container()
        config.model = model
        config.transcription_format = parsed.transcription_format
        config.beam_size = parsed.beam_size
        config.verbose = parsed.verbose

        params = configure_redis(parsed, config=config)
        run_harness(params, process_audio)

    except Exception as e:
        print(traceback.format_exc())
