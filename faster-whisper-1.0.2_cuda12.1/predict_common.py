import json
from faster_whisper import WhisperModel


TRANSCRIPTION_FORMAT_TEXT = "text"
TRANSCRIPTION_FORMAT_JSON = "json"
TRANSCRIPTION_FORMATS = [
    TRANSCRIPTION_FORMAT_TEXT,
    TRANSCRIPTION_FORMAT_JSON,
]


def load_model(model_size="small", device="cuda", compute_type="float16"):
    """
    Loads the model.

    :param model_size: the size of the whisper model, eg base/small/tiny/large-v3
    :type model_size: str
    :param device: the device to run the model on, eg cpu/cuda
    :type device: str
    :param compute_type: the data type to use, eg float16 (cuda) or int8 (cpu)
    :type compute_type: str
    :return: the model
    :rtype: WhisperModel
    """
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return model


def transcribe(model, audio, beam_size=5, transcription_format=TRANSCRIPTION_FORMAT_TEXT):
    """
    Transcribes the audio.

    :param model: the model to use for the transcription
    :type model: WhisperModel
    :param audio: the audio file, bytes or numpy array to transcribe
    :param beam_size: the beam size to use for decoding
    :type beam_size: int
    :param transcription_format: the transcription format to generate
    :type transcription_format: str
    :return: the generated transcription
    :rtype: str
    """
    result = []
    segments, info = model.transcribe(audio, beam_size=beam_size)
    for segment in segments:
        if transcription_format == TRANSCRIPTION_FORMAT_JSON:
            result.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            })
        else:
            result.append(segment.text)
    if transcription_format == TRANSCRIPTION_FORMAT_JSON:
        return json.dumps(result, indent=2)
    else:
        return "\n".join(result)
