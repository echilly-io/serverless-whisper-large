import os
import base64
from io import BytesIO
from faster_whisper import WhisperModel

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    model_path = "whisper-large-v2-ct2"
    compute_type = "int8_float16"
    model = WhisperModel(
        model_path=model_path,
        compute_type=compute_type,
    )
    print("Successfully loaded model")

def _parse_arg(args : str, data : dict, default = None, required = False):
    arg = data.get(args, None)
    if arg == None:
        if required:
            raise Exception(f"Missing required argument: {args}")
        else:
            return default

    return arg

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    print("Running inference")
    # Parse out your arguments
    try:
        BytesString = _parse_arg("base64String", model_inputs, required=True)
        format = _parse_arg("format", model_inputs, "mp3")
        kwargs = _parse_arg("kwargs", model_inputs, {})

    except Exception as e:
        print("Error parsing arguments")
        return {"error":str(e)}
    
    bytes = BytesIO(base64.b64decode(BytesString.encode("ISO-8859-1")))

    tmp_file = "input."+format
    with open(tmp_file,'wb') as file:
        file.write(bytes.getbuffer())
    
    print("Finished writing file")
    # Run the model
    segments, info = model.transcribe(tmp_file, **kwargs)
    print("Finished running model")
    text = ""
    real_segments = []
    for segment in segments:
        text += segment.text + " "
        current_segment = {
            "text": segment.text,
            "start": segment.start,
            "end": segment.end,
        }
        if kwargs.get("word_timestamps", False) and segment.words:
            current_segment["word_timestamps"] = segment.words

        real_segments.append(current_segment)

    # Format the results
    result = {
        "text": text,
        "segments": segments,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
    }

    os.remove(tmp_file)
    # Return the results as a dictionary
    return result
