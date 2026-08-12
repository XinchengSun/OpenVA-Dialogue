import argparse
import importlib.util
import sys
import types
from pathlib import Path

import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()

    sys.modules.setdefault(
        "omni_avatar_interactive_v2",
        types.SimpleNamespace(),
    )
    server_path = Path(__file__).resolve().parents[1] / "server_mse.py"
    spec = importlib.util.spec_from_file_location(
        "server_listener_export",
        server_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    audio = module.RealtimeMSEEngine._load_listener_virtual_audio(
        args.source,
        3200,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), audio, 16000, subtype="PCM_16")
    print(
        f"output={output} samples={len(audio)} "
        f"seconds={len(audio) / 16000.0:.3f}"
    )


if __name__ == "__main__":
    main()
