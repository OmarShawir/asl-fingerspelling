"""Capture one or more clips for the same label, signer and session."""

import argparse

from fingerspelling.app import capture
from fingerspelling.config import DEFAULT_CONFIG, read_config
from fingerspelling.dataset import ClipLabel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person", required=True, help="Anonymous signer code, e.g. p01")
    parser.add_argument("--session", required=True, help="Session code for this signer, e.g. s01")
    parser.add_argument("--hand", required=True, choices=["left", "right"])
    parser.add_argument("--kind", required=True, choices=["letter", "transition", "word"])
    parser.add_argument(
        "--text", required=True, type=str.upper, help="What is signed, e.g. A or HELLO"
    )
    parser.add_argument("--camera", type=int)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        label = ClipLabel(args.text, args.kind, args.person, args.session, args.hand)
    except ValueError as error:
        parser.error(str(error))
    capture(read_config(args.config), args.camera, label)


if __name__ == "__main__":
    main()
