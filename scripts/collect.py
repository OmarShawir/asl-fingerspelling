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
    parser.add_argument(
        "--save-diagnostic-frames",
        action="store_true",
        help="Save original PNG frames linked to the saved clip",
    )
    parser.add_argument(
        "--diagnostic-every",
        type=int,
        default=10,
        help="Save every Nth recorded frame (default 10; 1 saves every frame)",
    )
    args = parser.parse_args()
    if args.diagnostic_every < 1:
        parser.error("--diagnostic-every must be positive")
    try:
        label = ClipLabel(args.text, args.kind, args.person, args.session, args.hand)
    except ValueError as error:
        parser.error(str(error))
    capture(
        read_config(args.config),
        args.camera,
        label,
        save_diagnostic_frames=args.save_diagnostic_frames,
        diagnostic_every=args.diagnostic_every,
    )


if __name__ == "__main__":
    main()
