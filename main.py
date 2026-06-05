# main.py

import os
import sys
import argparse
import logging

from core.pipeline import Pipeline
from core.config import Config
from games.registry import list_games


def main():
    available_games = list_games()
    game_list = ", ".join(
        f"{k} ({v})" for k, v in available_games.items()
    )

    parser = argparse.ArgumentParser(
        description=(
            "AI Gaming Editor - Convert gaming videos "
            "to professional viral reels"
        ),
    )

    parser.add_argument(
        "video",
        help="Path to input video file",
    )

    parser.add_argument(
        "--out",
        default="outputs",
        help="Output directory (default: outputs)",
    )

    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high"],
        default="high",
        help="Output quality (default: high)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    parser.add_argument(
        "--game",
        default="generic",
        help=f"Game profile to use. Available: {game_list}",
    )

    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Disable Gemini AI Director (use legacy frame-by-frame pipeline)",
    )

    parser.add_argument(
        "--gemini-model",
        default=None,
        help=f"Gemini model name (default: {Config.GEMINI_MODEL})",
    )

    parser.add_argument(
        "--list-games",
        action="store_true",
        help="List available game profiles and exit",
    )

    args = parser.parse_args()

    if args.list_games:
        print("\nAvailable game profiles:\n")
        for name, display in available_games.items():
            print(f"  {name:12s}  {display}")
        print(f"\nUsage: python main.py video.mp4 --game pubg")
        sys.exit(0)

    if not os.path.exists(args.video):
        logging.error(f"Video file not found: {args.video}")
        sys.exit(1)

    if args.no_gemini:
        Config.USE_GEMINI = False

    if args.gemini_model:
        Config.GEMINI_MODEL = args.gemini_model

    pipeline = Pipeline(
        args.video,
        output_dir=args.out,
        quality=args.quality,
        parallel_workers=args.workers,
        game=args.game,
    )

    result = pipeline.run()

    if result:
        print(f"\n{'=' * 60}")
        print(f"SUCCESS: Reel generated")
        print(f"Location: {result}")
        print(f"Game profile: {args.game}")
        print(f"Pipeline: {'Gemini AI' if Config.USE_GEMINI else 'Legacy'}")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'=' * 60}")
        print(f"FAILED: Could not generate reel")
        print(f"{'=' * 60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
