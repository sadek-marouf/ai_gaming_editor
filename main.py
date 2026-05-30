# main.py

import os
import sys
import argparse
import logging

from core.pipeline import Pipeline


def main():
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
        default="medium",
        help="Output quality (default: medium)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.video):
        logging.error(f"Video file not found: {args.video}")
        sys.exit(1)

    pipeline = Pipeline(
        args.video,
        output_dir=args.out,
        quality=args.quality,
        parallel_workers=args.workers,
    )

    result = pipeline.run()

    if result:
        print(f"\n{'=' * 60}")
        print(f"SUCCESS: Reel generated")
        print(f"Location: {result}")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'=' * 60}")
        print(f"FAILED: Could not generate reel")
        print(f"{'=' * 60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
