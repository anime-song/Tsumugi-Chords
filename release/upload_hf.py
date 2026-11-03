"""Upload the model folder to the Hugging Face model repository, with the model card as its README.md and the
weights' license as its LICENSE.

    python release/upload_hf.py model [--repo anime-song/Tsumugi-Chords] [--public] [--tag v0.1.0]

A new repository is created private unless ``--public``; an existing one keeps its visibility (change it on
the Hub, or with ``--public``).  ``--tag`` tags the uploaded commit, so that ``load(revision=...)`` can pin it.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--repo", default="anime-song/Tsumugi-Chords")
    parser.add_argument("--public", action="store_true", help="make the repository public")
    parser.add_argument("--tag", help="tag the uploaded commit, e.g. v0.1.0")
    parser.add_argument("--message", default="Tsumugi-Chords weights and model card")
    args = parser.parse_args()
    if not (args.folder / "config.json").is_file():
        raise SystemExit(f"{args.folder} has no config.json")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    if args.public:
        api.update_repo_settings(args.repo, private=False)
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "hf"
        shutil.copytree(args.folder, tree)
        shutil.copy(HERE / "MODEL_CARD.md", tree / "README.md")
        shutil.copy(HERE / "MODEL_LICENSE", tree / "LICENSE")
        commit = api.upload_folder(repo_id=args.repo, folder_path=tree, commit_message=args.message)
    print(commit.commit_url)
    if args.tag:
        api.create_tag(args.repo, tag=args.tag, revision=commit.oid)
        print(f"tagged {args.tag}")


if __name__ == "__main__":
    main()
