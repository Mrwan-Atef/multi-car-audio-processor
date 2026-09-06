from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"


MODELS = {
    "cohere-transcribe-arabic-07-2026":
        "CohereLabs/cohere-transcribe-arabic-07-2026",

    "whisper-large-v3-arabic-dialectal-v2":
        "oddadmix/whisper-large-v3-arabic-dialectal-v2",
}


def download_model(name: str, repo_id: str) -> None:
    target_dir = MODELS_DIR / name

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print(f"DOWNLOADING: {repo_id}")
    print(f"TARGET:      {target_dir}")
    print("=" * 70)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
    )

    print()
    print(f"Finished: {repo_id}")
    print()


def main() -> None:

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, repo_id in MODELS.items():
        download_model(
            name,
            repo_id,
        )

    print("=" * 70)
    print("ALL MODELS DOWNLOADED")
    print("=" * 70)


if __name__ == "__main__":
    main()