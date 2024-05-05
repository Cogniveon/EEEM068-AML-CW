import os
import subprocess
import sys
import gdown


def main(dataset_dir: str) -> None:
    gdown.download(
        "https://drive.google.com/uc?id=1vQ--5fZjdDbHkkZN9k4gtCGhVUG-VQDo",
        os.path.join(dataset_dir, "HMDB_simp.zip"),
        quiet=False,
    )
    subprocess.run(
        f"unzip -o {os.path.join(dataset_dir, 'HMDB_simp.zip')} -d {dataset_dir}",
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        stdin=sys.stdin,
    )


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python prepare_dataset.py <dataset_dir>"

    if sys.argv[1].startswith("/"):
        dataset_dir = sys.argv[1]
    else:
        dataset_dir = os.path.join(os.path.dirname(__file__), sys.argv[1])

    main(dataset_dir)
