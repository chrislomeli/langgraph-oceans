import csv
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import requests
from PIL.Image import Image

log = logging.getLogger(__name__)

# --- config (hardcode for a one-off; promote to args later if you want) ---
CSV_PATH = Path("/Users/chrislomeli/Source/PROJECTS/agenticAI/langgraph-oceans/src/data/oceans_datasets_sightings_full.csv")
BLOB_DIR = Path("/Users/chrislomeli/Source/DATA/oceans/images")  # where the .jpg files land
MANIFEST_PATH = Path("manifest.csv")
MAX_WORKERS = 6  # modest — be polite to the CDN
TIMEOUT = 10


@dataclass(frozen=True)
class Task:
    """One row of work, resolved to where its file should live."""
    sighting_id: str
    individual_id: str
    url: str

    @property
    def dest(self) -> Path:
        # path encodes the data-tie: blobs/<individual_id>/<sighting_id>.jpg
        return BLOB_DIR / self.individual_id / f"{self.sighting_id}.jpg"


@dataclass(frozen=True)
class Result:
    task: Task
    status: str  # "ok" | "skipped" | "failed"
    error: str | None = None


def read_tasks(csv_path: Path):
    """Generator: stream the CSV one row at a time (low memory)."""
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            sighting_id, individual_id, url = row  # TODO: handle header? bad rows?
            yield Task(sighting_id.strip(), individual_id.strip(), url.strip())


def create_retry_session():
    session = requests.Session()

    # Configure retry behavior
    retry_strategy = Retry(
        total=5,  # Max number of retry attempts
        backoff_factor=1,  # Wait formula: backoff_factor * (2 ** (retry_number - 1))
        backoff_jitter=0.5,  # Adds up to 0.5s randomness to break synchronization
        status_forcelist=[408, 429, 500, 502, 503, 504],  # HTTP codes that trigger a retry
        allowed_methods=["GET"],  # HTTP methods to apply retries to
        raise_on_status=True,  # Raise HTTPError if final attempt still fails
        respect_retry_after_header=True  # Pause for the duration specified by the server's Retry-After header
    )

    # Create the adapter and mount it to both HTTP and HTTPS protocols
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

def download_one(task: Task) -> Result:
    """Worker, runs in a thread. Idempotent + one retry. No exceptions escape."""
    if task.dest.exists():
        print(f"found {task.individual_id}")
        return Result(task, "found")

    task.dest.parent.mkdir(parents=True, exist_ok=True)
    # TODO: GET task.url with TIMEOUT, raise_for_status
    with create_retry_session() as session:
        try:
            response = session.get(task.url, timeout=5)
            task.dest.write_bytes(response.content)
            if task.dest.exists():
                return Result(
                    task=task,
                    status=str(response.status_code),
                    error=None
                )
            else:
                return Result(
                    task=task,
                    status=501,
                    error="file was not written - empty image?"
                )

        except requests.exceptions.RetryError as e:
            print(f"Failed after max retries: {e}")
            return Result(
                task=task,
                status=str(response.status_code),
                error=str(e)
            )

        except requests.exceptions.RequestException as e:
            print(f"Request error occurred: {e}")
            return Result(
                task=task,
                status=str(response.status_code),
                error=str(e)
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    tasks = read_tasks(CSV_PATH)  # generator — not materialized

    with MANIFEST_PATH.open("w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["sighting_id", "individual_id", "asset_ref", "source_url", "status"])

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(download_one, t): t for t in tasks}

            # main thread drains results → only it touches the manifest (no lock needed)
            done = 0
            for fut in as_completed(futures):
                result = fut.result()
                t = result.task
                writer.writerow([t.sighting_id, t.individual_id, str(t.dest), t.url, result.status])
                done += 1
                if done % 500 == 0:
                    log.info("processed %d", done)


if __name__ == "__main__":
    main()
