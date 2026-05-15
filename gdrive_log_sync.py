import threading
import time
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from paths import LOG_ROOT, GDRIVE_FOLDER_ID, GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH


DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]
UPLOAD_DEBOUNCE_SECONDS = 5
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


def authenticate_drive():
    credentials = None
    if GDRIVE_TOKEN_PATH.exists():
        credentials = Credentials.from_authorized_user_file(str(GDRIVE_TOKEN_PATH), DRIVE_SCOPES)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(GDRIVE_CREDENTIALS_PATH), DRIVE_SCOPES)
            credentials = flow.run_local_server(port=0)
        GDRIVE_TOKEN_PATH.write_text(credentials.to_json())

    return build("drive", "v3", credentials=credentials)


def find_or_create_folder(service, folder_name, parent_id):
    query = (f"name='{folder_name}' and '{parent_id}' in parents "
             f"and mimeType='{FOLDER_MIME_TYPE}' and trashed=false")
    response = service.files().list(q=query, fields="files(id)").execute()
    existing = response.get("files", [])
    if existing:
        return existing[0]["id"]

    folder = service.files().create(
        body={"name": folder_name, "parents": [parent_id], "mimeType": FOLDER_MIME_TYPE},
        fields="id",
    ).execute()
    return folder["id"]


def find_existing_file(service, file_name, parent_id):
    query = f"name='{file_name}' and '{parent_id}' in parents and trashed=false"
    response = service.files().list(q=query, fields="files(id)").execute()
    files = response.get("files", [])
    return files[0]["id"] if files else None


def upload_or_update_file(service, local_path, remote_parent_id):
    file_name = local_path.name
    media = MediaFileUpload(str(local_path), resumable=False)
    existing_id = find_existing_file(service, file_name, remote_parent_id)

    if existing_id:
        service.files().update(fileId=existing_id, media_body=media).execute()
    else:
        service.files().create(
            body={"name": file_name, "parents": [remote_parent_id]},
            media_body=media,
            fields="id",
        ).execute()


def resolve_remote_folder(service, local_path, root_local, root_remote_id):
    relative_parts = local_path.relative_to(root_local).parts[:-1]
    current_parent = root_remote_id
    for part in relative_parts:
        current_parent = find_or_create_folder(service, part, current_parent)
    return current_parent


class DebouncedUploader:
    def __init__(self, service, root_local, root_remote_id):
        self.service = service
        self.root_local = root_local
        self.root_remote_id = root_remote_id
        self.pending_paths = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self.process_loop, daemon=True)
        self.worker_thread.start()

    def schedule(self, local_path):
        with self.lock:
            self.pending_paths[local_path] = time.time()

    def process_loop(self):
        while not self.stop_event.is_set():
            time.sleep(1)
            now = time.time()
            ready_paths = []
            with self.lock:
                for path, last_modified in list(self.pending_paths.items()):
                    if now - last_modified >= UPLOAD_DEBOUNCE_SECONDS:
                        ready_paths.append(path)
                        del self.pending_paths[path]

            for path in ready_paths:
                if not path.exists():
                    continue
                try:
                    remote_parent = resolve_remote_folder(self.service, path, self.root_local, self.root_remote_id)
                    upload_or_update_file(self.service, path, remote_parent)
                except Exception as error:
                    print(f"[gdrive] upload failed for {path}: {error}")

    def shutdown(self):
        self.stop_event.set()
        self.worker_thread.join(timeout=10)


class LogChangeHandler(FileSystemEventHandler):
    def __init__(self, uploader):
        self.uploader = uploader

    def on_created(self, event):
        if not event.is_directory:
            self.uploader.schedule(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory:
            self.uploader.schedule(Path(event.src_path))


def start_log_sync(run_name):
    service = authenticate_drive()
    run_folder_id = find_or_create_folder(service, run_name, GDRIVE_FOLDER_ID)

    local_run_dir = LOG_ROOT / run_name
    local_run_dir.mkdir(parents=True, exist_ok=True)

    uploader = DebouncedUploader(service, local_run_dir, run_folder_id)
    observer = Observer()
    observer.schedule(LogChangeHandler(uploader), str(local_run_dir), recursive=True)
    observer.start()

    print(f"[gdrive] syncing {local_run_dir} -> Drive folder '{run_name}'")
    return observer, uploader
