from gdrive_log_sync import start_log_sync
from paths import LOG_ROOT
import time

run_name = "test_sync"
observer, uploader = start_log_sync(run_name)

test_file = LOG_ROOT / run_name / "hello.txt"
test_file.write_text("hello from CVTAH")

print("Waiting 15 seconds for upload to complete...")
time.sleep(15)

observer.stop()
observer.join()
uploader.shutdown()
print("Done. Check Drive for 'test_sync/hello.txt'")
