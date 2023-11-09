import multiprocessing
import time
from queue import Empty


def training_process(progress_queue, event):
    for i in range(1, 11):
        time.sleep(1)  # Simulate training iteration taking 1 second
        progress_queue.put(f"Training Progress: {i * 10}%")
    progress_queue.put("Training Complete")
    event.set()  # Signal that training is complete


if __name__ == "__main__":
    progress_queue = multiprocessing.Queue()
    event = multiprocessing.Event()
    training = multiprocessing.Process(
        target=training_process, args=(progress_queue, event)
    )
    training.start()

    # Main process continues running without freezing the UI
    while not event.is_set():
        print("amog")
        try:
            progress = progress_queue.get(
                timeout=0.050
            )  # Wait for 1 second for progress update
            print(progress)  # Update your loading screen with this progress information
        except Empty:
            pass
        print("amog2\n")

    training.join()  # Wait for the training process to finish before exiting
