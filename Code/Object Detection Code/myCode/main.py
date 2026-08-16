from lpr_detection import *
from cloud_data import send_data_to_cloud
import multiprocessing

if __name__ == "__main__":
    # Create two processes
    process1 = multiprocessing.Process(target=main_lpr_detection)
    process2 = multiprocessing.Process(target=send_data_to_cloud)

    # Start both processes
    process1.start()
    process2.start()

    # Wait for both to finish
    process1.join()
    process2.join()

    print("Both processes finished.")
