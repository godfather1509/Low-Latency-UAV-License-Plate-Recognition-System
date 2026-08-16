from __future__ import print_function
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("drive", "v3", credentials=creds)

def upload_files(service, folder_id, files, file_path):
    for file in files:
        # file_name = os.path.basename(file_path)

        file_metadata = {
            "name": file,
            "parents": [folder_id]   # upload INTO this folder
        }

        media = MediaFileUpload(os.path.join(file_path,file), resumable=True)

        file_data = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id"
        ).execute()

        print(f"Uploaded: {file}  → File ID: {file_data.get('id')}")


def send_data_to_cloud():

    service = authenticate()

    IMAGE_FOLDER_ID="1YqiS0jeZPfGHtT9wGmHUsnEb0fivu52E"
    LABEL_FOLDER_ID="1G7JxOv_QlKq6eESKpITP4037YSouNYSV"

    ROOT_DIR=r"C:\Users\ayush\OneDrive\Desktop\major project\quadcopter\Code\Object Detection Code\myCode\frames_with_plate"

    IMAGE_DIR=os.path.join(ROOT_DIR,"images")
    LABEL_DIR=os.path.join(ROOT_DIR,"labels")

    # images=os.listdir(IMAGE_DIR)
    # labels=os.listdir(LABEL_DIR)

    # upload_files(service, IMAGE_FOLDER_ID, images, IMAGE_DIR)
    # upload_files(service, LABEL_FOLDER_ID, labels, LABEL_DIR)

    while True:
        images=os.listdir(IMAGE_DIR)
        labels=os.listdir(LABEL_DIR)

        if len(images)>0 and len(labels)>0:

            for image, label in zip(images, labels):

                image_metadata = {
                    "name": image,
                    "parents": [IMAGE_FOLDER_ID]   # upload INTO this folder
                }

                label_metadata = {
                    "name": label,
                    "parents": [LABEL_FOLDER_ID]   # upload INTO this folder
                }

                image_media = MediaFileUpload(os.path.join(IMAGE_DIR,image), resumable=True)
                label_media = MediaFileUpload(os.path.join(LABEL_DIR,label), resumable=True)

                image_data = service.files().create(
                    body=image_metadata,
                    media_body=image_media,
                    fields="id"
                ).execute()

                label_data = service.files().create(
                    body=label_metadata,
                    media_body=label_media,
                    fields="id"
                ).execute()

                print(f"Uploaded: {image}  → File ID: {image_data.get('id')}")
                print(f"Uploaded: {label}  → File ID: {label_data.get('id')}")

                os.remove(os.path.join(IMAGE_DIR,image))
                os.remove(os.path.join(LABEL_DIR,label))


if __name__=="__main__":
    send_data_to_cloud()