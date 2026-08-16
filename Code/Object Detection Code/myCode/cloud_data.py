from __future__ import print_function
import os
import random
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ['']

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

def upload_files(service, folder_id, file, file_path):

        file_metadata = {
            "name": file,
            "parents": [folder_id]   # upload INTO this folder
        }

        media = MediaFileUpload(os.path.join(file_path,file), resumable=True)

        try:
            file_data = service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id"
            ).execute()
        finally:
            if media._fd:
                media._fd.close()
        
        os.remove(os.path.join(file_path,file))
        print(f"Uploaded: {file}  → File ID: {file_data.get('id')}")


def send_data_to_cloud():

    service = authenticate()

    IMAGE_FOLDER_ID=""
    LABEL_FOLDER_ID=""

    VAL_IMAGE_FOLDER_ID=""
    VAL_LABEL_FOLDER_ID=""

    ROOT_DIR=r""

    IMAGE_DIR=os.path.join(ROOT_DIR,"images")
    LABEL_DIR=os.path.join(ROOT_DIR,"labels")

    val_frame=0

    while True:
        images=os.listdir(IMAGE_DIR)
        labels=os.listdir(LABEL_DIR)
        if len(images)>0 and len(labels)>0:
            for image, label in zip(images, labels):
                
                val_frame+=1                
                if val_frame>4:
                    # upload validation images
                    val_frame=0
                    val_image=random.choice(images)
                    val_label=labels[images.index(val_image)]
                    upload_files(service, VAL_IMAGE_FOLDER_ID, val_image, IMAGE_DIR)
                    upload_files(service, VAL_LABEL_FOLDER_ID, val_label, LABEL_DIR)
                    
                upload_files(service, IMAGE_FOLDER_ID, image, IMAGE_DIR)
                upload_files(service, LABEL_FOLDER_ID, label, LABEL_DIR)
        else:
            print("No Files to Upload")

if __name__=="__main__":
    send_data_to_cloud()
