import sys
import os
import pysftp
from io import BytesIO,StringIO

# command for running the computation
# The terminal can run this command, replace the names of the files if yours are named differently or they are not in the same folder,
# then the complete relative path is necessary
# RUN: python .\meta_communication.py .\model_1_train.py

# Load env variables from .env
from dotenv import load_dotenv
load_dotenv()

# Fetch filename from cmd args
FILENAME = sys.argv[1]

# Fetch env variables
SFTP_META_HOST = os.environ.get("SFTP_META_HOST")
SFTP_META_USER = os.environ.get("SFTP_META_USER")
SFTP_META_PWD = os.environ.get("SFTP_META_PWD")

# Project folder on the SFTP server
# replace the path with the actual path to your folder on the metacentrum
PROJECT_FOLDER = "/storage/projects/CVUT_Fsv_AO/Matyas_BP/results"
PROJECT_USER_FOLDER = f"{PROJECT_FOLDER}/train"

# Create connection
cnopts = pysftp.CnOpts()
cnopts.hostkeys = None

# host, username and password can be written explicitly here or be loaded from a .env file to ensure security
with pysftp.Connection(host=SFTP_META_HOST, username=SFTP_META_USER, password=SFTP_META_PWD, cnopts=cnopts) as sftp:

    # Change directory to the group folder
    sftp.chdir(PROJECT_FOLDER)
    #print(sftp.listdir(AO_PROJECT_FOLDER))

    # Create project folder if it does not exist
    if not sftp.isdir(PROJECT_USER_FOLDER):
        sftp.mkdir(PROJECT_USER_FOLDER)

    # Change to project folder
    sftp.chdir(PROJECT_USER_FOLDER)
    #print(sftp.pwd)

    runsh = BytesIO()
    with open("run.sh", "r") as f:
        [
            runsh.write(
                str.encode(line.replace("<FILE_NAME>", FILENAME))
            )
            for line in f.readlines()
        ]

    with open("run.sh", "wb") as f:
        f.write(runsh.getbuffer())

    # Upload file to meta
    sftp.put("run.sh")
    sftp.put(f"{FILENAME}")
    #print(sftp.listdir())

    #os.remove("run.sh")

    print(sftp.execute(f"qsub {PROJECT_USER_FOLDER}/run.sh"))

    # end
