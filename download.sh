#!/bin/bash

# File ID and output file
FILE_ID="1fiUhrSjwIHzE-og2C69tlpkXMu9ecNws"
OUTPUT_FILE="models.zip"

# Use gdown to download the file from Google Drive
gdown https://drive.google.com/uc?id=${FILE_ID} -O ${OUTPUT_FILE}

# Unzip the downloaded file
unzip ${OUTPUT_FILE}

# Remove the zip file
rm ${OUTPUT_FILE}

echo "Model downloaded and extracted"