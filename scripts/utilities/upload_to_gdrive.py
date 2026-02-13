#!/usr/bin/env python3
"""Upload PHATE plots to Google Drive and update Google Sheet with IMAGE() formulas."""

import sqlite3
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pandas as pd

# Google API scopes
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/spreadsheets'
]

# Configuration
DB_PATH = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/manylatents_datasets.db")
SERVICE_ACCOUNT_FILE = Path.home() / ".config" / "gdrive_service_account.json"
SPREADSHEET_ID = "1SgRAPzCNqxCeNA1RfvaRri62UfZDG224ol62cYB-dJE"
SHARED_DRIVE_FOLDER_ID = "1SNdt1xUF6Mjd0tQJvuCMkerVEQG8OihE"  # User's Shared Drive

def authenticate():
    """Authenticate with Google APIs using service account."""
    if not SERVICE_ACCOUNT_FILE.exists():
        print(f"ERROR: Service account file not found at {SERVICE_ACCOUNT_FILE}")
        return None

    creds = service_account.Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_FILE),
        scopes=SCOPES
    )
    return creds

def create_gdrive_folder(service, folder_name):
    """Create a folder in Google Drive."""
    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = service.files().create(
        body=folder_metadata,
        fields='id'
    ).execute()

    # Make folder publicly accessible
    permission = {
        'type': 'anyone',
        'role': 'reader'
    }
    service.permissions().create(
        fileId=folder['id'],
        body=permission
    ).execute()

    return folder['id']

def upload_image(service, file_path, folder_id):
    """Upload an image to Shared Drive and return shareable URL."""
    file_metadata = {
        'name': Path(file_path).name,
        'parents': [folder_id]
    }

    media = MediaFileUpload(
        file_path,
        mimetype='image/png',
        resumable=True
    )

    # Use supportsAllDrives for Shared Drive support
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, webViewLink',
        supportsAllDrives=True
    ).execute()

    # Make file publicly accessible
    permission = {
        'type': 'anyone',
        'role': 'reader'
    }
    service.permissions().create(
        fileId=file['id'],
        body=permission,
        supportsAllDrives=True
    ).execute()

    # Get direct image URL (not webViewLink)
    direct_url = f"https://drive.google.com/uc?export=view&id={file['id']}"

    return direct_url

def update_google_sheet(service, image_urls):
    """Update Google Sheet with IMAGE() formulas."""
    # Read current sheet data
    result = service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range='A:Z'
    ).execute()

    values = result.get('values', [])

    if not values:
        print("ERROR: Sheet is empty!")
        return

    # Find the column with PHATE plot paths (phate_plot_image)
    header = values[0]

    # Find column indices
    try:
        phate_col_idx = header.index('phate_plot_image')
        dataset_id_idx = header.index('dataset_id')
    except ValueError as e:
        print(f"ERROR: Could not find required columns: {e}")
        print(f"Available columns: {header}")
        return

    # Create mapping of dataset_id to row index
    dataset_to_row = {}
    for i, row in enumerate(values[1:], start=2):  # Start from row 2 (skip header)
        if len(row) > dataset_id_idx:
            dataset_to_row[row[dataset_id_idx]] = i

    # Prepare updates
    updates = []
    for dataset_id, image_url in image_urls.items():
        if dataset_id in dataset_to_row:
            row_idx = dataset_to_row[dataset_id]
            # Add IMAGE formula in the phate_plot_image column
            cell_range = f"{chr(65 + phate_col_idx)}{row_idx}"
            updates.append({
                'range': cell_range,
                'values': [[f'=IMAGE("{image_url}", 1)']]
            })

    # Batch update
    if updates:
        body = {'valueInputOption': 'USER_ENTERED', 'data': updates}
        service.spreadsheets().values().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body=body
        ).execute()
        print(f"✓ Updated {len(updates)} cells with IMAGE() formulas")
    else:
        print("WARNING: No updates to make")

def main():
    print("="*80)
    print("PHATE PLOTS TO GOOGLE DRIVE UPLOADER")
    print("="*80)

    # Authenticate
    print("\n1. Authenticating with Google (Service Account)...")
    creds = authenticate()
    if not creds:
        return

    drive_service = build('drive', 'v3', credentials=creds)
    sheets_service = build('sheets', 'v4', credentials=creds)
    print("✓ Authenticated successfully")

    # Use Shared Drive folder
    print("\n2. Using Shared Drive folder...")
    folder_id = SHARED_DRIVE_FOLDER_ID
    print(f"✓ Shared Drive folder ID: {folder_id}")

    # Get dataset info from database
    print("\n3. Reading dataset information...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT dataset_id, dataset_name, phate_plot_path
        FROM datasets
        WHERE downloaded = 1 AND phate_plot_path IS NOT NULL
        ORDER BY file_size_mb ASC
    """, conn)
    conn.close()
    print(f"✓ Found {len(df)} datasets with PHATE plots")

    # Upload images
    print("\n4. Uploading images to Google Drive...")
    image_urls = {}

    for idx, row in df.iterrows():
        dataset_id = row['dataset_id']
        phate_path = Path(row['phate_plot_path'])

        if not phate_path.exists():
            print(f"⚠ Skipping {dataset_id}: File not found")
            continue

        try:
            url = upload_image(drive_service, phate_path, folder_id)
            image_urls[dataset_id] = url

            if (idx + 1) % 10 == 0:
                print(f"  Uploaded {idx + 1}/{len(df)}")

        except Exception as e:
            print(f"⚠ Error uploading {dataset_id}: {e}")
            continue

    print(f"✓ Successfully uploaded {len(image_urls)}/{len(df)} images")

    # Update Google Sheet
    print("\n5. Updating Google Sheet with IMAGE() formulas...")
    update_google_sheet(sheets_service, image_urls)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Google Sheet: https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit")
    print(f"Total images uploaded: {len(image_urls)}")

if __name__ == "__main__":
    main()
