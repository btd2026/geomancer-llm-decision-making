#!/usr/bin/env python3
"""
Upload PHATE images to imgbb and create Google Sheet for collaborative labeling.
Requires: imgbb API key (free at https://api.imgbb.com/)
"""

import requests
import base64
import pandas as pd
from pathlib import Path
import time
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Configuration
LABELS_DIR = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/labels")
TEMPLATE_FILE = LABELS_DIR / "label_template.csv"
IMAGE_URLS_FILE = LABELS_DIR / "image_urls.json"
SERVICE_ACCOUNT_FILE = Path.home() / ".config" / "gdrive_service_account.json"

# Google Sheets config
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

def upload_to_imgbb(image_path, api_key):
    """Upload image to imgbb and return URL."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    response = requests.post(
        "https://api.imgbb.com/1/upload",
        data={
            "key": api_key,
            "image": image_data,
            "name": Path(image_path).stem
        },
        timeout=60
    )

    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            return data['data']['url']

    return None

def upload_all_images(api_key):
    """Upload all PHATE images to imgbb."""
    print("Loading dataset info...")
    df = pd.read_csv(TEMPLATE_FILE)

    # Load existing URLs if any
    if IMAGE_URLS_FILE.exists():
        with open(IMAGE_URLS_FILE, 'r') as f:
            urls = json.load(f)
        print(f"Loaded {len(urls)} existing URLs")
    else:
        urls = {}

    print(f"\nUploading {len(df)} images to imgbb...")
    print("(This may take a few minutes)\n")

    for idx, row in df.iterrows():
        dataset_id = row['dataset_id']

        # Skip if already uploaded
        if dataset_id in urls:
            continue

        image_path = Path(row['image_path'])
        if not image_path.exists():
            print(f"⚠ Skipping {dataset_id}: Image not found")
            continue

        try:
            url = upload_to_imgbb(image_path, api_key)
            if url:
                urls[dataset_id] = url
                print(f"✓ {idx+1}/{len(df)}: {dataset_id[:8]}...")

                # Save progress every 10 uploads
                if len(urls) % 10 == 0:
                    with open(IMAGE_URLS_FILE, 'w') as f:
                        json.dump(urls, f, indent=2)

                # Rate limiting
                time.sleep(0.5)
            else:
                print(f"✗ {idx+1}/{len(df)}: Failed to upload {dataset_id[:8]}")
        except Exception as e:
            print(f"✗ Error uploading {dataset_id}: {e}")

    # Final save
    with open(IMAGE_URLS_FILE, 'w') as f:
        json.dump(urls, f, indent=2)

    print(f"\n✓ Uploaded {len(urls)} images")
    print(f"URLs saved to: {IMAGE_URLS_FILE}")

    return urls

def create_google_sheet(urls, spreadsheet_id=None):
    """Create or update Google Sheet with image URLs."""
    print("\nConnecting to Google Sheets...")

    # Authenticate
    creds = service_account.Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_FILE),
        scopes=SCOPES
    )
    sheets_service = build('sheets', 'v4', credentials=creds)
    drive_service = build('drive', 'v3', credentials=creds)

    # Load template
    df = pd.read_csv(TEMPLATE_FILE)

    # Create new spreadsheet if needed
    if not spreadsheet_id:
        spreadsheet = sheets_service.spreadsheets().create(
            body={
                'properties': {'title': 'PHATE Structure Labeling'},
                'sheets': [{'properties': {'title': 'Labeling'}}]
            }
        ).execute()
        spreadsheet_id = spreadsheet['spreadsheetId']

        # Make it accessible
        drive_service.permissions().create(
            fileId=spreadsheet_id,
            body={'type': 'anyone', 'role': 'writer'},
            supportsAllDrives=True
        ).execute()

        print(f"✓ Created new spreadsheet: {spreadsheet_id}")

    # Prepare data
    header = ['Dataset ID', 'PHATE Image', 'Label (0-3)', 'Notes', 'Labeler']
    rows = [header]

    for _, row in df.iterrows():
        dataset_id = row['dataset_id']
        image_url = urls.get(dataset_id, '')

        if image_url:
            # Use IMAGE formula
            image_formula = f'=IMAGE("{image_url}", 4, 200, 200)'
        else:
            image_formula = 'No image'

        rows.append([
            dataset_id,
            image_formula,
            '',  # Label
            '',  # Notes
            ''   # Labeler
        ])

    # Update sheet
    sheets_service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range='Labeling!A1',
        valueInputOption='USER_ENTERED',
        body={'values': rows}
    ).execute()

    # Format sheet
    requests_body = {
        'requests': [
            # Set column widths
            {'updateDimensionProperties': {
                'range': {'sheetId': 0, 'dimension': 'COLUMNS', 'startIndex': 0, 'endIndex': 1},
                'properties': {'pixelSize': 300},
                'fields': 'pixelSize'
            }},
            {'updateDimensionProperties': {
                'range': {'sheetId': 0, 'dimension': 'COLUMNS', 'startIndex': 1, 'endIndex': 2},
                'properties': {'pixelSize': 220},
                'fields': 'pixelSize'
            }},
            # Set row heights for images
            {'updateDimensionProperties': {
                'range': {'sheetId': 0, 'dimension': 'ROWS', 'startIndex': 1, 'endIndex': len(rows)},
                'properties': {'pixelSize': 210},
                'fields': 'pixelSize'
            }},
            # Freeze header row
            {'updateSheetProperties': {
                'properties': {'sheetId': 0, 'gridProperties': {'frozenRowCount': 1}},
                'fields': 'gridProperties.frozenRowCount'
            }},
            # Add data validation for label column
            {'setDataValidation': {
                'range': {'sheetId': 0, 'startRowIndex': 1, 'endRowIndex': len(rows),
                         'startColumnIndex': 2, 'endColumnIndex': 3},
                'rule': {
                    'condition': {'type': 'ONE_OF_LIST', 'values': [
                        {'userEnteredValue': '0 - Clusters'},
                        {'userEnteredValue': '1 - Trajectories'},
                        {'userEnteredValue': '2 - Continuous'},
                        {'userEnteredValue': '3 - Noisy/Mixed'}
                    ]},
                    'showCustomUi': True,
                    'strict': False
                }
            }}
        ]
    }

    sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=requests_body
    ).execute()

    sheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"
    print(f"\n✓ Google Sheet ready!")
    print(f"URL: {sheet_url}")

    return spreadsheet_id, sheet_url

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create collaborative labeling Google Sheet")
    parser.add_argument('--api-key', type=str, help='imgbb API key')
    parser.add_argument('--skip-upload', action='store_true', help='Skip image upload (use existing URLs)')
    parser.add_argument('--sheet-id', type=str, help='Existing Google Sheet ID to update')
    args = parser.parse_args()

    print("="*60)
    print("PHATE Labeling Google Sheet Creator")
    print("="*60)

    # Check for API key
    if not args.skip_upload and not args.api_key:
        print("\nTo upload images, you need a free imgbb API key:")
        print("1. Go to https://api.imgbb.com/")
        print("2. Sign up and get your API key")
        print("3. Run: python create_labeling_sheet.py --api-key YOUR_KEY")
        print("\nOr use --skip-upload if images are already uploaded.")
        return

    # Upload images
    if args.skip_upload:
        if IMAGE_URLS_FILE.exists():
            with open(IMAGE_URLS_FILE, 'r') as f:
                urls = json.load(f)
            print(f"✓ Loaded {len(urls)} existing image URLs")
        else:
            print("✗ No existing URLs found. Run without --skip-upload first.")
            return
    else:
        urls = upload_all_images(args.api_key)

    # Create Google Sheet
    if urls:
        create_google_sheet(urls, args.sheet_id)

if __name__ == "__main__":
    main()
