# Skin Disease Detection

## Deploy on Render

1. **Upload your model to Google Drive**
   - Upload `efficientnet_b3_final.pth` to Google Drive.
   - Right‑click the file → **Share** → set to **Anyone with the link can view**.
   - Copy the **file ID** from the share link:  
     `https://drive.google.com/file/d/`**`YOUR_FILE_ID`**`/view`

2. **Set environment variable on Render**
   - In your Render service → **Environment** → add:
   - **Key:** `GOOGLE_DRIVE_FILE_ID`  
   - **Value:** your Google Drive file ID (e.g. `1abc...xyz`)

3. **Deploy**
   - On each deploy, the app will download the model from Google Drive at startup if the file is not present (e.g. on Render’s ephemeral filesystem).
   - For local development, either place the `.pth` file in the project folder or set `GOOGLE_DRIVE_FILE_ID` to download it once.
