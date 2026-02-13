# 🚀 Replit W&B Gallery Deployment Guide

This guide will help you deploy your Weights & Biases (W&B) PHATE visualizations on Replit as an interactive web gallery.

## 📋 Prerequisites

1. **W&B Account Access**: Make sure you can access your W&B project `cesar-valdez-mcgill-university/geomancer-phate-labeled`
2. **W&B API Key**: You'll need to authenticate with W&B
3. **Replit Account**: Free account works fine

## 🎯 Quick Start (Option 1: Full Automation)

### Step 1: Create New Replit Project
1. Go to [Replit](https://replit.com)
2. Click "Create Repl"
3. Choose "Python" template
4. Name it `wandb-phate-gallery`

### Step 2: Upload the Gallery Script
Copy the contents of `create_replit_wandb_gallery.py` to your Replit project's main file.

### Step 3: Install Dependencies
In the Replit shell, run:
```bash
pip install wandb requests tqdm
```

### Step 4: Login to W&B
```bash
wandb login
```
Enter your W&B API key when prompted.

### Step 5: Generate Your Gallery
```bash
python create_replit_wandb_gallery.py
```

This will:
- ✅ Download all images from your W&B project
- ✅ Create a self-contained HTML gallery
- ✅ Set up a simple web server
- ✅ Configure Replit hosting

### Step 6: View Your Gallery
1. Click the "Run" button in Replit
2. Click the web preview icon
3. Your gallery will open showing all your PHATE visualizations!

---

## 🛠️ Advanced Options

### Limit Number of Runs (for faster testing)
```bash
python create_replit_wandb_gallery.py --max-runs 20
```

### Skip Download (if you already have the data)
```bash
python create_replit_wandb_gallery.py --skip-download
```

### Custom Configuration
You can modify these variables in the script:
```python
# W&B project configuration
ENTITY = "cesar-valdez-mcgill-university"
PROJECT = "geomancer-phate-labeled"

# Change output directory
REPLIT_OUTPUT_DIR = Path("./my_custom_gallery")
```

---

## 🎨 What You'll Get

Your Replit gallery will include:

### 📊 **Interactive Dashboard**
- **Search & Filter**: Find datasets by name, label type, or run ID
- **Statistics**: Overview of total visualizations, datasets, and label types
- **Responsive Design**: Works on desktop and mobile

### 🖼️ **Image Gallery**
- **Grid Layout**: Clean, organized display of PHATE visualizations
- **Modal Viewer**: Click images for full-screen viewing
- **Metadata Display**: Dataset names, label types, creation dates
- **Direct Links**: Jump to original W&B runs

### 🔗 **W&B Integration**
- **Real Metadata**: All run configurations and summaries
- **Live Links**: Direct links back to W&B runs
- **Run Information**: IDs, creation dates, status

### 📱 **Mobile Friendly**
- Responsive design that works on all devices
- Touch-friendly navigation
- Optimized loading

---

## 🚀 Alternative: Quick Deploy (Option 2)

If you want the simplest possible deployment:

### 1. Use Your Existing HTML Gallery
Your project already has `create_wandb_gallery_v2.py` which creates self-contained HTML files.

### 2. Generate HTML Locally
```bash
cd scripts/visualization
python create_wandb_gallery_v2.py
```

### 3. Upload to Replit
1. Create a new Replit project
2. Upload the generated `.html` file
3. Set up simple hosting:

```python
# main.py
import http.server
import socketserver

PORT = 8000

with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    print(f"Gallery serving at http://localhost:{PORT}")
    httpd.serve_forever()
```

---

## 🛟 Troubleshooting

### W&B Authentication Issues
```bash
# Check if logged in
wandb status

# Re-login if needed
wandb login --relogin
```

### Memory Issues with Large Projects
```bash
# Limit runs processed
python create_replit_wandb_gallery.py --max-runs 50
```

### Missing Images
- Some W&B runs might not have images uploaded
- The script will skip these and continue
- Check your W&B project to ensure images are properly logged

### Replit-Specific Issues
- **Port Access**: Replit auto-assigns ports, the script handles this
- **File Size**: Large galleries might hit Replit's storage limits
- **Performance**: Consider limiting runs for better performance

---

## 🔧 Customization

### Styling
Edit the CSS in the HTML template to customize:
- Colors and themes
- Layout and spacing
- Typography
- Mobile responsiveness

### Functionality
Add features like:
- **Filtering by metadata**: Filter by specific config values
- **Sorting**: Sort by date, dataset name, etc.
- **Annotations**: Add your own notes to runs
- **Export**: Download filtered subsets

### Integration
Connect with:
- **Google Sheets**: Export data for collaboration
- **Slack/Discord**: Automated notifications
- **Other APIs**: Integrate with your lab's systems

---

## 📈 Performance Tips

1. **Limit Initial Load**: Use `--max-runs` for testing
2. **Image Optimization**: Consider resizing large images
3. **Caching**: The script caches downloaded images
4. **Incremental Updates**: Use `--skip-download` for quick regeneration

---

## 🤝 Sharing Your Gallery

Your Replit gallery will have a public URL like:
```
https://wandb-phate-gallery.<username>.repl.co
```

Share this with:
- **Lab Members**: Easy access to visualizations
- **Collaborators**: No W&B account needed
- **Publications**: Supplementary material link
- **Conferences**: Interactive poster displays

---

## 📞 Support

If you run into issues:
1. Check the [Replit documentation](https://docs.replit.com/)
2. Review [W&B API docs](https://docs.wandb.ai/guides/track/public-api-guide)
3. Look at the error messages - they're usually helpful!

---

## 🎉 Next Steps

Once your gallery is running:
1. **Customize the styling** to match your preferences
2. **Add more metadata** from your W&B runs
3. **Set up automated updates** to refresh the gallery
4. **Integrate with other tools** in your workflow

Your W&B data is now beautifully displayed and easily accessible! 🧬✨