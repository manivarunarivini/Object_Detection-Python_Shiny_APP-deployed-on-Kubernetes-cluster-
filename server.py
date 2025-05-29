import json
import owncloud
from shiny import render, reactive, ui
from urllib3 import Timeout
import subprocess
import os
from datetime import datetime

PUBLIC_LINK = "https:Link_Protected"
FOLDER_PASSWORD = "protected"
CONFIG_FILE = "config.json"
PREDICTED_FOLDER = "Predicted_Images"

CLASS_NAMES = [
    "Ivy", "Fern", "Ginkgo", "Kummerowia striata",
    "Laciniata", "Macrolobium acaciifolium",
    "Micranthes odontoloma", "Murraya",
    "Robinia pseudoacacia", "Selaginella davidi franch"
]

def get_cloud_connection():
    return owncloud.Client.from_public_link(
        PUBLIC_LINK,
        folder_password=FOLDER_PASSWORD,
        timeout=Timeout(connect=3.0, read=5.0)
    )

def load_config():
    try:
        cloud = get_cloud_connection()
        config_data = cloud.get_file_contents(CONFIG_FILE)
        config = json.loads(config_data.decode())
        if not all(k in config["CameraSettings"] for k in ["resolution", "measurement_interval"]):
            raise ValueError("Invalid config structure")
        return config
    except Exception as e:
        print(f"[ERROR] Config load error: {e}")
        return None

def save_config(updated_config):
    try:
        cloud = get_cloud_connection()
        config_str = json.dumps(updated_config, indent=4)
        cloud.put_file_contents(CONFIG_FILE, config_str.encode())
        saved = json.loads(cloud.get_file_contents(CONFIG_FILE).decode())
        if saved != updated_config:
            raise ValueError("Config verification failed")
        return True
    except Exception as e:
        print(f"[ERROR] Config save error: {e}")
        return False

def extract_class(filename):
    """Extract class name from filename (e.g., 'Fern_20250514_175012.jpg' -> 'Fern')"""
    return filename.split('_')[0]

def extract_date(filename):
    """Extract date from filename (e.g., 'Fern_20250514_175012.jpg' -> datetime(2025,5,14))"""
    try:
        date_str = filename.split('_')[1]
        return datetime.strptime(date_str, "%Y%m%d").date()
    except:
        return None

def list_predictions(selected_class="All"):
    try:
        cloud = get_cloud_connection()
        files = cloud.list(PREDICTED_FOLDER)
        
        filtered_files = []
        for f in files:
            filename = f.get_name()
            
            # Skip non-image files
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            file_class = extract_class(filename)
            
            # Apply class filter (no "Other" option)
            class_match = (
                selected_class == "All" or
                file_class == selected_class
            )
            
            if class_match:
                filtered_files.append(f)
        
        # Sort by newest first and get last 3
        filtered_files.sort(key=lambda x: x.get_last_modified(), reverse=True)
        filtered_files = filtered_files[:3]  # Only keep 3 most recent
        
        # Generate URLs
        image_urls = []
        for f in filtered_files:
            url = f"{PUBLIC_LINK}/download?path=/{PREDICTED_FOLDER}&files={f.get_name()}&password={FOLDER_PASSWORD}"
            image_urls.append(url)
        
        return image_urls
        
    except Exception as e:
        print(f"[ERROR] Failed to list predictions: {e}")
        return []

def server(input, output, session):
    config = reactive.Value(None)
    last_success = reactive.Value(None)
    predictions = reactive.Value([])

    @reactive.Calc
    def filtered_predictions():
        # Simplified without date range
        return list_predictions(selected_class=input.class_filter())

    @reactive.Effect
    async def load_config_on_start():
        loaded = load_config()
        if loaded:
            config.set(loaded)
            last_success.set(loaded)
            await update_ui(loaded)
            predictions.set(filtered_predictions())

    async def update_ui(values):
        await session.send_custom_message("update_inputs", {
            "resolution": values["CameraSettings"]["resolution"],
            "measurement_interval": values["CameraSettings"]["measurement_interval"]
        })

    @output
    @render.text
    def status():
        if not config():
            return "[INFO] Loading config..."
        return f"Current: {config()['CameraSettings']['resolution']} @ {config()['CameraSettings']['measurement_interval']} sec"

    @output
    @render.text
    def filter_stats():
        count = len(predictions())
        return f"Showing {count} matching images"

    @reactive.Effect
    @reactive.event(input.update)
    async def handle_update():
        new_res = input.resolution()
        new_fps = input.measurement_interval()
        try:
            sec = int(new_fps) if new_fps else 60
            if sec <= 0:
                raise ValueError("measurement_interval must be positive")
        except ValueError as e:
            await session.send_custom_message("status_update", f"[ERROR] Invalid input: {e}")
            return

        updated_config = {
            "CameraSettings": {
                "resolution": new_res if new_res else "1920x1080",
                "measurement_interval": sec
            }
        }

        if save_config(updated_config):
            config.set(updated_config)
            last_success.set(updated_config)
            await update_ui(updated_config)
            await session.send_custom_message("status_update", "✅ Config saved!")
        else:
            if last_success():
                await update_ui(last_success())
            await session.send_custom_message("status_update", "❌ Save failed - using last good config")

    @reactive.Effect
    @reactive.event(input.refresh_predictions)
    async def handle_refresh():
        await session.send_custom_message("status_update", "[INFO] Running prediction script...")
        try:
            result = subprocess.run(["python", "yolov8_nextcloud_predictor_timestamped.py"], check=True, capture_output=True, text=True)
            await session.send_custom_message("status_update", "✅ Predictions refreshed.")
            predictions.set(filtered_predictions())
        except subprocess.CalledProcessError as e:
            await session.send_custom_message("status_update", f"[ERROR] Prediction failed: {e.stderr}")

    @output
    @render.ui
    def predicted_gallery():
        imgs = predictions()
        if not imgs:
            return ui.p("No matching predictions found.")
        
        # Always show max 3 images
        display_images = imgs[:3]
        
        return ui.div(
            *[ui.tags.img(
                src=img,
                style="height: 250px; margin: 10px; border: 2px solid #eee;",
                title=img.split('files=')[1].split('&')[0]
            ) for img in display_images],
            style="display: flex; flex-wrap: wrap;"
        )