import os
import io
import zipfile
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
import cv2

# --- Performance tweak: Disable Streamlit's file watcher ---
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# --- Configurable limits to avoid OOM ---
MAX_DESIGNS = 20
MAX_SHIRTS = 30

st.set_page_config(page_title="Shirt Mockup Generator", layout="centered")
st.title("👕 Shirt Mockup Generator – Manual Tag for Model Shirts")

st.markdown("""
Upload **multiple design PNGs** and **shirt templates**.<br>
Tag shirt mockups as either plain or with a model to fine-tune placement offsets.
""", unsafe_allow_html=True)

# --- Sidebar Sliders ---
PADDING_RATIO = st.sidebar.slider("Padding Ratio", 0.1, 1.0, 0.45, 0.05)
plain_offset_pct = st.sidebar.slider("Vertical Offset – Plain Shirt (%)", -50, 100, -7, 1)
model_offset_pct = st.sidebar.slider("Vertical Offset – Model Shirt (%)", -50, 100, 3, 1)

# --- Session Setup ---
for k, v in {
    "zip_files_output": {},
    "design_files": None,
    "design_names": {}
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Clear Design Button (safe!) ---
if st.button("🔄 Start Over (Clear Generated Mockups)"):
    for key in ["design_files", "design_names", "zip_files_output"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# --- Upload Section ---
st.session_state.design_files = st.file_uploader(
    "📌 Upload Design Images (PNG, JPG, JPEG) [Max: 20]", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)
shirt_files = st.file_uploader(
    "🎨 Upload Shirt Templates (PNG, JPG, JPEG) [Max: 30]", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

# --- Check file limits ---
design_files = st.session_state.design_files or []
if len(design_files) > MAX_DESIGNS:
    st.error(f"Please upload **no more than {MAX_DESIGNS} design images**.")
    st.stop()
if shirt_files and len(shirt_files) > MAX_SHIRTS:
    st.error(f"Please upload **no more than {MAX_SHIRTS} shirt templates**.")
    st.stop()

# --- Design Naming ---
if design_files:
    st.markdown("### ✏️ Name Each Design")
    for i, file in enumerate(design_files):
        default_name = os.path.splitext(file.name)[0]
        custom_name = st.text_input(
            f"Name for Design {i+1} ({file.name})",
            value=st.session_state.design_names.get(file.name, default_name),
            key=f"name_input_{i}_{file.name}"
        )
        st.session_state.design_names[file.name] = custom_name

# --- Helper Function: Bounding Box ---
def get_shirt_bbox(pil_image):
    img_cv = np.array(pil_image.convert("RGB"))[:, :, ::-1]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest)
    return None

# --- Batch Script ---
BATCH_SCRIPT = r"""@echo off
setlocal enabledelayedexpansion

for %%f in (*.png) do (
    set "filename=%%~nf"
    mkdir "!filename!"
    move "%%f" "!filename!\\"
)

echo All images moved into their own folders.
pause
"""

# --- Generate Mockups with Progress (sequential, memory-safe) ---
def generate_mockups_with_progress(design_files, shirt_files, design_names, padding_ratio, model_offset, plain_offset):
    zip_outputs = {}
    total = len(design_files) * len(shirt_files)
    progress = st.progress(0, text="Generating mockups...")

    completed = 0
    for design_file in design_files:
        graphic_name = design_names.get(design_file.name, "graphic")
        try:
            design = Image.open(design_file).convert("RGBA")
        except UnidentifiedImageError:
            st.error(f"Failed to open {design_file.name}. Skipping.")
            continue

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zipf:
            for shirt_file in shirt_files:
                color_name = os.path.splitext(shirt_file.name)[0]
                try:
                    shirt = Image.open(shirt_file).convert("RGBA")
                except UnidentifiedImageError:
                    st.error(f"Failed to open {shirt_file.name}. Skipping.")
                    continue

                is_model = "model" in shirt_file.name.lower()
                offset_pct = model_offset if is_model else plain_offset

                bbox = get_shirt_bbox(shirt)
                if bbox:
                    sx, sy, sw, sh = bbox
                    scale = min(sw / design.width, sh / design.height, 1.0) * padding_ratio
                    new_width = int(design.width * scale)
                    new_height = int(design.height * scale)
                    resized_design = design.resize((new_width, new_height))
                    x = sx + (sw - new_width) // 2
                    y = sy + int(sh * offset_pct / 100)
                else:
                    resized_design = design
                    x = (shirt.width - design.width) // 2
                    y = (shirt.height - design.height) // 2

                shirt_copy = shirt.copy()
                shirt_copy.paste(resized_design, (x, y), resized_design)

                output_name = f"{graphic_name}_{color_name}_tee.png"
                img_byte_arr = io.BytesIO()
                shirt_copy.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                zipf.writestr(output_name, img_byte_arr.getvalue())

                completed += 1
                progress.progress(completed / total, text=f"Generating mockups... ({completed}/{total})")

            # Add batch script ONCE per design zip
            zipf.writestr("folder_script.bat", BATCH_SCRIPT)

        zip_buffer.seek(0)
        zip_outputs[graphic_name] = zip_buffer

    progress.empty()
    return zip_outputs

# --- Run Generation ---
if st.button("🚀 Generate Mockups"):
    if not (design_files and shirt_files):
        st.warning("Please upload at least one design and one shirt template.")
    else:
        st.session_state.zip_files_output = generate_mockups_with_progress(
            design_files,
            shirt_files,
            st.session_state.design_names,
            PADDING_RATIO,
            model_offset_pct,
            plain_offset_pct
        )
        st.success("✅ All mockups generated and centered!")

# --- Download Buttons ---
if st.session_state.zip_files_output:
    for name, zip_data in st.session_state.zip_files_output.items():
        st.download_button(
            label=f"📦 Download {name}.zip",
            data=zip_data,
            file_name=f"{name}.zip",
            mime="application/zip",
            key=f"download_{name}"
        )
