import streamlit as st
import pandas as pd
import folium
import json
from streamlit_folium import st_folium
from PIL import Image
from datetime import datetime
# ============================================================
# GEO-TAGGING / SURVEY GPS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Geo-tagging")

gps_mode = st.sidebar.radio(
    "GPS Mode",
    [
        "Real GPS / Survey CSV",
        "Simulator GPS"
    ],
    index=0
)

survey_df = None

if gps_mode == "Real GPS / Survey CSV":

    st.sidebar.info(
        "Upload the GPS/navigation CSV recorded during the sonar survey."
    )

    gps_file = st.sidebar.file_uploader(
        "Upload survey GPS CSV",
        type=["csv"],
        key="survey_gps_csv"
    )

    if gps_file is not None:

        try:
            survey_df = pd.read_csv(gps_file)

            # Normalize column names
            survey_df.columns = [
                str(c).strip().lower().replace(" ", "_")
                for c in survey_df.columns
            ]

            # Find latitude column
            latitude_candidates = [
                "latitude",
                "lat",
                "gps_latitude",
                "gps_lat"
            ]

            longitude_candidates = [
                "longitude",
                "lon",
                "lng",
                "gps_longitude",
                "gps_lon"
            ]

            lat_col = next(
                (c for c in latitude_candidates if c in survey_df.columns),
                None
            )

            lon_col = next(
                (c for c in longitude_candidates if c in survey_df.columns),
                None
            )

            if lat_col is None or lon_col is None:

                st.sidebar.error(
                    "CSV must contain latitude and longitude columns."
                )

                st.sidebar.write(
                    "Columns found:",
                    list(survey_df.columns)
                )

                survey_df = None

            else:

                survey_df[lat_col] = pd.to_numeric(
                    survey_df[lat_col],
                    errors="coerce"
                )

                survey_df[lon_col] = pd.to_numeric(
                    survey_df[lon_col],
                    errors="coerce"
                )

                survey_df = survey_df.dropna(
                    subset=[lat_col, lon_col]
                ).reset_index(drop=True)

                if len(survey_df) == 0:

                    st.sidebar.error(
                        "No valid GPS coordinates were found."
                    )

                    survey_df = None

                else:

                    # Store standardized names
                    survey_df["latitude"] = survey_df[lat_col]
                    survey_df["longitude"] = survey_df[lon_col]

                    st.sidebar.success(
                        f"Loaded {len(survey_df)} GPS positions"
                    )

        except Exception as e:

            st.sidebar.error(
                f"Could not read GPS CSV: {e}"
            )

else:

    st.sidebar.info(
        "Simulator GPS is for demonstration only."
    )

    simulator_lat = st.sidebar.number_input(
        "Latitude",
        value=11.341000,
        format="%.6f"
    )

    simulator_lon = st.sidebar.number_input(
        "Longitude",
        value=76.955000,
        format="%.6f"
    )
        # Optional AI packages:
# pip install transformers torch torchvision sentencepiece

st.set_page_config(
    page_title="Aquadex AI",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Aquadex AI")
st.caption("Underwater sonar image analysis")

# ----------------------------- 
# Load zero-shot image model
# -----------------------------
@st.cache_resource
def load_ai_model():
    from transformers import CLIPProcessor, CLIPModel

    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return processor, model


def classify_image(image, labels):
    import torch

    processor, model = load_ai_model()

    prompts = [
        f"a side scan sonar image showing {label}"
        for label in labels
    ]

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    results = []
    for label, probability in zip(labels, probs):
        results.append({
            "Type": label,
            "Confidence": float(probability)
        })

    results.sort(key=lambda x: x["Confidence"], reverse=True)
    return results
# ============================================================
# SONAR IMAGE VALIDATION
# ============================================================

def validate_sonar_image(image):
    """
    Dedicated SONAR vs NON-SONAR gate.

    This does NOT use the debris/object labels. It asks CLIP the
    same binary question several different ways and averages the
    pairwise sonar probabilities. This is more reliable than
    putting "unclear sonar scene" into the same competition.
    """

    import torch

    processor, model = load_ai_model()

    prompt_pairs = [
        (
            "a genuine side-scan sonar image",
            "a normal camera photograph"
        ),
        (
            "an acoustic sonar scan of the seafloor",
            "an ordinary underwater camera photograph"
        ),
        (
            "a marine sonar survey image",
            "a regular underwater photograph"
        ),
        (
            "a side-looking sonar / sidescan sonar image",
            "a non-sonar photograph"
        ),
        (
            "a grayscale sonar waterfall or seabed scan",
            "a normal grayscale photograph"
        ),
        (
            "an underwater acoustic imaging scan",
            "a conventional underwater photo"
        )
    ]

    sonar_probabilities = []

    for sonar_prompt, non_sonar_prompt in prompt_pairs:

        inputs = processor(
            text=[sonar_prompt, non_sonar_prompt],
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        pair_probability = (
            outputs.logits_per_image
            .softmax(dim=1)[0][0]
            .item()
        )

        sonar_probabilities.append(pair_probability)

    sonar_probability = sum(
        sonar_probabilities
    ) / len(sonar_probabilities)

    # For a hackathon prototype, 50% is the neutral point.
    # We use a moderate gate so genuine sonar imagery is not
    # rejected just because CLIP is uncertain.
    is_sonar = sonar_probability >= sonar_threshold

    return {
        "is_sonar": is_sonar,
        "sonar_probability": sonar_probability,
        "pair_scores": sonar_probabilities
    }



# ============================================================
# SIDEBAR - ANALYSIS SETTINGS
# ============================================================

st.sidebar.header("⚙️ Analysis Settings")

threshold = st.sidebar.slider(
    "Minimum object confidence",
    0.0,
    1.0,
    0.20,
    0.01
)

sonar_threshold = st.sidebar.slider(
    "Sonar validation threshold",
    0.45,
    0.80,
    0.52,
    0.01,
    help="Lower values accept uncertain sonar images; higher values make validation stricter."
)

st.sidebar.info(
    "Only images classified as sonar/side-scan sonar will continue "
    "to the debris analysis."
)
# ============================================================
# GPS VALUES - ALWAYS INITIALIZED
# ============================================================

latitude = 0.0
longitude = 0.0
gps_source = "No GPS"

if gps_mode == "Real GPS / Survey CSV" and survey_df is not None:
    latitude = float(survey_df.iloc[0]["latitude"])
    longitude = float(survey_df.iloc[0]["longitude"])
    gps_source = "Verified survey CSV"

elif gps_mode == "Simulator GPS":
    latitude = float(simulator_lat)
    longitude = float(simulator_lon)
    gps_source = "Simulator GPS"



# -----------------------------
# Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an underwater or side-scan sonar image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is None:
    st.info("Upload a sonar image to begin analysis.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("Input Image")
    st.image(image, use_container_width=True)

# ============================================================
# SONAR VALIDATION
# ============================================================

with st.spinner("🔎 Checking whether the image is a sonar image..."):

    sonar_check = validate_sonar_image(image)

sonar_probability = sonar_check["sonar_probability"]

# ------------------------------------------------------------
# IMPORTANT:
# Stop the application if the image is not sonar.
# No debris/object classification is performed.
# ------------------------------------------------------------

if sonar_probability < sonar_threshold:

    with col2:

        st.subheader("AI Analysis")

        st.error(
            "❌ Invalid image: This does not appear to be a "
            "sonar / side-scan sonar image."
        )

        st.metric(
            "Sonar confidence",
            f"{sonar_probability:.1%}"
        )

        st.warning(
            "Please upload a genuine underwater sonar or "
            "side-scan sonar image."
        )

        st.info(
            "Examples accepted: side-scan sonar scans, "
            "underwater acoustic sonar survey images, and "
            "seabed sonar imagery."
        )

    st.divider()

    st.subheader("🚫 Analysis stopped")

    st.write(
        "Aquadex AI does not run marine-debris classification "
        "on non-sonar images."
    )

    st.caption(
        "Note: This is a prototype sonar gate. It is not a "
        "sonar-trained scientific classifier."
    )

    st.stop()

# ============================================================
# SONAR ACCEPTED
# ============================================================

with col2:

    st.subheader("AI Analysis")

    st.success(
        f"✅ Sonar image accepted\n\n"
        f"**Sonar confidence:** {sonar_probability:.1%}"
    )

    st.caption(
        "Sonar validation uses six independent sonar-vs-photo "
        "comparisons and averages their results."
    )


# -----------------------------
# Classification
# -----------------------------
with st.spinner("AI is analysing the sonar image..."):

    # First: broad classification
    broad_labels = [
        "a man-made object or marine debris",
        "a natural underwater object",
        "an unclear or ambiguous sonar scene"
    ]

    broad_results = classify_image(image, broad_labels)

    broad_best = broad_results[0]
    broad_conf = broad_best["Confidence"]

    # Second: object type
    object_labels = [
        "a shipwreck",
        "an underwater pipe or pipeline",
        "plastic debris",
        "a fishing net or ghost net",
        "metal debris",
        "a concrete or artificial structure",
        "natural rock or seabed formation",
        "another man-made marine object",
        "an unclear sonar object"
    ]

    type_results = classify_image(image, object_labels)

    type_best = type_results[0]
    type_conf = type_best["Confidence"]


# -----------------------------
# Results
# -----------------------------
with col2:
    st.subheader("AI Analysis")

    broad_name = broad_best["Type"]
    broad_display = broad_name.replace("a ", "").replace("an ", "").capitalize()

    if broad_conf >= threshold:
        st.success(
            f"**Classification:** {broad_display}\n\n"
            f"**Confidence:** {broad_conf:.1%}"
        )
    else:
        st.warning(
            f"No broad classification above the selected threshold.\n\n"
            f"Top result: {broad_display} ({broad_conf:.1%})"
        )

    st.divider()

    type_name = type_best["Type"]
    type_display = type_name.replace("a ", "").replace("an ", "").capitalize()

    st.metric(
        "Most likely object type",
        type_display,
        f"{type_conf:.1%} confidence"
    )

    if type_conf >= threshold:
        st.info(
            f"**AI interpretation:** The image is most similar to "
            f"**{type_display}** among the requested categories."
        )
    else:
        st.warning(
            "The object-type confidence is below the selected threshold."
        )
# ============================================================
# SONAR VALIDATION DETAILS
# ============================================================

st.divider()
st.subheader("🛰️ Sonar Validation")

# ============================================================
# GOOGLE MAPS LOCATION LINK
# ============================================================

st.subheader("📍 Survey Location")

if gps_mode == "Real GPS / Survey CSV" and survey_df is not None:

    # Use the first valid GPS point from the survey CSV
    latitude = float(survey_df.iloc[0]["latitude"])
    longitude = float(survey_df.iloc[0]["longitude"])

    google_maps_url = (
        "https://www.google.com/maps/search/?api=1"
        f"&query={latitude},{longitude}"
    )

    st.markdown(
        f"🌍 [Open Survey Location in Google Maps]({google_maps_url})"
    )

    st.write(f"**Latitude:** {latitude:.6f}")
    st.write(f"**Longitude:** {longitude:.6f}")

elif gps_mode == "Simulator GPS":

    latitude = simulator_lat
    longitude = simulator_lon

    google_maps_url = (
        "https://www.google.com/maps/search/?api=1"
        f"&query={latitude},{longitude}"
    )

    st.markdown(
        f"🌍 [Open Simulator Location in Google Maps]({google_maps_url})"
    )

    st.write(f"**Latitude:** {latitude:.6f}")
    st.write(f"**Longitude:** {longitude:.6f}")


validation_table = pd.DataFrame([
    {
        "Check": "Sonar image",
        "Result": "PASS",
        "Confidence": f"{sonar_probability:.1%}"
    },
])


# -----------------------------
# Ranking
# -----------------------------
st.divider()
st.subheader("🔎 Object Type Comparison")

table = pd.DataFrame(type_results)
table["Confidence"] = table["Confidence"].map(lambda x: f"{x:.1%}")

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# Export
# -----------------------------
st.divider()
st.subheader("📊 Export Report")

report = {
    "project": "Aquadex AI",
    "image": uploaded_file.name,
    "timestamp": datetime.now().isoformat(),
     "sonar_validation": "PASS",
    "sonar_confidence": round(sonar_probability,4),
    "broad_classification": broad_display,
    "broad_confidence": round(broad_conf, 4),
    "object_type": type_display,
    "object_confidence": round(type_conf, 4),
    "latitude": latitude,
    "longitude": longitude,
    "gps_source": gps_source,
    "model": "OpenAI CLIP ViT-B/32 zero-shot classifier",
    "note": (
        "Prototype only. Zero-shot CLIP is not a sonar-trained "
        "scientific classifier."
    )
}

json_data = json.dumps(report, indent=4)

st.download_button(
    "⬇️ Download JSON report",
    data=json_data,
    file_name="aquadex_report.json",
    mime="application/json"
)
# ============================================================
# CSV REPORT
# ============================================================

# Convert report dictionary to a one-row DataFrame
csv_df = pd.DataFrame([report])

# Convert complex values (lists/dictionaries) to text
for col in csv_df.columns:
    csv_df[col] = csv_df[col].apply(
        lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
    )

csv_data = csv_df.to_csv(index=False)

st.download_button(
    label="📊 Download CSV Report",
    data=csv_data,
    file_name="aquadex_report.csv",
    mime="text/csv"
)
st.caption(
    "Important: This is a prototype. CLIP was not trained specifically "
    "for side-scan sonar, so its results should not be treated as verified "
    "marine-debris identification."
)