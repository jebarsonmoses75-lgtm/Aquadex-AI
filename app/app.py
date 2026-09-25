import streamlit as st
import pandas as pd
import json

from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aquadex AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# COMPACT / SINGLE-SCREEN UI
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .block-container {
        padding-top: 0.7rem;
        padding-bottom: 0.5rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
        max-width: 1500px;
    }

    /* Reduce title spacing */
    h1 {
        margin-top: 0rem !important;
        margin-bottom: 0.1rem !important;
        font-size: 2.5rem !important;
        max-width: 100% !important;
    }

    h2 {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
        font-size: 1.35rem !important;
    }

    h3 {
        margin-top: 0.25rem !important;
        margin-bottom: 0.25rem !important;
        font-size: 1.05rem !important;
    }

    /* Reduce alert size */
    div[data-testid="stAlert"] {
        padding: 0.45rem 0.7rem !important;
        margin-bottom: 0.35rem !important;
    }

    /* Reduce metrics */
    div[data-testid="stMetric"] {
        padding: 0.15rem !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
    }


    /* Horizontal lines */
    hr {
        margin-top: 0.4rem !important;
        margin-bottom: 0.4rem !important;
    }

    /* Expander */
    details {
        margin-top: 0.25rem !important;
        margin-bottom: 0.25rem !important;
    }

    /* Caption */
    .stCaption {
        margin-top: 0rem !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title("🌊 Aquadex AI")

st.caption(
    "Underwater sonar image analysis • AI debris detection • GPS survey tagging"
)


# ============================================================
# LOAD CLIP MODEL
# ============================================================

@st.cache_resource
def load_ai_model():

    from transformers import CLIPProcessor, CLIPModel

    model_name = "openai/clip-vit-base-patch32"

    processor = CLIPProcessor.from_pretrained(model_name)

    model = CLIPModel.from_pretrained(model_name)

    model.eval()

    return processor, model


# ============================================================
# LOAD YOLO DEBRIS MODEL
# ============================================================

@st.cache_resource
def load_debris_model():

    """
    Searches for best.pt in common project locations.
    """

    from ultralytics import YOLO

    base_dir = Path(__file__).resolve().parent

    possible_paths = [

        # Same folder as app.py
        base_dir / "best.pt",

        # Parent folder
        base_dir.parent / "best.pt",

        # Current working directory
        Path("best.pt"),

        # Common folders
        base_dir.parent / "models" / "best.pt",
        base_dir.parent / "model" / "best.pt",
        base_dir.parent / "weights" / "best.pt",

        # app/models
        base_dir.parent / "app" / "best.pt",

    ]

    for model_path in possible_paths:

        if model_path.exists():

            model = YOLO(str(model_path))

            return model, str(model_path)

    return None, None


# ============================================================
# YOLO DEBRIS DETECTION
# ============================================================

def detect_debris(image, confidence=0.25):

    model, model_path = load_debris_model()

    if model is None:
        return image.copy(), [], None

    # YOLO prediction
    results = model.predict(
        source=image,
        conf=confidence,
        iou=0.50,
        max_det=1,
        agnostic_nms=True,
        imgsz=640,
        verbose=False
    )

    if not results:
        return image.copy(), [], model_path

    result = results[0]

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    detections = []

    # No detection
    if result.boxes is None or len(result.boxes) == 0:
        return annotated, detections, model_path

    # Highest-confidence detection
    best_index = 0

    if len(result.boxes) > 1:
        confidences = result.boxes.conf.tolist()
        best_index = confidences.index(max(confidences))

    box = result.boxes[best_index]

    xyxy = box.xyxy[0].tolist()

    x1, y1, x2, y2 = [
        int(v)
        for v in xyxy
    ]

    conf = float(box.conf[0])
    cls_id = int(box.cls[0])

    # Class name
    names = model.names

    if isinstance(names, dict):
        class_name = names.get(
            cls_id,
            f"Class {cls_id}"
        )
    else:
        class_name = names[cls_id]

    # Detection information
    detection = {
        "type": str(class_name),
        "confidence": conf,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": x2 - x1,
        "height": y2 - y1,
    }

    detections.append(detection)

    # ========================================================
    # RED BOUNDING BOX
    # ========================================================

    box_width = max(
        5,
        int(min(image.size) / 160)
    )

    draw.rectangle(
        [x1, y1, x2, y2],
        outline=(255, 0, 0),
        width=box_width
    )

    # ========================================================
    # MODERATE FONT
    # ========================================================

    font_size = max(
        28,
        int(min(image.size) / 32)
    )

    font = None

    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf"
    ]

    for font_path in font_paths:

        try:
            font = ImageFont.truetype(
                font_path,
                font_size
            )
            break

        except Exception:
            pass

    if font is None:
        font = ImageFont.load_default()

    # ========================================================
    # LABEL
    # ========================================================

    label = f"{class_name} {conf:.0%}"

    bbox = draw.textbbox(
        (0, 0),
        label,
        font=font
    )

    label_w = bbox[2] - bbox[0]
    label_h = bbox[3] - bbox[1]

    # Put label above the box
    label_y = max(
        0,
        y1 - label_h - 8
    )

    # ========================================================
    # RED LABEL BACKGROUND
    # ========================================================

    draw.rectangle(
        [
            x1,
            label_y,
            x1 + label_w + 14,
            label_y + label_h + 10
        ],
        fill=(255, 0, 0)
    )

    # ========================================================
    # WHITE LABEL TEXT
    # ========================================================

    draw.text(
        (
            x1 + 7,
            label_y + 4
        ),
        label,
        fill=(255, 255, 255),
        font=font
    )

    return (
        annotated,
        detections,
        model_path
    )
# ============================================================
# CLIP CLASSIFICATION
# ============================================================

def classify_image(image, labels):

    import torch

    processor, model = load_ai_model()


    prompts = [

        f"{label}"

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

        probabilities = (
            outputs.logits_per_image
            .softmax(dim=1)[0]
        )


    results = []


    for label, probability in zip(
        labels,
        probabilities
    ):

        results.append({

            "Type": label,

            "Confidence": float(
                probability
            )

        })


    results.sort(

        key=lambda x: x["Confidence"],

        reverse=True
    )


    return results


# ============================================================
# SONAR VALIDATION
# ============================================================

def validate_sonar_image(
    image,
    sonar_threshold=0.52
):

    """
    Determines whether an uploaded image resembles
    side-scan / underwater sonar imagery.

    Uses six independent CLIP comparisons and averages them.
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
            "a side-looking sonar image",
            "a non-sonar photograph"
        ),

        (
            "a grayscale sonar seabed scan",
            "a normal grayscale photograph"
        ),

        (
            "an underwater acoustic imaging scan",
            "a conventional underwater photo"
        )

    ]


    sonar_probabilities = []


    for (
        sonar_prompt,
        non_sonar_prompt
    ) in prompt_pairs:


        inputs = processor(

            text=[
                sonar_prompt,
                non_sonar_prompt
            ],

            images=image,

            return_tensors="pt",

            padding=True
        )


        with torch.no_grad():

            outputs = model(**inputs)


        pair_probability = (

            outputs
            .logits_per_image
            .softmax(dim=1)[0][0]
            .item()

        )


        sonar_probabilities.append(
            pair_probability
        )


    sonar_probability = (

        sum(sonar_probabilities)
        /
        len(sonar_probabilities)

    )


    is_sonar = (

        sonar_probability
        >=
        sonar_threshold

    )


    return {

        "is_sonar": is_sonar,

        "sonar_probability":
            sonar_probability,

        "pair_scores":
            sonar_probabilities
    }


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ Analysis Settings")


# ------------------------------------------------------------
# SINGLE CONFIDENCE SLIDER
# ------------------------------------------------------------

confidence_threshold = st.sidebar.slider(

    "Detection confidence",

    min_value=0.10,

    max_value=0.90,

    value=0.25,

    step=0.05,

    help=(
        "Only detections above this confidence "
        "are displayed."
    )
)


# ------------------------------------------------------------
# FIXED SONAR THRESHOLD
# ------------------------------------------------------------

SONAR_THRESHOLD = 0.52


st.sidebar.info(

    "Only one confidence bar is used. "
    "The highest-confidence debris detection "
    "is shown with one RED bounding box."
)


# ============================================================
# GEO-TAGGING / SURVEY GPS
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "📍 Geo-tagging"
)


gps_mode = st.sidebar.radio(

    "GPS Mode",

    [
        "Real GPS / Survey CSV",
        "Simulator GPS"
    ],

    index=0
)


survey_df = None


# ============================================================
# REAL GPS CSV
# ============================================================

if gps_mode == "Real GPS / Survey CSV":

    st.sidebar.info(

        "Upload the GPS/navigation CSV "
        "recorded during the sonar survey."
    )


    gps_file = st.sidebar.file_uploader(

        "Upload survey GPS CSV",

        type=["csv"],

        key="survey_gps_csv"
    )


    if gps_file is not None:

        try:

            survey_df = pd.read_csv(
                gps_file
            )


            # Normalize column names
            survey_df.columns = [

                str(c)
                .strip()
                .lower()
                .replace(" ", "_")

                for c in survey_df.columns

            ]


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

                (
                    c

                    for c
                    in latitude_candidates

                    if c in survey_df.columns

                ),

                None

            )


            lon_col = next(

                (
                    c

                    for c
                    in longitude_candidates

                    if c in survey_df.columns

                ),

                None

            )


            if (
                lat_col is None
                or
                lon_col is None
            ):

                st.sidebar.error(

                    "CSV must contain "
                    "latitude and longitude columns."
                )


                st.sidebar.write(

                    "Columns found:",

                    list(
                        survey_df.columns
                    )

                )


                survey_df = None


            else:

                survey_df[
                    lat_col
                ] = pd.to_numeric(

                    survey_df[
                        lat_col
                    ],

                    errors="coerce"
                )


                survey_df[
                    lon_col
                ] = pd.to_numeric(

                    survey_df[
                        lon_col
                    ],

                    errors="coerce"
                )


                survey_df = (

                    survey_df

                    .dropna(
                        subset=[
                            lat_col,
                            lon_col
                        ]
                    )

                    .reset_index(
                        drop=True
                    )

                )


                if len(survey_df) == 0:

                    st.sidebar.error(

                        "No valid GPS coordinates "
                        "were found."
                    )

                    survey_df = None


                else:

                    survey_df[
                        "latitude"
                    ] = survey_df[
                        lat_col
                    ]


                    survey_df[
                        "longitude"
                    ] = survey_df[
                        lon_col
                    ]


                    st.sidebar.success(

                        f"Loaded "
                        f"{len(survey_df)} "
                        f"GPS positions"
                    )


        except Exception as e:

            st.sidebar.error(

                f"Could not read GPS CSV: {e}"
            )


# ============================================================
# SIMULATOR GPS
# ============================================================

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


# ============================================================
# GPS VALUES
# ============================================================

latitude = 0.0

longitude = 0.0

gps_source = "No GPS"


if (

    gps_mode
    ==
    "Real GPS / Survey CSV"

    and

    survey_df is not None

):

    latitude = float(

        survey_df.iloc[0][
            "latitude"
        ]

    )


    longitude = float(

        survey_df.iloc[0][
            "longitude"
        ]

    )


    gps_source = (
        "Verified survey CSV"
    )


elif gps_mode == "Simulator GPS":

    latitude = float(
        simulator_lat
    )

    longitude = float(
        simulator_lon
    )

    gps_source = (
        "Simulator GPS"
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(

    "📤 Upload an underwater or side-scan sonar image",

    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ]
)


if uploaded_file is None:

    st.info(
        "Upload a sonar image to begin analysis."
    )

    st.stop()


# ============================================================
# OPEN IMAGE
# ============================================================

try:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

except Exception:

    st.error(
        "❌ Could not read this image. "
        "Please upload a valid image."
    )

    st.stop()


# ============================================================
# SONAR VALIDATION
# ============================================================

with st.spinner(
    "🔎 Validating sonar image..."
):

    sonar_check = validate_sonar_image(

        image,

        sonar_threshold=SONAR_THRESHOLD
    )


sonar_probability = (

    sonar_check[
        "sonar_probability"
    ]

)


# ============================================================
# INVALID IMAGE
# ============================================================

if sonar_probability < SONAR_THRESHOLD:

    left, right = st.columns(
        [1.15, 1]
    )


    with left:

        st.subheader(
            "🛰️ Input Image"
        )

        st.image(

            image,

            use_container_width=True
        )


    with right:

        st.subheader(
            "🤖 AI Analysis"
        )


        st.error(

            "❌ Invalid image\n\n"
            "This does not appear to be "
            "a sonar / side-scan sonar image."
        )


        st.metric(

            "Sonar confidence",

            f"{sonar_probability:.1%}"
        )


        st.warning(

            "Please upload a genuine "
            "underwater sonar or "
            "side-scan sonar image."
        )


    st.stop()


# ============================================================
# YOLO DETECTION
# ============================================================

with st.spinner(
    "🎯 Detecting debris..."
):

    (
        annotated_image,
        debris_detections,
        debris_model_path
    ) = detect_debris(

        image,

        confidence=confidence_threshold
    )


# ============================================================
# CLASSIFICATION
# ============================================================

with st.spinner(
    "🤖 AI is analysing the sonar image..."
):


    # --------------------------------------------------------
    # BROAD CLASSIFICATION
    # --------------------------------------------------------

    broad_labels = [

        "a man-made object or marine debris",

        "a natural underwater object",

        "an unclear or ambiguous sonar scene"

    ]


    broad_results = classify_image(

        image,

        broad_labels
    )


    broad_best = broad_results[0]

    broad_conf = (
        broad_best[
            "Confidence"
        ]
    )


    # --------------------------------------------------------
    # OBJECT TYPE CLASSIFICATION
    # --------------------------------------------------------

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


    type_results = classify_image(

        image,

        object_labels
    )


    type_best = type_results[0]

    type_conf = (
        type_best[
            "Confidence"
        ]
    )


# ============================================================
# DISPLAY NAMES
# ============================================================

broad_name = broad_best["Type"]


broad_display = (

    broad_name

    .replace(
        "a ",
        ""
    )

    .replace(
        "an ",
        ""
    )

    .capitalize()

)


type_name = type_best["Type"]


type_display = (

    type_name

    .replace(
        "a ",
        ""
    )

    .replace(
        "an ",
        ""
    )

    .capitalize()

)


# ============================================================
# SINGLE-SCREEN MAIN DASHBOARD
# ============================================================

st.divider()


left_col, right_col = st.columns(

    [1.15, 1],

    gap="medium"
)


# ============================================================
# LEFT SIDE - IMAGE
# ============================================================

with left_col:

    st.subheader(
        "🛰️ Sonar Image"
    )


    if debris_detections:

        st.image(

            annotated_image,

            caption=(
                "ONE highest-confidence "
                "debris detection"
            ),

            use_container_width=True
        )

    else:

        st.image(

            image,

            caption=(
                "No debris detected "
                "above selected confidence"
            ),

            use_container_width=True
        )


# ============================================================
# RIGHT SIDE - AI RESULTS
# ============================================================

with right_col:

    st.subheader(
        "🤖 AI Analysis"
    )


    # --------------------------------------------------------
    # SONAR RESULT
    # --------------------------------------------------------

    st.success(

        f"✅ **Sonar image accepted**\n\n"
        f"**Sonar confidence:** "
        f"{sonar_probability:.1%}"
    )


    # --------------------------------------------------------
    # DEBRIS RESULT
    # --------------------------------------------------------

    if debris_detections:

        best_detection = (
            debris_detections[0]
        )


        detected_type = (
            best_detection["type"]
            .title()
        )


        detected_confidence = (
            best_detection["confidence"]
        )


        st.markdown(
            "### 🎯 Debris Detection"
        )


        st.success(

            f"**{detected_type}**\n\n"
            f"Confidence: "
            f"**{detected_confidence:.1%}**"
        )


        st.caption(
            "Only the highest-confidence "
            "detection is displayed."
        )


    elif debris_model_path is None:

        st.error(
            "❌ `best.pt` not found."
        )


        st.caption(

            "Place best.pt in the same "
            "folder as app.py."
        )


    else:

        st.warning(

            "No debris detected above "
            f"{confidence_threshold:.0%}."
        )


    # --------------------------------------------------------
    # OBJECT CLASSIFICATION
    # --------------------------------------------------------

    st.markdown(
        "### 🔎 AI Classification"
    )


    st.info(

        f"**{type_display}**\n\n"
        f"Confidence: **{type_conf:.1%}**"
    )


    # --------------------------------------------------------
    # GPS
    # --------------------------------------------------------

    st.markdown(
        "### 📍 Survey Location"
    )


    gps_col1, gps_col2 = st.columns(2)


    with gps_col1:

        st.metric(

            "Latitude",

            f"{latitude:.6f}"
        )


    with gps_col2:

        st.metric(

            "Longitude",

            f"{longitude:.6f}"
        )


    st.caption(
        f"GPS source: {gps_source}"
    )


# ============================================================
# COMPACT DETAILS
# ============================================================

st.divider()


# ============================================================
# DETECTION DETAILS
# ============================================================

with st.expander(
    "🔎 Detection Details"
):


    if debris_detections:

        detection_table = pd.DataFrame([

            {

                "Debris type":
                    d["type"],

                "Confidence":
                    f'{d["confidence"]:.1%}',

                "X":
                    d["x1"],

                "Y":
                    d["y1"],

                "Width":
                    d["width"],

                "Height":
                    d["height"]

            }

            for d in debris_detections

        ])


        st.dataframe(

            detection_table,

            use_container_width=True,

            hide_index=True
        )


    else:

        st.write(
            "No debris detections."
        )


# ============================================================
# OBJECT TYPE COMPARISON
# ============================================================

with st.expander(
    "📊 Object Type Comparison"
):


    table = pd.DataFrame(
        type_results
    )


    table["Confidence"] = (

        table["Confidence"]

        .map(
            lambda x:
            f"{x:.1%}"
        )

    )


    st.dataframe(

        table,

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# SONAR VALIDATION DETAILS
# ============================================================

with st.expander(
    "🛰️ Sonar Validation"
):


    validation_table = pd.DataFrame([

        {

            "Check":
                "Sonar image",

            "Result":
                "PASS",

            "Confidence":
                f"{sonar_probability:.1%}"

        },

        {

            "Check":
                "Object analysis",

            "Result":
                "RUN",

            "Confidence":
                f"{type_conf:.1%}"

        }

    ])


    st.dataframe(

        validation_table,

        use_container_width=True,

        hide_index=True
    )


    st.caption(

        "Sonar validation uses six "
        "independent sonar-vs-photo "
        "comparisons and averages "
        "their results."
    )
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





# ============================================================
# EXPORT REPORT
# ============================================================

with st.expander(
    "📥 Export Report"
):


    report = {

        "project":
            "Aquadex AI",

        "image":
            uploaded_file.name,

        "timestamp":
            datetime.now().isoformat(),

        "sonar_validation":
            "PASS",

        "sonar_confidence":
            round(
                sonar_probability,
                4
            ),

        "broad_classification":
            broad_display,

        "broad_confidence":
            round(
                broad_conf,
                4
            ),

        "object_type":
            type_display,

        "object_confidence":
            round(
                type_conf,
                4
            ),

        "latitude":
            latitude,

        "longitude":
            longitude,

        "gps_source":
            gps_source,

        "debris_detections":
            debris_detections,

        "debris_model":
            (
                debris_model_path
                if debris_model_path
                else
                "best.pt not found"
            ),

        "model":
            (
                "OpenAI CLIP ViT-B/32 "
                "zero-shot classifier + "
                "YOLO debris detector"
            ),

        "detection_settings":
            {

                "confidence":
                    confidence_threshold,

                "iou":
                    0.50,

                "max_detections":
                    1

            },

        "note":
            (
                "Prototype only. CLIP was "
                "not trained specifically "
                "for side-scan sonar or "
                "marine debris."
            )

    }


    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_data = json.dumps(

        report,

        indent=4
    )


    st.download_button(

        "⬇️ Download JSON Report",

        data=json_data,

        file_name=(
            "aquadex_report.json"
        ),

        mime="application/json"
    )


    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_report = {

        "Project":
            report["project"],

        "Image":
            report["image"],

        "Timestamp":
            report["timestamp"],

        "Sonar Validation":
            report["sonar_validation"],

        "Sonar Confidence":
            report["sonar_confidence"],

        "Classification":
            report["broad_classification"],

        "Classification Confidence":
            report["broad_confidence"],

        "Object Type":
            report["object_type"],

        "Object Confidence":
            report["object_confidence"],

        "Latitude":
            report["latitude"],

        "Longitude":
            report["longitude"],

        "GPS Source":
            report["gps_source"],

        "Number of Debris Detections":
            len(
                debris_detections
            ),

        "YOLO Model":
            report["debris_model"]

    }


    csv_df = pd.DataFrame(
        [csv_report]
    )


    csv_data = csv_df.to_csv(
        index=False
    )


    st.download_button(

        "⬇️ Download CSV Report",

        data=csv_data,

        file_name=(
            "aquadex_report.csv"
        ),

        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "🌊 Aquadex AI • Sonar validation + "
    "YOLO debris detection + CLIP classification + GPS tagging"
)