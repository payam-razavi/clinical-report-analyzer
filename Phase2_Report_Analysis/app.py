# =========================================================
# Clinical Report Analyzer
# Developed by Payam Razavi
# Cedars-Sinai / Los Angeles Pierce College AI Initiative
# =========================================================

import html
import re
from typing import List, Dict, Any

import joblib
import streamlit as st
from PyPDF2 import PdfReader


# =========================
# PAGE CONFIG
# =========================
# Configure the Streamlit page before any content is rendered.
# A wide layout gives more room for side-by-side prediction and keyword sections.
st.set_page_config(
    page_title="Clinical Report Analyzer",
    page_icon="🩺",
    layout="wide",
)


# =========================
# LOAD MODEL ASSETS
# =========================
@st.cache_resource
def load_assets():
    """
    Load the trained classification model and TF-IDF vectorizer.

    st.cache_resource prevents Streamlit from reloading these files every time
    the interface refreshes, which makes the app faster and more stable.
    """
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


# Load saved Phase 1 artifacts once and reuse them throughout the app.
loaded_model, loaded_vectorizer = load_assets()


# =========================
# TEXT PROCESSING
# =========================
def preprocess_text(text: str) -> str:
    """
    Normalize clinical report text before vectorization.

    This function matches the basic preprocessing used during model training:
    lowercase text, remove line breaks, and normalize extra whitespace.
    """
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract raw text from an uploaded PDF pathology report.

    PyPDF2 extracts text page by page. Some PDFs may extract imperfectly,
    so later cleaning and validation steps are used before model inference.
    """
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


def clean_extracted_report_text(text: str) -> str:
    """
    Clean extracted PDF text by keeping clinically relevant pathology sections
    and removing common administrative/header/footer content.

    This is a rule-based cleaner designed to work across different pathology
    report formats. It reduces noise such as page numbers, MRNs, signatures,
    addresses, disclaimers, and repeated report headers.
    """

    # Normalize line breaks and remove long separator lines commonly found in reports.
    text = text.replace("\r", "\n")
    text = re.sub(r"_+", "\n", text)
    text = re.sub(r"-{3,}", "\n", text)

    # Remove common administrative/header/footer lines before section extraction.
    lines = text.splitlines()
    cleaned_lines = []

    # These patterns are intentionally general rather than hospital-specific.
    # They target metadata and administrative content that usually does not help
    # cancer type classification or clinical interpretation.
    skip_patterns = [
        r"^page\s+\d+\s+of\s+\d+",
        r"^page\s+\d+",
        r"^mrn\b",
        r"^medical record",
        r"^patient name",
        r"^name:",
        r"^dob\b",
        r"^d\.o\.b\.",
        r"^date of birth",
        r"^billing",
        r"^account",
        r"^accession",
        r"^pathology no",
        r"^case no",
        r"^specimen no",
        r"^location:",
        r"^surgeon:",
        r"^attending:",
        r"^ordering physician",
        r"^provider",
        r"^collected:",
        r"^received:",
        r"^reported:",
        r"^report date",
        r"^procedure date",
        r"^copies to",
        r"^electronically signed",
        r"^signed out",
        r"^phone",
        r"^fax",
        r"^\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}",
        r"^\d{1,5}\s+.*\b(street|st\.|road|rd\.|avenue|ave\.|blvd|drive|dr\.)\b",
        r"\bdepartment of pathology\b",
        r"\blaboratory medicine\b",
        r"\bpathology report\b",
        r"\bfinal surgical pathology report\b",
        r"\bdisclaimer\b",
        r"\bthis test was developed\b",
        r"\bfood and drug administration\b",
        r"\bfda\b",
        r"\bcase reviewed\b",
    ]

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Skip lines that match common administrative/report metadata patterns.
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
            continue

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    # Try to start the cleaned text at the first clinically meaningful section.
    # This helps remove hospital headers that appear before the diagnosis section.
    start_markers = [
        "FINAL DIAGNOSIS:",
        "DIAGNOSIS:",
        "PATHOLOGIC DIAGNOSIS:",
        "SURGICAL PATHOLOGY DIAGNOSIS:",
        "CLINICAL DIAGNOSIS:",
        "CLINICAL INFORMATION:",
        "SPECIMEN:",
        "GROSS DESCRIPTION:",
        "MICROSCOPIC DESCRIPTION:",
        "COMMENT:",
        "CANCER SYNOPTIC SUMMARY REPORT:",
        "SYNOPTIC REPORT:",
        "PATHOLOGIC STAGING:",
    ]

    upper_text = cleaned_text.upper()
    start_positions = []

    for marker in start_markers:
        pos = upper_text.find(marker)
        if pos != -1:
            start_positions.append(pos)

    # If a clinical section marker is found, keep text from the earliest marker onward.
    if start_positions:
        cleaned_text = cleaned_text[min(start_positions):]

    # Remove common trailing legal/disclaimer/signature sections.
    # NOTE: We avoid stopping at "NOTE:" because pathology notes can contain
    # clinically meaningful diagnostic interpretation.
    stop_markers = [
        "DISCLAIMER:",
        "ADDENDUM DISCLAIMER:",
        "CONFIDENTIALITY NOTICE:",
        "THIS TEST WAS DEVELOPED",
        "THE FDA HAS DETERMINED",
        "CASE REVIEWED AT",
        "THE TEST(S) THAT ARE REPORTED HERE",
        "THIS REPORT MAY INCLUDE",
        "PARTIN PROGNOSTIC TABLES",
        "FINAL DIAGNOSIS ON",
        "ELECTRONICALLY SIGNED",
    ]

    upper_text = cleaned_text.upper()
    stop_positions = []

    for marker in stop_markers:
        pos = upper_text.find(marker)
        if pos != -1:
            stop_positions.append(pos)

    # If a non-clinical trailing marker is found, truncate before it.
    if stop_positions:
        cleaned_text = cleaned_text[:min(stop_positions)]

    return cleaned_text.strip()


def validate_pathology_report(text: str):
    """
    Check whether extracted text appears to be a real pathology report.

    This prevents the model from analyzing documents that are too short,
    generic templates, or instructional samples rather than actual diagnostic
    pathology content.
    """

    text_lower = text.lower()

    # Terms commonly found in real diagnostic pathology reports.
    clinical_terms = [
        "diagnosis", "final diagnosis", "histologic", "tumor", "carcinoma",
        "sarcoma", "lymph", "invasion", "specimen", "gross description",
        "microscopic", "staging", "malignant", "neoplasm", "mass"
    ]

    # Phrases that indicate the uploaded PDF is likely a template or educational sample.
    template_phrases = [
        "the final diagnosis made by the pathologist",
        "the note sometimes called comment",
        "this is a description of the specimen",
        "information the clinical team think",
        "specimen site and medical procedure",
        "name or signature of pathologist"
    ]

    clinical_score = sum(1 for term in clinical_terms if term in text_lower)
    template_score = sum(1 for phrase in template_phrases if phrase in text_lower)

    word_count = len(text.split())

    # Very short documents usually do not contain enough signal for reliable analysis.
    if word_count < 50:
        return False, "The extracted text is too short to analyze reliably."

    # Template/instructional PDFs should not be classified as real clinical reports.
    if template_score >= 2:
        return False, "This appears to be a template or instructional sample, not an actual pathology report."

    # If there are too few pathology-related terms, the content is likely not valid input.
    if clinical_score < 3:
        return False, "The document does not appear to contain enough diagnostic pathology content."

    return True, "Report appears valid for analysis."


def split_into_sentences(text: str) -> List[str]:
    """
    Split a pathology report into sentence-like units.

    This supports sentence highlighting by breaking the report into smaller
    sections that can be scored against extracted keywords.
    """
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in sentences if s.strip()]


# =========================
# MODEL OUTPUTS
# =========================
def extract_top_keywords(report_text: str, vectorizer, top_k: int = 10):
    """
    Extract the highest-scoring TF-IDF terms from a report.

    These terms provide a simple interpretability layer by surfacing words or
    phrases that are important in the report's TF-IDF representation.
    """
    clean_text = preprocess_text(report_text)
    text_vector = vectorizer.transform([clean_text])

    feature_names = vectorizer.get_feature_names_out()
    scores = text_vector.toarray()[0]

    # Sort feature indices by TF-IDF score from highest to lowest.
    top_indices = scores.argsort()[-top_k:][::-1]

    keywords = []
    for idx in top_indices:
        if scores[idx] > 0:
            keywords.append((feature_names[idx], scores[idx]))

    return keywords


def extract_top_keyword_list(report_text: str, vectorizer, top_k: int = 10) -> List[str]:
    """
    Return only the keyword strings from the TF-IDF keyword output.

    This simplified list is used for downstream sentence matching and
    highlighted evidence extraction.
    """
    keywords = extract_top_keywords(report_text, vectorizer, top_k=top_k)
    return [word for word, _ in keywords]


def highlight_top_sentences(
    report_text: str,
    vectorizer,
    top_k_keywords: int = 10,
    top_k_sentences: int = 3,
) -> List[Dict[str, Any]]:
    """
    Identify the most relevant report sentences based on keyword overlap.

    Sentences are scored by counting how many extracted keywords appear in
    each sentence. The highest-scoring sentences are returned as evidence for
    the model's output.
    """
    keywords = extract_top_keyword_list(report_text, vectorizer, top_k=top_k_keywords)
    sentences = split_into_sentences(report_text)

    scored_sentences = []

    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = 0
        matched_keywords = []

        for kw in keywords:
            if kw in sentence_lower:
                score += 1
                matched_keywords.append(kw)

        # Keep only sentences that match at least one extracted keyword.
        if score > 0:
            scored_sentences.append(
                {
                    "sentence": sentence,
                    "score": score,
                    "matched_keywords": matched_keywords,
                }
            )

    scored_sentences.sort(key=lambda x: x["score"], reverse=True)
    return scored_sentences[:top_k_sentences]


def analyze_report(
    report_text: str,
    top_k_predictions: int = 3,
    top_k_keywords: int = 10,
    top_k_sentences: int = 3,
) -> Dict[str, Any]:
    """
    Run the complete NLP analysis pipeline for one pathology report.

    The output includes:
    - top cancer type predictions with probabilities
    - important TF-IDF keywords
    - highlighted evidence sentences
    """
    clean_text = preprocess_text(report_text)
    text_vector = loaded_vectorizer.transform([clean_text])

    # Generate probability scores for each cancer class.
    probs = loaded_model.predict_proba(text_vector)[0]
    top_indices = probs.argsort()[-top_k_predictions:][::-1]

    predictions = []
    for idx in top_indices:
        predictions.append(
            {
                "cancer_type": loaded_model.classes_[idx],
                "probability": float(probs[idx]),
            }
        )

    # Extract interpretable keywords from the same report text.
    keywords = extract_top_keywords(report_text, loaded_vectorizer, top_k=top_k_keywords)
    keywords = [{"word": word, "score": float(score)} for word, score in keywords]

    # Select sentences that provide evidence for the extracted keywords.
    highlighted_sentences = highlight_top_sentences(
        report_text,
        loaded_vectorizer,
        top_k_keywords=top_k_keywords,
        top_k_sentences=top_k_sentences,
    )

    return {
        "predictions": predictions,
        "keywords": keywords,
        "highlighted_sentences": highlighted_sentences,
    }


def generate_plain_english_summary(result):
    """
    Generate a short plain-English summary from NLP analysis results.

    This is a rule-based summary, not medical advice. It uses the model's top
    prediction, confidence score, extracted keywords, and highlighted evidence
    to create a readable overview for the user.
    """

    top_prediction = result["predictions"][0]["cancer_type"]
    confidence = result["predictions"][0]["probability"]

    keywords = [k["word"] for k in result["keywords"][:5]]
    highlighted = result["highlighted_sentences"]

    summary_parts = []

    # Main model interpretation sentence.
    # "Most consistent with" is intentionally cautious wording.
    summary_parts.append(
        f"This pathology report is most consistent with "
        f"{top_prediction.lower()} "
        f"(model confidence: {confidence:.2f})."
    )

    # Include the most important extracted terms as a high-level finding summary.
    if keywords:
        keyword_text = ", ".join(keywords)
        summary_parts.append(
            f"The report contains clinically relevant findings related to "
            f"{keyword_text}."
        )

    # Mention that evidence is highlighted below when available.
    if highlighted:
        summary_parts.append(
            "The report contains diagnostically relevant findings "
            "highlighted from the pathology text."
        )

    return " ".join(summary_parts)


def highlight_sentences_in_report(report_text: str, highlighted_sentences: List[Dict[str, Any]]) -> str:
    """
    Render the full report as HTML with selected evidence sentences highlighted.

    The report text is HTML-escaped first for safety. Then each highlighted
    sentence is wrapped in a <mark> tag so it appears visually highlighted in
    the Streamlit interface.
    """
    escaped_report = html.escape(report_text)

    # Replace longer sentences first to avoid partial replacements.
    sorted_sentences = sorted(
        highlighted_sentences,
        key=lambda x: len(x["sentence"]),
        reverse=True,
    )

    highlighted_html = escaped_report

    for item in sorted_sentences:
        sentence = item["sentence"].strip()
        if not sentence:
            continue

        escaped_sentence = html.escape(sentence)
        replacement = (
            f'<mark style="background-color:#fff59d; padding:2px 4px; '
            f'border-radius:4px;">{escaped_sentence}</mark>'
        )
        highlighted_html = highlighted_html.replace(escaped_sentence, replacement)

    # Preserve original line breaks when displayed as HTML.
    highlighted_html = highlighted_html.replace("\n", "<br>")
    return highlighted_html


# =========================
# UI HELPERS
# =========================
def probability_bar(prob: float, label: str):
    """
    Display a prediction label with a Streamlit progress bar.
    """
    st.markdown(f"**{label}** — {prob:.3f}")
    st.progress(float(prob))


def metric_card(title: str, value: str):
    """
    Render a simple card-style metric using custom HTML.
    """
    st.markdown(
        f"""
        <div style="
            background-color:#f8f9fa;
            padding:16px;
            border-radius:12px;
            border:1px solid #e9ecef;
            margin-bottom:12px;
        ">
            <div style="font-size:0.95rem; color:#555;">{title}</div>
            <div style="font-size:1.2rem; font-weight:600; margin-top:6px;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# SIDEBAR
# =========================
# The sidebar gives users control over how many predictions, keywords,
# and highlighted evidence sentences are displayed.
st.sidebar.title("Settings")
top_k_predictions = st.sidebar.slider("Top predictions", min_value=1, max_value=5, value=3)
top_k_keywords = st.sidebar.slider("Top keywords", min_value=5, max_value=15, value=10)
top_k_sentences = st.sidebar.slider("Highlighted sentences", min_value=1, max_value=5, value=3)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**About**
- Upload a pathology report PDF
- Extract text
- Predict likely cancer types
- Surface important keywords
- Highlight relevant report sentences
"""
)

st.sidebar.markdown("""
Developed by Payam Razavi  
Cedars-Sinai / LA Pierce College AI Initiative
""")


# =========================
# MAIN LAYOUT
# =========================
st.title("🩺 Clinical Report Analyzer")
st.caption("Upload a pathology report PDF and analyze it with a trained clinical NLP pipeline.")

uploaded_file = st.file_uploader("Upload a pathology report (PDF)", type=["pdf"])

if uploaded_file is not None:
    try:
        # Step 1: Extract raw text from the uploaded PDF.
        raw_text = extract_text_from_pdf(uploaded_file)

        # Step 2: Remove common PDF noise and keep clinically relevant sections.
        report_text = clean_extracted_report_text(raw_text)

        # Stop early if the PDF does not contain extractable text.
        if not report_text.strip():
            st.error("No readable text was extracted from the PDF.")
            st.stop()

        # Step 3: Validate that the uploaded file appears to be a real pathology report.
        is_valid, validation_message = validate_pathology_report(report_text)

        if not is_valid:
            st.warning(validation_message)
            st.stop()

        # Let users inspect the cleaned text without cluttering the main interface.
        with st.expander("Cleaned extracted text preview", expanded=False):
            st.text(report_text[:2500])

        if st.button("Analyze Report", use_container_width=True):
            # Step 4: Run the complete NLP pipeline.
            result = analyze_report(
                report_text,
                top_k_predictions=top_k_predictions,
                top_k_keywords=top_k_keywords,
                top_k_sentences=top_k_sentences,
            )

            # Step 5: Prepare outputs for display.
            summary = generate_plain_english_summary(result)
            predictions = result["predictions"]
            keywords = result["keywords"]
            highlighted_sentences = result["highlighted_sentences"]

            # Top metrics provide a quick snapshot of the analysis result.
            st.markdown("## Results")
            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Top predicted cancer type", predictions[0]["cancer_type"])
            with c2:
                metric_card("Top confidence", f'{predictions[0]["probability"]:.3f}')
            with c3:
                metric_card("Highlighted sentences", str(len(highlighted_sentences)))

            # Plain-English summary gives users a high-level interpretation first.
            st.subheader("Plain-English Summary")
            st.info(summary)

            # Show prediction probabilities and extracted keywords side by side.
            left_col, right_col = st.columns([1.2, 1])

            with left_col:
                st.markdown("### Top Predictions")
                for item in predictions:
                    probability_bar(item["probability"], item["cancer_type"])

            with right_col:
                st.markdown("### Top Keywords")
                for item in keywords:
                    st.markdown(
                        f"- **{item['word']}** <span style='color:#666;'>({item['score']:.4f})</span>",
                        unsafe_allow_html=True,
                    )

            # Display the highest-scoring evidence sentences separately.
            st.markdown("### Highlighted Sentences")
            for idx, item in enumerate(highlighted_sentences, start=1):
                matched = ", ".join(item["matched_keywords"]) if item["matched_keywords"] else "None"
                st.markdown(
                    f"""
                    <div style="
                        background-color:#fffbea;
                        padding:14px;
                        border-radius:12px;
                        border-left:6px solid #f4c542;
                        margin-bottom:12px;
                    ">
                        <div style="font-weight:600; margin-bottom:8px;">Sentence {idx}</div>
                        <div style="margin-bottom:8px;">{html.escape(item["sentence"])}</div>
                        <div style="font-size:0.9rem; color:#555;">
                            <strong>Score:</strong> {item["score"]}<br>
                            <strong>Matched keywords:</strong> {html.escape(matched)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Render the full cleaned report with important sentences highlighted in context.
            st.markdown("### Report with Highlighted Sentences")
            highlighted_report_html = highlight_sentences_in_report(report_text, highlighted_sentences)

            st.markdown(
                f"""
                <div style="
                    background-color:#fafafa;
                    padding:18px;
                    border-radius:12px;
                    border:1px solid #ddd;
                    line-height:1.7;
                    font-size:0.98rem;
                ">
                    {highlighted_report_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        # Catch unexpected errors so the app fails gracefully instead of crashing.
        st.error(f"An error occurred while processing the report: {e}")

else:
    st.info("Upload a PDF report to begin analysis.")
