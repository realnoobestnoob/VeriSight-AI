"""VeriSight AI — AI-generated image detector. Launch: `streamlit run app.py`"""

from __future__ import annotations

import hashlib
import logging

import streamlit as st

from verisight_ai.config import AppSettings
from verisight_ai.database import make_repository
from verisight_ai.explainability import GRADCAM_CAVEAT
from verisight_ai.inference import ModelUnavailableError, Predictor, resolve_bundle_dir
from verisight_ai.logging_setup import setup_logging
from verisight_ai.preprocessing import InvalidImageError, decode_image_bytes
from verisight_ai.schemas import DecisionStatus

setup_logging()
log = logging.getLogger("verisight.app")
st.set_page_config(page_title="VeriSight AI", page_icon="🔎", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def get_predictor(model_dir: str | None, artifacts_dir: str):
    return Predictor(resolve_bundle_dir(model_dir, artifacts_dir))


@st.cache_resource
def get_repo(url: str | None):
    return make_repository(url)


settings = AppSettings.from_env()
repo = get_repo(settings.database_url)
st.session_state.setdefault("history", [])

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔎 VeriSight AI")
st.caption("AI-generated image detector")
st.info(
    "**About:** VeriSight AI uses a neural network to estimate whether an uploaded image "
    "resembles AI-generated content from its training data. Upload or photograph an image "
    "to get a result.\n\n"
    "**Disclaimer:** Results are probabilistic and not proof of authenticity. "
    "Do not use for legal, identity, security, or financial decisions."
)

# ── Input ──────────────────────────────────────────────────────────────────────
mode = st.radio(
    "Input method",
    ["Upload image", "Use camera"],
    horizontal=True,
    label_visibility="collapsed",
)

data, input_type = None, "upload"
if mode == "Use camera":
    shot = st.camera_input("Take a photo")
    if shot:
        data, input_type = shot.getvalue(), "camera_capture"
else:
    up = st.file_uploader("JPEG, PNG or WebP", type=["jpg", "jpeg", "png", "webp"])
    if up:
        data = up.getvalue()

st.button("Analyse", type="primary", disabled=data is None, key="analyse_btn")


# ── Analysis ───────────────────────────────────────────────────────────────────
def _record(input_type, file_type, file_hash, status, error=None, result=None):
    return repo.record_scan(
        input_type=input_type,
        file_type=file_type,
        file_hash=file_hash,
        predicted_class=result.predicted_class if result else None,
        raw_score=result.raw_score if result else None,
        calibrated_score=result.calibrated_score if result else None,
        decision_status=result.decision_status.value if result else None,
        model_release_id=result.model_release_id if result else None,
        training_run_id=result.training_run_id if result else None,
        inference_latency_ms=result.latency_ms if result else None,
        status=status,
        error_category=error,
        llm_used=False,
        provenance_status="not_checked",
    )


if st.session_state.get("analyse_btn"):
    try:
        decoded = decode_image_bytes(data, settings.max_upload_bytes, settings.max_pixels)
    except InvalidImageError as exc:
        _record(input_type, None, None, "rejected_input", "invalid_image")
        st.error(str(exc))
        decoded = None

    if decoded is not None:
        try:
            predictor = get_predictor(settings.model_dir, settings.artifacts_dir)
        except ModelUnavailableError as exc:
            _record(input_type, decoded.file_type, None, "failed", "model_unavailable")
            st.error("The model is not available right now. " + str(exc))
            predictor = None

        if predictor is not None:
            try:
                result = predictor.predict(decoded.image)
                overlay = predictor.explain(decoded.image)
            except Exception:  # noqa: BLE001
                log.exception("inference failed")
                _record(input_type, decoded.file_type, None, "failed", "internal_error")
                st.error("Something went wrong while analysing this image.")
                result = None

            if result is not None:
                scan_id = _record(
                    input_type,
                    decoded.file_type,
                    hashlib.sha256(data).hexdigest(),
                    "succeeded",
                    None,
                    result,
                )
                st.session_state["last"] = {
                    "result": result,
                    "overlay": overlay,
                    "original": decoded.image,
                    "scan_id": scan_id,
                }
                st.session_state["history"].append(
                    {
                        "verdict": result.decision_label,
                        "score": round(result.score_used, 3),
                    }
                )

# ── Results ────────────────────────────────────────────────────────────────────
last = st.session_state.get("last")
if last:
    res = last["result"]
    st.divider()

    icon = {
        DecisionStatus.SYNTHETIC_LIKE: "🟠",
        DecisionStatus.NO_STRONG_SIGNAL: "🟢",
        DecisionStatus.INCONCLUSIVE: "⚪",
    }[res.decision_status]

    # Verdict + score
    st.markdown(f"### {icon} {res.decision_label}")
    score_pct = int(res.score_used * 100)
    st.progress(res.score_used, text=f"AI likelihood score: **{score_pct}%**")
    score_kind = "calibrated" if res.score_label == "calibrated" else "raw (not a probability)"
    st.caption(
        f"Score type: {score_kind} · "
        f"Threshold: {res.threshold:.2f} · "
        f"Abstain band: ±{res.abstain_margin:.2f} · "
        f"Model: {res.model_release_id}"
    )

    # Visual explanation
    st.markdown("#### Where the model focused")
    col_a, col_b = st.columns(2)
    col_a.image(last["original"], caption="Original", use_container_width=True)
    col_b.image(last["overlay"], caption="Model focus area", use_container_width=True)
    st.caption(GRADCAM_CAVEAT)

    # Feedback
    if repo.enabled and last.get("scan_id"):
        st.markdown("**Was this result useful?**")
        fb1, fb2, fb3 = st.columns(3)
        for col, label, key in (
            (fb1, "👍 Yes", "helpful"),
            (fb2, "👎 No", "not_helpful"),
            (fb3, "🤔 Unsure", "unsure"),
        ):
            if col.button(label, key=f"fb_{key}"):
                ok = repo.record_feedback(last["scan_id"], key)
                st.success("Thanks for the feedback.") if ok else st.warning(
                    "Feedback could not be saved."
                )

# Session history
if st.session_state["history"]:
    with st.expander("This session's scans"):
        st.dataframe(st.session_state["history"], use_container_width=True)
