from __future__ import annotations

import json
import math
from pathlib import Path, PureWindowsPath
from typing import List, Tuple

import pandas as pd
import streamlit as st


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS_PATH = Path("./outputs/stage_predictions.json")


def page_setup() -> None:
    """Configure global Streamlit page settings."""
    st.set_page_config(
        page_title="Stage Prediction Viewer",
        layout="wide",
        page_icon="🩺",
    )
    st.title("Stage Prediction Viewer")
    st.caption("Inspect stage predictions, confidence scores, and preview testing images.")


def normalize_image_path(raw_image: str, source_file: Path, image_root: Path | None = None) -> Path:
    """
    Convert an image path coming from Windows or relative JSON paths to a usable Path.

    - Windows absolute paths (e.g., C:\\...) are mapped to /mnt/<drive>/... for WSL.
    - Relative paths are resolved against the predictions file directory.
    - When the above do not exist, we try the user-provided image root (helps on Streamlit Cloud).
    """
    raw_str = str(raw_image or "").strip()
    if not raw_str:
        return Path()

    candidates: list[Path] = []

    path = Path(raw_str).expanduser()
    windows_path = PureWindowsPath(raw_str)
    is_windows_abs = bool(windows_path.drive)

    # POSIX-style absolute path
    if path.is_absolute() and not is_windows_abs:
        candidates.append(path)

    if is_windows_abs:
        # Convert "C:\\path\\to\\file" -> "/mnt/c/path/to/file" when running under WSL.
        drive_letter = windows_path.drive.rstrip(":").lower()
        candidates.append(Path("/mnt") / drive_letter / Path(*windows_path.parts[1:]))
    else:
        if not path.is_absolute():
            candidates.append(source_file.parent / path)

    if image_root:
        if is_windows_abs:
            candidates.append(image_root / Path(*windows_path.parts[1:]))
            candidates.append(image_root / windows_path.name)
        else:
            candidates.append(image_root / path)
            candidates.append(image_root / Path(raw_str).name)

    if source_file.parent.name == "outputs":
        candidates.append(source_file.parent.parent / "out_dir" / "images" / Path(raw_str).name)

    # Fallback to common in-repo locations (useful on Streamlit Cloud).
    candidates.append(APP_ROOT / "out_dir" / "images" / Path(raw_str).name)
    candidates.append(APP_ROOT.parent / "out_dir" / "images" / Path(raw_str).name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[-1] if candidates else Path()


@st.cache_data(show_spinner=False)
def load_predictions(path_str: str | Path) -> pd.DataFrame:
    """Load predictions from a JSON file into a DataFrame."""
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Predictions JSON must be a list of objects.")
    frame = pd.DataFrame(data)
    if "confidence" in frame.columns:
        frame["confidence"] = frame["confidence"].astype(float)
    return frame


def resolve_stage_column(frame: pd.DataFrame) -> Tuple[str, str]:
    """
    Decide which column represents the stage label.

    Returns a tuple of (display_column, fallback_column_name).
    """
    if "stage_value" in frame.columns:
        return "stage_value", "pred_stage" if "pred_stage" in frame.columns else "stage_value"
    if "pred_stage" in frame.columns:
        return "pred_stage", "pred_stage"
    raise KeyError("Predictions are missing both 'stage_value' and 'pred_stage' columns.")


def sidebar_controls(frame: pd.DataFrame) -> Tuple[pd.DataFrame, int, int, str]:
    """Render sidebar filters and return the filtered DataFrame with pagination state."""
    stage_column, raw_stage_column = resolve_stage_column(frame)
    stage_values = sorted(frame[stage_column].dropna().unique().tolist())

    st.sidebar.header("Filters")
    selected_stages = st.sidebar.multiselect(
        "Stage filter",
        options=stage_values,
        default=stage_values,
        help="Select which stages to include in the gallery.",
    )

    has_confidence = "confidence" in frame.columns and frame["confidence"].notna().any()
    confidence_range: Tuple[float, float] | None = None
    if has_confidence:
        min_conf = float(frame["confidence"].min())
        max_conf = float(frame["confidence"].max())
        if math.isclose(min_conf, max_conf):
            st.sidebar.info(f"All confidences are {min_conf:.2f}.")
            confidence_range = (min_conf, max_conf)
        else:
            confidence_range = st.sidebar.slider(
                "Confidence range",
                value=(min_conf, max_conf),
                min_value=min_conf,
                max_value=max_conf,
                step=0.01,
            )

    search_term = st.sidebar.text_input(
        "Search by filename",
        value="",
        help="Filter images whose filename contains this text (case insensitive).",
    ).strip().lower()

    images_per_page = int(
        st.sidebar.number_input(
            "Images per page",
            min_value=3,
            max_value=30,
            value=9,
            step=3,
            help="Adjust how many images to display at once.",
            format="%d",
        )
    )

    filtered = frame.copy()
    if selected_stages:
        filtered = filtered[filtered[stage_column].isin(selected_stages)]
    else:
        filtered = filtered.iloc[0:0]

    if has_confidence and confidence_range:
        low, high = confidence_range
        filtered = filtered[(filtered["confidence"] >= low) & (filtered["confidence"] <= high)]

    if search_term:
        filtered = filtered[
            filtered["image"].astype(str).str.lower().str.contains(search_term, na=False)
        ]

    total_pages = max(1, math.ceil(len(filtered) / images_per_page))
    selected_page = int(
        st.sidebar.slider(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
        )
    )

    st.sidebar.caption(f"{len(filtered)} image(s) match the current filters.")

    return filtered, images_per_page, selected_page, raw_stage_column


def render_stage_summary(frame: pd.DataFrame, stage_column: str) -> None:
    """Display a quick summary of predictions per stage and confidence stats."""
    cols = st.columns(3)
    cols[0].metric("Total predictions", f"{len(frame):,}")
    if "confidence" in frame.columns and frame["confidence"].notna().any():
        cols[1].metric("Average confidence", f"{frame['confidence'].mean():.1%}")
        cols[2].metric("Top confidence", f"{frame['confidence'].max():.1%}")

    st.subheader("Stage distribution")
    counts = (
        frame[stage_column]
        .value_counts(dropna=False)
        .rename_axis("Stage")
        .to_frame("Predictions")
        .sort_index()
    )
    st.dataframe(counts, use_container_width=True)


def render_gallery(
    frame: pd.DataFrame,
    images_per_page: int,
    page: int,
    stage_column: str,
    predictions_file: Path,
    image_root: Path | None,
) -> None:
    """Render a paginated gallery of prediction thumbnails."""
    if frame.empty:
        st.info("No images to display with the selected filters.")
        return

    start = (page - 1) * images_per_page
    stop = start + images_per_page
    subset = frame.iloc[start:stop]

    num_columns = min(3, images_per_page)
    rows = math.ceil(len(subset) / num_columns)

    for row_idx in range(rows):
        columns = st.columns(num_columns)
        for col_idx, column in enumerate(columns):
            index = row_idx * num_columns + col_idx
            if index >= len(subset):
                column.empty()
                continue
            record = subset.iloc[index]
            image_path = normalize_image_path(str(record["image"]), predictions_file, image_root)
            if image_path.exists():
                column.image(str(image_path), use_column_width=True)
            else:
                column.warning(f"Image not found:\n{image_path}")
            stage_value = record.get(stage_column, "N/A")
            confidence = record.get("confidence")
            caption_lines: List[str] = [
                f"Prediction: stage {stage_value}",
            ]
            if "pred_stage" in record and stage_column != "pred_stage":
                caption_lines.append(f"Raw index: {record.get('pred_stage', '—')}")
            if confidence is not None and not pd.isna(confidence):
                caption_lines.append(f"Confidence: {confidence:.1%}")
            column.caption(" | ".join(caption_lines))

            if isinstance(record.get("probabilities"), list):
                with column.expander("Probabilities", expanded=False):
                    prob_series = pd.Series(record["probabilities"], name="Probability")
                    prob_series.index.name = "Class index"
                    prob_df = prob_series.reset_index().set_index("Class index")
                    st.bar_chart(prob_df)


def render_table(frame: pd.DataFrame) -> None:
    """Show the underlying prediction records."""
    st.subheader("Prediction details")
    display = frame.copy()
    if "confidence" in display.columns:
        display["confidence"] = display["confidence"].map(lambda x: f"{x:.4f}")
    st.dataframe(display, use_container_width=True)


def main() -> None:
    """Entrypoint for the Streamlit app."""
    page_setup()

    st.sidebar.header("Data source")
    default_path = DEFAULT_PREDICTIONS_PATH if DEFAULT_PREDICTIONS_PATH.exists() else ""
    predictions_path = st.sidebar.text_input(
        "Predictions JSON path",
        value=str(default_path),
        help="Provide the path to the JSON file created by batch_infer_stage.py.",
    ).strip()

    if not predictions_path:
        st.info("Provide a path to a predictions JSON file to get started.")
        return

    try:
        predictions_file = Path(predictions_path).expanduser().resolve()
        predictions = load_predictions(predictions_file)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except ValueError as exc:
        st.error(f"Failed to parse predictions JSON: {exc}")
        return

    # When deploying to Streamlit Cloud, allow overriding where images live.
    default_image_root = None
    for candidate in (
        predictions_file.parent / "images",
        predictions_file.parent.parent / "out_dir" / "images",
        APP_ROOT / "out_dir" / "images",
        APP_ROOT.parent / "out_dir" / "images",
    ):
        if candidate.exists():
            default_image_root = candidate
            break

    image_root_input = st.sidebar.text_input(
        "Image root directory (optional)",
        value=str(default_image_root or ""),
        help="Set if your predictions JSON points to image paths that don't exist locally (e.g., Windows paths). "
        "Provide a folder containing the images; filenames will be matched automatically.",
    ).strip()
    image_root = Path(image_root_input).expanduser().resolve() if image_root_input else default_image_root

    filtered, per_page, page, stage_column = sidebar_controls(predictions)

    render_stage_summary(filtered if not filtered.empty else predictions, stage_column)
    render_gallery(filtered, per_page, page, stage_column, predictions_file, image_root)
    render_table(filtered)


if __name__ == "__main__":
    main()
