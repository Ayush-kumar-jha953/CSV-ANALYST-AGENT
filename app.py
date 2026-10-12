## df-pandas 
## agent - [llm,tools,prompt]
## ui

import os
import uuid
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver


NO_DATA = "No data available. Please upload a CSV file first."
PLOT_DIR = "plots"
MAX_ONEHOT_CARDINALITY = 20


@st.cache_resource
def get_store() -> dict:
    return {"df": None, "original_df": None, "pending_plots": []}


STORE = get_store()


def get_df():
    return STORE["df"]


def set_df(df: pd.DataFrame):
    STORE["df"] = df


def _clip(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"


def _numeric_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(include="number").columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def _resolve_columns(columns: str, pool: list):
    """'all' -> every column in pool; otherwise a comma separated list.
    Returns (valid_columns, invalid_columns)."""
    columns = (columns or "all").strip()
    if columns.lower() == "all":
        return list(pool), []
    requested = [c.strip() for c in columns.split(",") if c.strip()]
    return [c for c in requested if c in pool], [c for c in requested if c not in pool]


def _iqr_bounds(s: pd.Series):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr, iqr


def _save_fig(fig, name: str) -> str:
    """Save a figure and queue it so the Streamlit UI can display it."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, f"{name}_{uuid.uuid4().hex[:8]}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    STORE["pending_plots"].append(path)
    return path


# ----------------------------------------------------------------------------
# Original tools (small fixes: typo'd name, missing else branch, output limits)
# ----------------------------------------------------------------------------
@tool
def get_data_info(request: str = "summary") -> str:
    """Get shape, columns, data types, missing values and the first 5 rows of the
    uploaded CSV. Call this first when unsure about the structure of the data."""
    df = get_df()
    if df is None:
        return NO_DATA
    return _clip(
        f"shape: {df.shape}\n"
        f"columns and data types:\n{df.dtypes.to_string()}\n"
        f"missing values:\n{df.isna().sum().to_string()}\n"
        f"first 5 rows:\n{df.head().to_string()}"
    )


@tool
def filter_rows(condition: str) -> str:
    """Filter rows using a pandas query condition and return the matches.
    Example: "Age > 30 and Department == 'Engineering'" """
    df = get_df()
    if df is None:
        return NO_DATA
    try:
        result = df.query(condition)
        return _clip(f"Found {len(result)} rows. First 20:\n{result.head(20).to_string()}")
    except Exception as e:
        return f"Error filtering data: {e}"


@tool
def analyze_data(expression: str) -> str:
    """Run a pandas expression on the dataframe (referred to as 'df').
    Examples: df['Age'].mean()   |   df.groupby('Department')['Salary'].sum()"""
    df = get_df()
    if df is None:
        return NO_DATA
    try:
        result = eval(expression, {"df": df, "pd": pd, "np": np})
        return _clip(str(result))
    except Exception as e:
        return f"Error analyzing data: {e}"


# ----------------------------------------------------------------------------
# 1. Outlier tools
# ----------------------------------------------------------------------------
@tool
def detect_outliers(columns: str = "all") -> str:
    """Report outliers in numerical columns WITHOUT changing the data.
    Uses the IQR rule (1.5 x IQR) and the z-score rule (|z| > 3).
    columns: 'all' or comma separated numeric column names."""
    df = get_df()
    if df is None:
        return NO_DATA
    cols, invalid = _resolve_columns(columns, _numeric_cols(df))
    if not cols:
        return f"No valid numeric columns found. Invalid/non-numeric: {invalid}"

    rows = []
    for c in cols:
        s = df[c].dropna()
        if s.nunique() <= 2:  # binary / flag columns have no meaningful outliers
            continue
        lo, hi, _ = _iqr_bounds(s)
        iqr_count = int(((s < lo) | (s > hi)).sum())
        z_count = 0 if s.std() == 0 else int((((s - s.mean()) / s.std()).abs() > 3).sum())
        rows.append(
            {
                "column": c,
                "lower_bound": round(lo, 3),
                "upper_bound": round(hi, 3),
                "iqr_outliers": iqr_count,
                "iqr_pct": round(100 * iqr_count / len(s), 2),
                "zscore_outliers": z_count,
                "skew": round(s.skew(), 2),
            }
        )
    if not rows:
        return "No eligible numeric columns (binary columns are skipped)."
    msg = pd.DataFrame(rows).to_string(index=False)
    if invalid:
        msg += f"\nSkipped (not numeric / not found): {invalid}"
    return msg


@tool
def treat_outliers(
    method: Literal["cap", "median", "remove_iqr", "remove_zscore"] = "cap",
    columns: str = "all",
) -> str:
    """Treat outliers in numerical columns and update the working dataframe.
    method:
      cap          - clip values to the IQR bounds (winsorizing; keeps all rows, safest default)
      median       - replace outliers with the column median
      remove_iqr   - drop rows that are IQR outliers in any selected column
      remove_zscore- drop rows with |z| > 3 in any selected column
    columns: 'all' or comma separated numeric column names."""
    df = get_df()
    if df is None:
        return NO_DATA
    df = df.copy()
    cols, invalid = _resolve_columns(columns, _numeric_cols(df))
    if not cols:
        return f"No valid numeric columns found. Invalid/non-numeric: {invalid}"

    rows_before = len(df)
    drop_mask = pd.Series(False, index=df.index)
    changes = {}

    for c in cols:
        s = df[c]
        valid = s.dropna()
        if valid.nunique() <= 2:
            continue
        lo, hi, iqr = _iqr_bounds(valid)

        if method == "remove_zscore":
            std = valid.std()
            out = pd.Series(False, index=df.index) if std == 0 else ((s - valid.mean()) / std).abs() > 3
        else:
            if iqr == 0:
                continue
            out = (s < lo) | (s > hi)

        n_out = int(out.sum())
        if n_out == 0:
            continue
        changes[c] = n_out

        if method == "cap":
            df[c] = s.clip(lo, hi)
        elif method == "median":
            df[c] = s.mask(out, valid.median())
        else:
            drop_mask |= out

    if method.startswith("remove"):
        df = df[~drop_mask].reset_index(drop=True)

    set_df(df)
    if not changes:
        return "No outliers found in the selected columns. Data unchanged."
    msg = f"Applied '{method}'. Outliers found per column: {changes}. "
    msg += f"Rows: {rows_before} -> {len(df)}."
    if invalid:
        msg += f" Skipped (not numeric / not found): {invalid}"
    return msg


# ----------------------------------------------------------------------------
# 2. Encoding tool
# ----------------------------------------------------------------------------
@tool
def encode_data(
    method: Literal["label", "onehot", "ordinal", "frequency"] = "label",
    columns: str = "all",
    order: str = "",
) -> str:
    """Encode categorical (text) columns to numbers and update the working dataframe.
    method:
      label     - each category -> integer code (good for tree models / target-like columns)
      onehot    - one 0/1 column per category, first dropped (good for linear models);
                  columns with more than 20 categories are skipped
      ordinal   - categories with a natural order; give the order low->high in `order`,
                  e.g. order='Low,Medium,High' (applied to every selected column)
      frequency - each category -> its relative frequency (good for high-cardinality columns)
    columns: 'all' or comma separated categorical column names."""
    df = get_df()
    if df is None:
        return NO_DATA
    df = df.copy()
    cols, invalid = _resolve_columns(columns, _categorical_cols(df))
    if not cols:
        return f"No categorical columns to encode. Invalid/non-categorical: {invalid}"

    notes = []

    if method == "label":
        for c in cols:
            cat = df[c].astype("category")
            codes = cat.cat.codes.replace(-1, np.nan)
            mapping = dict(enumerate(cat.cat.categories))
            df[c] = codes
            shown = dict(list(mapping.items())[:10])
            notes.append(f"{c}: {shown}{' ...' if len(mapping) > 10 else ''}")

    elif method == "onehot":
        ok = [c for c in cols if df[c].nunique() <= MAX_ONEHOT_CARDINALITY]
        skipped = [c for c in cols if c not in ok]
        if skipped:
            notes.append(
                f"Skipped high-cardinality columns (> {MAX_ONEHOT_CARDINALITY} categories): {skipped}. "
                "Consider 'frequency' or 'label' for these."
            )
        if ok:
            df = pd.get_dummies(df, columns=ok, drop_first=True, dtype=int)
            notes.append(f"One-hot encoded: {ok}. New shape: {df.shape}")
        else:
            return "Nothing encoded. " + " ".join(notes)

    elif method == "ordinal":
        levels = [v.strip() for v in order.split(",") if v.strip()]
        if len(levels) < 2:
            return "For ordinal encoding, pass `order` as comma separated categories from lowest to highest."
        mapping = {v: i for i, v in enumerate(levels)}
        for c in cols:
            mapped = df[c].astype(str).str.strip().map(mapping)
            unmapped = int(mapped.isna().sum() - df[c].isna().sum())
            df[c] = mapped
            notes.append(f"{c}: mapped with {mapping}" + (f" ({unmapped} values not in order -> NaN)" if unmapped else ""))

    elif method == "frequency":
        for c in cols:
            df[c] = df[c].map(df[c].value_counts(normalize=True))
            notes.append(f"{c}: frequency encoded")

    set_df(df)
    msg = f"Applied '{method}' encoding to {cols}.\n" + "\n".join(notes)
    if invalid:
        msg += f"\nSkipped (not categorical / not found): {invalid}"
    return _clip(msg)


# ----------------------------------------------------------------------------
# 3. Graph tools (figures are queued and displayed by the Streamlit UI)
# ----------------------------------------------------------------------------
@tool
def plot_numerical(column: str = "all") -> str:
    """Plot numerical data. For one column: histogram with density curve plus a boxplot.
    For 'all': a grid of histograms (max 12 numeric columns).
    Returns summary statistics so you can explain the distribution (skew, spread, outliers)."""
    df = get_df()
    if df is None:
        return NO_DATA
    num = _numeric_cols(df)
    if not num:
        return "No numeric columns to plot."

    sns.set_theme(style="whitegrid")
    if column.strip().lower() == "all":
        cols = num[:12]
        ncols = 3
        nrows = int(np.ceil(len(cols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        for ax, c in zip(np.array(axes).ravel(), cols):
            s = df[c].dropna()
            sns.histplot(s, kde=s.nunique() > 1, ax=ax)
            ax.set_title(c)
        for ax in np.array(axes).ravel()[len(cols):]:
            ax.axis("off")
        fig.tight_layout()
        _save_fig(fig, "numerical_overview")
        stats = df[cols].agg(["mean", "median", "std", "skew"]).round(2).T
        return "Histogram grid shown to the user.\n" + stats.to_string()

    if column not in num:
        return f"'{column}' is not a numeric column. Numeric columns: {num}"
    s = df[column].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(s, kde=s.nunique() > 1, ax=axes[0])
    axes[0].set_title(f"Distribution of {column}")
    sns.boxplot(x=s, ax=axes[1])
    axes[1].set_title(f"Boxplot of {column}")
    fig.tight_layout()
    _save_fig(fig, f"numerical_{column}")
    lo, hi, _ = _iqr_bounds(s)
    n_out = int(((s < lo) | (s > hi)).sum())
    return (
        f"Plot shown to the user. {column}: mean={s.mean():.2f}, median={s.median():.2f}, "
        f"std={s.std():.2f}, min={s.min():.2f}, max={s.max():.2f}, skew={s.skew():.2f}, "
        f"IQR outliers={n_out}"
    )


@tool
def plot_categorical(column: str = "all", top_n: int = 10) -> str:
    """Plot categorical data as bar charts of category counts.
    column: one categorical column, or 'all' for a grid (max 9 columns with <= 50 categories).
    top_n: number of most frequent categories to show per chart."""
    df = get_df()
    if df is None:
        return NO_DATA
    cat = _categorical_cols(df)
    if not cat:
        return "No categorical columns to plot."

    sns.set_theme(style="whitegrid")
    if column.strip().lower() == "all":
        cols = [c for c in cat if df[c].nunique() <= 50][:9]
        if not cols:
            return "All categorical columns have more than 50 categories; plot them one at a time."
        ncols = 3
        nrows = int(np.ceil(len(cols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows))
        for ax, c in zip(np.array(axes).ravel(), cols):
            vc = df[c].value_counts().head(top_n)
            sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax)
            ax.set_title(c)
        for ax in np.array(axes).ravel()[len(cols):]:
            ax.axis("off")
        fig.tight_layout()
        _save_fig(fig, "categorical_overview")
        summary = {c: f"{df[c].nunique()} categories, top='{df[c].mode().iloc[0]}'" for c in cols}
        return f"Bar chart grid shown to the user. {summary}"

    if column not in cat:
        return f"'{column}' is not a categorical column. Categorical columns: {cat}"
    vc = df[column].value_counts()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * min(top_n, len(vc)) + 1.5)))
    sns.barplot(x=vc.head(top_n).values, y=vc.head(top_n).index.astype(str), ax=ax)
    ax.set_title(f"Top {min(top_n, len(vc))} categories of {column}")
    ax.set_xlabel("count")
    fig.tight_layout()
    _save_fig(fig, f"categorical_{column}")
    return f"Plot shown to the user. {column} has {len(vc)} categories. Counts:\n{vc.head(top_n).to_string()}"


@tool
def plot_correlation() -> str:
    """Plot a correlation heatmap of all numeric columns and return the strongest
    correlated pairs. Useful for spotting multicollinearity before model building."""
    df = get_df()
    if df is None:
        return NO_DATA
    num = _numeric_cols(df)
    if len(num) < 2:
        return "Need at least 2 numeric columns for a correlation heatmap."
    num = num[:30]
    corr = df[num].corr()
    size = max(6, 0.6 * len(num))
    fig, ax = plt.subplots(figsize=(size + 2, size))
    sns.heatmap(corr, annot=len(num) <= 12, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    _save_fig(fig, "correlation")
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().dropna()
    top = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(5).round(2)
    return "Heatmap shown to the user. Strongest pairs:\n" + top.to_string()


@tool
def plot_numeric_by_category(numeric_column: str, categorical_column: str) -> str:
    """Boxplot of a numeric column split by the top categories of a categorical column.
    Shows how a numeric variable differs between groups."""
    df = get_df()
    if df is None:
        return NO_DATA
    if numeric_column not in _numeric_cols(df):
        return f"'{numeric_column}' is not numeric."
    if categorical_column not in df.columns:
        return f"'{categorical_column}' not found."
    top = df[categorical_column].value_counts().head(10).index
    sub = df[df[categorical_column].isin(top)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=sub, x=categorical_column, y=numeric_column, ax=ax)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"{numeric_column} by {categorical_column}")
    fig.tight_layout()
    _save_fig(fig, f"{numeric_column}_by_{categorical_column}")
    means = sub.groupby(categorical_column)[numeric_column].mean().round(2)
    return f"Plot shown to the user. Mean {numeric_column} per group:\n{means.to_string()}"


@tool
def reset_data() -> str:
    """Undo all outlier treatment and encoding by restoring the originally uploaded CSV."""
    original = STORE["original_df"]
    if original is None:
        return NO_DATA
    set_df(original.copy())
    return f"Data restored to the original upload. Shape: {original.shape}"


all_tools = [
    get_data_info,
    filter_rows,
    analyze_data,
    detect_outliers,
    treat_outliers,
    encode_data,
    plot_numerical,
    plot_categorical,
    plot_correlation,
    plot_numeric_by_category,
    reset_data,
]

system_prompt = """You are a data analyst and ML-preparation assistant. You work on a pandas dataframe
called 'df' that the user uploaded. Use your tools; never guess numbers.

Tools:
- get_data_info: shape, dtypes, missing values, first rows. Use first if unsure about the structure.
- filter_rows: filter rows with a pandas query string.
- analyze_data: run a pandas expression on df.
- detect_outliers: report outliers (IQR and z-score) without changing data.
- treat_outliers: cap / median / remove outliers. Changes the working data.
- encode_data: label / onehot / ordinal / frequency encoding. Changes the working data.
- plot_numerical, plot_categorical, plot_correlation, plot_numeric_by_category: draw graphs.
  Graphs are displayed to the user automatically, so do not describe pixels or paste file paths;
  explain what the graph shows using the statistics the tool returns.
- reset_data: restore the original uploaded data.

Guidelines:
- Typical ML-prep order: inspect -> detect outliers -> treat outliers -> encode categoricals.
  Detect before you treat, and tell the user which method you chose and why.
- Prefer 'cap' for outliers unless the user asks otherwise, since it keeps all rows.
- For encoding, suggest onehot for low-cardinality columns with linear models, label for tree models,
  ordinal only when the categories have a real order, frequency for high-cardinality columns.
- After any tool that changes the data, state clearly what changed (columns, row counts).
- If a tool reports an error or skipped columns, relay that honestly.
"""


@st.cache_resource
def get_agent():
    llm = ChatGroq(model="openai/gpt-oss-20b")
    return create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )


def run_agent(query: str) -> str:
    try:
        res = get_agent().invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"configurable": {"thread_id": st.session_state["thread_id"]}},
        )
        return res["messages"][-1].content
    except Exception as e:
        return f"Something went wrong while running the agent: {e}"


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="CSV Analyst Agent", layout="wide")
    st.header("CSV ANALYST AGENT")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))

    file = st.file_uploader(label="Upload a CSV file", type=["csv"])
    if file:
        file_key = f"{file.name}-{file.size}"
        if st.session_state.get("file_key") != file_key:  # load only when a NEW file arrives
            df = pd.read_csv(file)
            st.session_state["file_key"] = file_key
            STORE["original_df"] = df.copy()
            STORE["pending_plots"] = []
            st.session_state["messages"] = []
            st.session_state["thread_id"] = str(uuid.uuid4())  # fresh chat memory per file
            set_df(df)

    df = get_df()
    if df is not None:
        with st.expander(f"Current data: {df.shape[0]} rows x {df.shape[1]} columns", expanded=True):
            st.dataframe(df.head())
            st.download_button(
                "Download processed CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="processed_data.csv",
                mime="text/csv",
            )

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            for path in m.get("images", []):
                if os.path.exists(path):
                    st.image(path)

    query = st.chat_input("Ask a question about the data")
    if query:
        if get_df() is None:
            st.warning("Please upload a CSV file first.")
            return
        st.chat_message("user").markdown(query)
        st.session_state["messages"].append({"role": "user", "content": query})
        STORE["pending_plots"] = []
        with st.spinner("Analysing..."):
            answer = run_agent(query)
        plots = list(STORE["pending_plots"])
        STORE["pending_plots"] = []
        st.session_state["messages"].append({"role": "assistant", "content": answer, "images": plots})
        st.rerun()  


if __name__ == "__main__":
    main()