import streamlit as st
import numpy as np
import pandas as pd

def topsis(matrix, weights, bencos, targets=None):
    matrix = np.array(matrix, dtype=float)
    weights = np.array(weights, dtype=float)

    if targets is not None:
        for j, t in enumerate(targets):
            if bencos[j] == "Match":
                try:
                    matrix[:, j] = np.abs(matrix[:, j] - float(t))
                except:
                    pass  # if invalid target, skip

    normalized = matrix / np.sqrt((matrix ** 2).sum(axis=0))
    weighted = normalized * weights

    ideal, negideal = np.zeros(weighted.shape[1]), np.zeros(weighted.shape[1])
    for j in range(weighted.shape[1]):
        if bencos[j] == "Benefit":
            ideal[j], negideal[j] = np.max(weighted[:, j]), np.min(weighted[:, j])
        elif bencos[j] == "Cost":
            ideal[j], negideal[j] = np.min(weighted[:, j]), np.max(weighted[:, j])
        else:  # Match
            ideal[j], negideal[j] = np.min(weighted[:, j]), np.max(weighted[:, j])

    sepideal = np.linalg.norm(weighted - ideal, axis=1)
    sepnegideal = np.linalg.norm(weighted - negideal, axis=1)
    finalscore = sepnegideal / (sepideal + sepnegideal)

    return finalscore

st.set_page_config(page_title="TOPSIS", page_icon="📊")
st.title("Technique for Order Preference by Similarity to Ideal Solution (TOPSIS)")

# Inputs
altamt = st.number_input("Number of Alternatives", min_value=2, max_value=99, value=3)
critamt = st.number_input("Number of Main Criteria", min_value=1, max_value=20, value=2)

# Alternatives
st.header("Alternatives")
alt_names = st.text_area("Alternative Names (one per line)",
    "\n".join([f"Alternative {i+1}" for i in range(altamt)])
).splitlines()
alt_names = [name.strip() or f"Alternative {i+1}" for i, name in enumerate(alt_names)]
if len(alt_names) != altamt:
    st.warning(f"Please provide exactly {altamt} alternative names.")

# Main Criterion Table
st.header("Main Criteria")
main_df = pd.DataFrame({
    "Main Criterion": [f"Criterion {i+1}" for i in range(critamt)],
    "Weight": [round(1/critamt, 3)] * critamt,
    "Type": ["Benefit"] * critamt,
    "Target": [None] * critamt
})
main_df = st.data_editor(
    main_df,
    column_config={
        "Type": st.column_config.SelectboxColumn("Type", options=["Benefit", "Cost", "Match"]),
    },
    num_rows="dynamic",
    key="main_criterion"
)
use_subcriterion = st.checkbox("Use Sub-Criteria?", value=False)

# Sub-Criterion Table
if use_subcriterion:
    st.header("Sub-Criterion per Main Criterion")
    subcrit_counts = {
        row["Main Criterion"]: st.number_input(
            f"Number of Sub-Criterion for '{row['Main Criterion']}'", min_value=1, max_value=10, value=2, key=f"count_{row['Main Criterion']}"
        ) for _, row in main_df.iterrows()
    }

    sub_rows = []
    for _, row in main_df.iterrows():
        main_crit, main_weight, main_type = row["Main Criterion"], row["Weight"], row["Type"]
        count = subcrit_counts[main_crit]
        for i in range(count):
            sub_rows.append({
                "Main Criterion": main_crit,
                "Main Weight": main_weight,
                "Sub-Criterion": f"{main_crit} - Sub-{i+1}",
                "Sub Weight": round(1 / count, 3),
                "Type": main_type,
                "Target": None
            })

    sub_df = pd.DataFrame(sub_rows)
    sub_df = st.data_editor(
        sub_df,
        num_rows="fixed",
        key="subcriterion",
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["Benefit", "Cost", "Match"]),
        }
    )

    # Compute effective weights and targets
    crit_names, weights, types, targets = [], [], [], []
    for main_crit in main_df["Main Criterion"]:
        rows = sub_df[sub_df["Main Criterion"] == main_crit]
        main_weight = main_df.loc[main_df["Main Criterion"] == main_crit, "Weight"].values[0]
        total_sub_weight = rows["Sub Weight"].sum()
        for _, sub in rows.iterrows():
            crit_names.append(sub["Sub-Criterion"])
            norm_w = sub["Sub Weight"] / total_sub_weight if total_sub_weight else 0
            weights.append(main_weight * norm_w)
            types.append(sub["Type"])
            targets.append(sub["Target"])
else:
    crit_names = main_df["Main Criterion"].tolist()
    weights = main_df["Weight"].tolist()
    types = main_df["Type"].tolist()
    targets = main_df["Target"].tolist()

# Scores input
st.header("Input or Upload CSV for Scores")
uploaded_file = st.file_uploader("Upload a CSV with alternatives as rows and criteria as columns", type="csv")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df.index = df.iloc[:, 0]
        df.drop(df.columns[0], axis=1, inplace=True)

        if list(df.columns) != crit_names:
            st.warning(f"CSV columns do not match expected criteria: {crit_names}")
            scores_df = pd.DataFrame(0.0, index=alt_names, columns=crit_names)
        else:
            scores_df = df
            alt_names = df.index.tolist()
            st.success("CSV loaded successfully.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        scores_df = pd.DataFrame(0.0, index=alt_names, columns=crit_names)
else:
    scores_df = pd.DataFrame(0.0, index=alt_names, columns=crit_names)

edited_scores = st.data_editor(scores_df, key="scores")

# Compute
if st.button("Compute TOPSIS"):
    try:
        matrix = edited_scores.values.astype(float)
        weights = np.array(weights, dtype=float)
        if weights.sum(): weights /= weights.sum()
        scores = topsis(matrix, weights, types, targets)

        result_df = pd.DataFrame({
            "Alternative": alt_names,
            "Score": scores
        }).sort_values(by="Score", ascending=False)
        result_df["Rank"] = range(1, len(result_df) + 1)

        st.success("TOPSIS calculation complete.")
        st.dataframe(result_df[["Rank", "Alternative", "Score"]])
        st.subheader("Scores Visualization")
        st.bar_chart(result_df.set_index("Alternative")["Score"])

        best = result_df.iloc[0]
        st.markdown(f"### ✅ Best Alternative: **{best['Alternative']}** with score **{best['Score']:.4f}**")
    except Exception as e:
        st.error(f"Error in calculation: {e}")
