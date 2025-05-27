import streamlit as st
import numpy as np
import pandas as pd

def wp(matrix, weights, types, targets):
    norm = np.copy(matrix).astype(float)
    for i in range(matrix.shape[1]):
        if types[i] == 'Cost':
            # Avoid division by zero or negative values
            with np.errstate(divide='ignore', invalid='ignore'):
                norm[:, i] = np.where(matrix[:, i] > 0, 1 / matrix[:, i], 0)
        elif types[i] == 'Match':
            # Use provided target or max value if None
            target = targets[i] if targets[i] is not None else np.max(matrix[:, i])
            # Avoid division by zero target
            if target == 0:
                norm[:, i] = 1.0  # or 0, depending on logic
            else:
                norm[:, i] = 1 - np.abs(matrix[:, i] - target) / target
    # Log transform
    log_matrix = np.log(np.clip(norm, a_min=1e-10, a_max=None))  # clip to avoid log(0)
    raw_scores = np.exp(np.sum(log_matrix * weights, axis=1))
    normalized_scores = raw_scores / raw_scores.sum()
    return normalized_scores

st.set_page_config(page_title="WP", page_icon="📊")
st.title('Weighted Product (WP)')

# Input settings
altamt = st.number_input("Number of Alternatives", min_value=2, max_value=99, value=3)
criteria_count = st.number_input("Number of Main Criteria", min_value=1, max_value=20, value=2)


# Input alternative names
st.header("Alternatives")
default_alt_names = [f"Alternative {i+1}" for i in range(altamt)]
alt_names_text = st.text_area("Alternative Names (one per line)", "\n".join(default_alt_names))
alt_names = [name.strip() if name.strip() else f"Alternative {i+1}" for i, name in enumerate(alt_names_text.splitlines())]

if len(alt_names) != altamt:
    st.warning(f"Please provide exactly {altamt} alternative names.")

# Input main criteria
st.header("Main Criteria")
main_criteria_df = pd.DataFrame({
    "Main Criterion": [f"Criterion {i+1}" for i in range(criteria_count)],
    "Weight": [round(1/criteria_count, 3)] * criteria_count,
    "Type": ["Benefit"] * criteria_count,
    "Target": [None] * criteria_count
})
main_criteria_df = st.data_editor(
    main_criteria_df,
    num_rows="dynamic",
    column_config={
        "Type": st.column_config.SelectboxColumn("Type", options=["Benefit", "Cost", "Match"])
    },
    key="main_criteria"
)
use_subcriteria = st.checkbox("Use Sub-Criteria?", value=False)

# Handle subcriteria
if use_subcriteria:
    st.header("Sub-Criterion per Main Criterion")
    subcriteria_counts = {}
    for crit in main_criteria_df["Main Criterion"]:
        subcriteria_counts[crit] = st.number_input(
            f"Number of Sub-Criteria for '{crit}'", min_value=1, max_value=10, value=2, key=f"count_{crit}"
        )

    subcriteria_rows = []
    for _, row in main_criteria_df.iterrows():
        main_crit = row["Main Criterion"]
        main_weight = row["Weight"]
        main_type = row["Type"]
        main_target = row["Target"]
        count = subcriteria_counts[main_crit]
        for i in range(count):
            subcriteria_rows.append({
                "Main Criterion": main_crit,
                "Main Weight": main_weight,
                "Sub-Criterion": f"{main_crit} - Sub-{i+1}",
                "Sub Weight": round(1 / count, 3),
                "Type": main_type,
                "Target": main_target
            })

    subcriteria_df = pd.DataFrame(subcriteria_rows)
    subcriteria_df = st.data_editor(
        subcriteria_df,
        num_rows="fixed",
        key="subcriteria",
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["Benefit", "Cost", "Match"])
        }
    )

    # Compute effective weights, types, names, and targets
    effective_weights = []
    types = []
    criteria_names = []
    targets = []
    for main_crit in main_criteria_df["Main Criterion"]:
        sub_rows = subcriteria_df[subcriteria_df["Main Criterion"] == main_crit]
        sub_weight_sum = sub_rows["Sub Weight"].sum()
        main_weight = main_criteria_df.loc[main_criteria_df["Main Criterion"] == main_crit, "Weight"].values[0]
        for _, sub_row in sub_rows.iterrows():
            norm_sub_weight = sub_row["Sub Weight"] / sub_weight_sum if sub_weight_sum != 0 else 0
            effective_weights.append(main_weight * norm_sub_weight)
            types.append(sub_row["Type"])
            criteria_names.append(sub_row["Sub-Criterion"])
            targets.append(sub_row["Target"] if pd.notna(sub_row["Target"]) else None)
else:
    effective_weights = main_criteria_df["Weight"].values
    types = main_criteria_df["Type"].tolist()
    criteria_names = main_criteria_df["Main Criterion"].tolist()
    targets = [t if pd.notna(t) else None for t in main_criteria_df["Target"]]

st.header("Input or Upload CSV for Scores")
uploaded_file = st.file_uploader("Upload a CSV file with alternatives as rows and criteria as columns", type="csv")

if uploaded_file:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        uploaded_df.index = uploaded_df.iloc[:, 0]
        uploaded_df.drop(uploaded_df.columns[0], axis=1, inplace=True)

        if list(uploaded_df.columns) != criteria_names:
            st.warning(f"Uploaded CSV columns do not match expected criteria: {criteria_names}")
            scores_df = pd.DataFrame(0.0, index=alt_names, columns=criteria_names)
        else:
            scores_df = uploaded_df
            alt_names = uploaded_df.index.tolist()
            st.success("CSV loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        scores_df = pd.DataFrame(0.0, index=alt_names, columns=criteria_names)
else:
    scores_df = pd.DataFrame(0.0, index=alt_names, columns=criteria_names)

edited_scores_df = st.data_editor(scores_df, key="scores")

# Compute WP
if st.button("Compute WP"):
    try:
        matrix = edited_scores_df.values.astype(float)
        weights = np.array(effective_weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()  # Normalize weights
        scores = wp(matrix, weights, types, targets)

        results_df = pd.DataFrame({
            "Alternative": alt_names,
            "Score": scores
        }).sort_values(by="Score", ascending=False)
        results_df["Rank"] = range(1, len(results_df) + 1)

        st.success("WP calculation completed!")
        st.dataframe(results_df[["Rank", "Alternative", "Score"]])

        # Bar chart
        st.subheader("Scores Visualization")
        st.bar_chart(results_df.set_index("Alternative")["Score"])

        # Best alternative
        best_alt = results_df.iloc[0]["Alternative"]
        st.markdown(f"### ✅ The best alternative is **{best_alt}** with a score of **{results_df.iloc[0]['Score']:.4f}**.")

    except Exception as e:
        st.error(f"Error: {e}")
