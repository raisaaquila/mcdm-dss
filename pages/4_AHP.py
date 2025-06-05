import streamlit as st
import numpy as np
import pandas as pd

def calculate_ahp(matrix):
    matrix = np.array(matrix, dtype=float)
    normalized = matrix / matrix.sum(axis=0)
    weights = normalized.mean(axis=1)

    lambdamax = np.sum((matrix @ weights) / weights) / len(weights)
    conin = (lambdamax - len(weights)) / (len(weights) - 1)

    ridict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
              6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    randin = ridict.get(len(weights), 1.49)
    conrat = conin / randin if randin != 0 else 0

    return weights.tolist(), conrat

def ahp(criteria_matrix, alternative_matrices):
    criteria_weights, criteria_conrat = calculate_ahp(criteria_matrix)

    alt_weights_all = []
    alt_conrats = []

    for alt_matrix in alternative_matrices:
        weights, conrat = calculate_ahp(alt_matrix)
        alt_weights_all.append(weights)
        alt_conrats.append(conrat)

    alt_weights_all = np.array(alt_weights_all)
    final_scores = np.dot(criteria_weights, alt_weights_all)

    return final_scores.tolist(), criteria_weights, criteria_conrat, alt_conrats


st.set_page_config(page_title="AHP Decision Support", page_icon="📊")
st.title('Analytic Hierarchy Process (AHP)')

altamt = st.number_input("Number of Alternatives", min_value=2, max_value=20, value=3)
criteriano = st.number_input("Number of Criteria", min_value=2, max_value=10, value=3)

# Alternatives
st.header("Alternatives")
alt_names = st.text_area("Alternative Names (one per line)",
    "\n".join([f"Alternative {i+1}" for i in range(altamt)])
).splitlines()
alt_names = [name.strip() or f"Alternative {i+1}" for i, name in enumerate(alt_names)]
if len(alt_names) != altamt:
    st.warning(f"Please provide exactly {altamt} alternative names.")

criteria_text = st.text_area(
    "Enter Criteria Names (one per line):",
    value="\n".join([f"Crit {i+1}" for i in range(criteriano)]),
)
criteria = [line.strip() for line in criteria_text.splitlines() if line.strip()]
if len(criteria) != criteriano:
    st.warning(f"Please enter exactly {criteriano} criteria names.")

st.markdown("---")

# === Criteria matrix CSV upload ===
criteria_csv_file = st.file_uploader("Upload Criteria Pairwise Comparison CSV (optional)", type=["csv"])

if criteria_csv_file is not None:
    try:
        criteria_matrix = pd.read_csv(criteria_csv_file, header=None).values
        if criteria_matrix.shape != (criteriano, criteriano):
            st.error(f"CSV matrix shape {criteria_matrix.shape} does not match criteria count {criteriano}.")
            criteria_matrix = None
    except Exception as e:
        st.error(f"Failed to read criteria CSV: {e}")
        criteria_matrix = None
else:
    criteria_matrix = None

# === Input criteria matrix manually if CSV not uploaded ===
if criteria_matrix is None:
    with st.expander("🔷 Criteria Comparison Matrix", expanded=True):
        criteria_matrix = []
        for i in range(criteriano):
            cols = st.columns(criteriano)
            row = []
            for j in range(criteriano):
                if i == j:
                    val = 1.0
                    cols[j].number_input(f"{criteria[i]} vs {criteria[j]}", value=1.0, disabled=True, key=f"crit_{i}_{j}")
                elif i < j:
                    val = cols[j].number_input(
                        f"{criteria[i]} vs {criteria[j]}",
                        min_value=0.01, max_value=9.0, step=1.0, format="%.2f",
                        key=f"crit_{i}_{j}"
                    )
                else:
                    val = 1 / criteria_matrix[j][i]
                    cols[j].number_input(f"{criteria[i]} vs {criteria[j]}", value=round(val, 3), disabled=True, key=f"crit_{i}_{j}")
                row.append(val)
            criteria_matrix.append(row)
    criteria_matrix = np.array(criteria_matrix, dtype=float)

# === Alternative matrices CSV upload per criterion ===
alternative_matrices = []
for c_idx, crit_name in enumerate(criteria):
    st.markdown(f"---")
    st.subheader(f"Alternatives Comparison for Criterion: {crit_name}")

    alt_csv_file = st.file_uploader(f"Upload Alternatives Pairwise Comparison CSV for '{crit_name}' (optional)", type=["csv"], key=f"alt_csv_{c_idx}")

    if alt_csv_file is not None:
        try:
            alt_matrix = pd.read_csv(alt_csv_file, header=None).values
            if alt_matrix.shape != (altamt, altamt):
                st.error(f"CSV matrix shape {alt_matrix.shape} does not match alternatives count {altamt}.")
                alt_matrix = None
        except Exception as e:
            st.error(f"Failed to read alternatives CSV for '{crit_name}': {e}")
            alt_matrix = None
    else:
        alt_matrix = None

    if alt_matrix is None:
        with st.expander(f"Manual Input: Alternatives Comparison Matrix for '{crit_name}'", expanded=True):
            matrix = []
            for i in range(altamt):
                cols = st.columns(altamt)
                row = []
                for j in range(altamt):
                    if i == j:
                        val = 1.0
                        cols[j].number_input(
                            f"{alt_names[i]} vs {alt_names[j]} ({crit_name})",
                            value=1.0, disabled=True,
                            key=f"alt_{c_idx}_{i}_{j}"
                        )
                    elif i < j:
                        val = cols[j].number_input(
                            f"{alt_names[i]} vs {alt_names[j]} ({crit_name})",
                            min_value=0.01, max_value=9.0, step=1.0, format="%.2f",
                            key=f"alt_{c_idx}_{i}_{j}"
                        )
                    else:
                        val = 1 / matrix[j][i]
                        cols[j].number_input(
                            f"{alt_names[i]} vs {alt_names[j]} ({crit_name})",
                            value=round(val, 3), disabled=True,
                            key=f"alt_{c_idx}_{i}_{j}"
                        )
                    row.append(val)
                matrix.append(row)
            alt_matrix = np.array(matrix, dtype=float)

    alternative_matrices.append(alt_matrix)

# === Final calculation and output ===
if st.button("Compute Final Scores"):
    try:
        final_scores, crit_weights, crit_conrat, alt_conrats = ahp(criteria_matrix, alternative_matrices)

        st.success("✅ AHP calculation completed!")

        st.subheader("Final Rankings")
        results = pd.DataFrame({
            "Alternative": alt_names,
            "Score": final_scores
        }).sort_values(by="Score", ascending=False).reset_index(drop=True)
        results.index += 1
        st.dataframe(results)

        st.subheader("Criteria Weights")
        weights_df = pd.DataFrame({
            "Criterion": criteria,
            "Weight": crit_weights,
            "CR": [crit_conrat] * len(criteria)
        })
        st.dataframe(weights_df)

        st.info(f"Overall Criteria Consistency Ratio (CR): **{crit_conrat:.4f}**")
        if crit_conrat > 0.1:
            st.warning("⚠️ Criteria CR > 0.1 — Consider adjusting pairwise comparisons for better consistency.")

        st.subheader("Per-Criterion Consistency Ratios")
        for i, conrat in enumerate(alt_conrats):
            label = f"{criteria[i]} → CR: {conrat:.4f}"
            if conrat > 0.1:
                st.warning(f"⚠️ {label} (Consider revising this comparison matrix)")
            else:
                st.success(f"✅ {label}")
        st.subheader("Scores Visualization")
        st.bar_chart(results.set_index("Alternative")["Score"])

        best = results.iloc[0]
        st.markdown(f"### ✅ Best Alternative: **{best['Alternative']}** with score **{best['Score']:.4f}**")
    except Exception as e:
        st.error(f"❌ Error during AHP computation: {e}")