import streamlit as st
import folium
import json
import os
import plotly.express as px
import pandas as pd
from datetime import datetime
from streamlit_folium import st_folium
from databricks import sql
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

# Ensure environment variable is set correctly
assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

# --- App Config ---
st.set_page_config(page_title="Medicare Plan Search", layout="wide")

# Cached SQL query
@st.cache_data(ttl=30)
def sqlQuery(query: str) -> pd.DataFrame:
    cfg = Config() # Pull environment variables for auth
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
        credentials_provider=lambda: cfg.authenticate
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()

###########Useful Metrics#######

daset_df = sqlQuery("SELECT * FROM workspace.wellness.landscape_special_needs_plan limit 100")

# Group by organization and compute average rating
top_orgs = (
    daset_df.groupby("organization_name")["overall_star_rating"]
    .mean()
    .reset_index()
    .sort_values(by="overall_star_rating", ascending=False)
    .head(10)
)

# Create pie chart
fig = px.pie(
    top_orgs,
    values="overall_star_rating",
    names="organization_name",
    title="Top 10 Organizations by Average Rating",
    color_discrete_sequence=px.colors.qualitative.Set3
)

# Display in Streamlit
st.plotly_chart(fig)

################################

# SQL query options
options = {
    "Dual-Eligible Plans": "SELECT * FROM workspace.wellness.landscape_special_needs_plan WHERE special_needs_plan_type='Dual-Eligible' LIMIT 100"
}

# --- State ---
chat_history = st.session_state.setdefault("chat_history", [])
selected_index = st.session_state.setdefault("selected_index", len(chat_history) - 1 if chat_history else None)

# --- Sidebar ---
st.sidebar.header("AI Assistant")

# Clear history
if st.sidebar.button("Clear History"):
    st.session_state.chat_history = []
    st.session_state.selected_index = None
    # st.experimental_rerun()

# Select predefined SQL query
user_input = st.sidebar.selectbox("Choose an option:", list(options.keys()))

# Load available states and counties only if 'Dual-Eligible Plans' is selected
selected_state = None

if user_input == "Dual-Eligible Plans":
    # Pull distinct states and counties for dropdowns
    try:
        df_state = sqlQuery("SELECT DISTINCT state FROM workspace.wellness.landscape_special_needs_plan WHERE special_needs_plan_type='Dual-Eligible'")
        state_options = sorted(df_state['state'].dropna().unique())

        selected_state = st.sidebar.selectbox("Filter by State (optional):", [""] + state_options)

    except Exception as e:
        st.sidebar.error(f"Failed to load filters: {str(e)}")

# Run query on Send
if st.sidebar.button("Send"):
    if user_input:
        try:
            # Build filtered query
            base_query = options[user_input]
            where_clauses = []

            if selected_state:
                where_clauses.append(f"state = '{selected_state}'")

            if where_clauses:
                base_query = base_query.replace("WHERE", f"WHERE {' AND '.join(where_clauses)} AND")
                
            df = sqlQuery(base_query)

            plans_json = df.to_dict(orient="records")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            chat_history.append({
                "user_input": user_input,
                "plans": plans_json,
                "timestamp": timestamp
            })
            st.session_state.selected_index = len(chat_history) - 1
        except Exception as e:
            st.sidebar.error(f"Query failed: {str(e)}")

# --- Chat History Display ---
st.sidebar.markdown("### Previous Queries")
if chat_history:
    for i, entry in enumerate(reversed(chat_history)):
        label = f"{len(chat_history) - 1 - i + 1}: {entry['user_input']} ({entry['timestamp']})"
        if st.sidebar.button(label, key=f"history_{i}"):
            st.session_state.selected_index = len(chat_history) - 1 - i
else:
    st.sidebar.info("No previous queries yet.")

# --- Main Content ---
st.title("Medicare Plan Search Results")

if chat_history and st.session_state.selected_index is not None:
    selected_entry = chat_history[st.session_state.selected_index]
    plans = selected_entry["plans"]

    st.info(f"Showing results for: **{selected_entry['user_input']}** (at {selected_entry['timestamp']})")

    for plan in plans:
        st.subheader(plan.get("plan_name", plan.get("name", "Unknown Plan")))
        st.write(f"**Organization:** {plan.get('organization_name', 'N/A')}")
        st.write(f"**Type:** {plan.get('type_of_medicare_health_plan', 'N/A')}")
        st.write(f"**Premium:** ${plan.get('monthly_consolidated_premium_includes_part_c_d', 'N/A')}")
        st.write(f"**Location:** {plan.get('county', 'N/A')} County, {plan.get('state', 'N/A')}")
        st.markdown("---")

     # Extract unique (state, county) pairs
    county_state_pairs = set()
    for plan in plans:
        county = plan.get("county")
        state = plan.get("state")
        if county and state:
            county_state_pairs.add((state, county))

    # Prepare for SQL IN clause
    if county_state_pairs:
        filters = " OR ".join(
            f"(NAME = '{county}')" for state, county in county_state_pairs
        )
        centroid_query = f"""
            WITH filtered AS (
                SELECT NAME AS county, x, y, STATEFP
                FROM workspace.wellness.county_centroids
                WHERE {filters}
            ),
            state_freq AS (
                SELECT STATEFP, COUNT(*) AS count
                FROM filtered
                GROUP BY STATEFP
                ORDER BY count DESC
                LIMIT 1
            )
            SELECT f.*
            FROM filtered f
            INNER JOIN state_freq s ON f.STATEFP = s.STATEFP
        """

        try:
            centroid_df = sqlQuery(centroid_query)
        except Exception as e:
            st.warning(f"Failed to load county centroids: {e}")
            centroid_df = pd.DataFrame()
    else:
        centroid_df = pd.DataFrame()

    # Build map
    m = folium.Map(location=[31.0, -99.0], zoom_start=6)
    marker_locations = []

    for _, row in centroid_df.iterrows():
        lat, lon = row["y"], row["x"]
        marker_locations.append((lat, lon))
        folium.Marker(
            location=(lat, lon),
            tooltip=f"{row['county']} County"
        ).add_to(m)

    # Adjust map to fit markers
    if marker_locations:
        if len(marker_locations) == 1:
            m.location = marker_locations[0]
            m.zoom_start = 10
        else:
            m.fit_bounds(marker_locations)

    st.subheader("Plan County Locations Map")
    st_folium(m, width=700, height=500)
else:
    st.info("Enter a query or select a previous one to view results.")

st.chat_input("Type your message...")

# --- Footer Disclaimer ---
st.markdown("""
---
*Disclaimer: This tool is for informational purposes only and is not affiliated with CMS or Medicare. Plan availability and features vary by location. Please consult an official Medicare resource or licensed agent before enrolling.*
""")
