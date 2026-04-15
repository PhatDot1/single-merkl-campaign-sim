import streamlit as st

from pages.login import login
from pages.logout import logout
from pages.admin_panel import admin_panel
from sim.dashboard.app5 import main as app5

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

home_page = st.Page(
    app5,
    title="Homepage",
    icon=":material/home:",
    default=True,
)


admin_panel_page = st.Page(
    admin_panel, title="Admin Panel", icon=":material/security:", default=False
)

if st.session_state.logged_in:
    account_pages = [logout_page]
    if st.session_state.user_data.get("is_admin"):
        account_pages.append(admin_panel_page)

    pg = st.navigation(
        {
            "Account": account_pages,
            "Homepage": [
                # sentora_summary_page,
                home_page,
            ],
        }
    )
else:
    pg = st.navigation([login_page])

pg.run()
