import streamlit as st


def logout():
    st.markdown(
        f"""
Hi {st.session_state.user_data.get("first_name")} {st.session_state.user_data.get("last_name")},

You are logged in as {st.session_state.user_data.get("email")}.

Click the button below to log out.
"""
    )
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()
