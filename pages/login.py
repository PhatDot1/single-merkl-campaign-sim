import streamlit as st

from src.database import login_user, register_user


def login():
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Log in")

        if submit_button:
            with st.spinner("Logging in..."):
                if email and password:
                    success, message, user_data = login_user(email, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_data = user_data
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please enter both email and password")

    with st.expander("Register"):
        with st.form("register_form"):
            first_name = st.text_input("First Name", key="register_first_name")
            last_name = st.text_input("Last Name", key="register_last_name")
            email = st.text_input("Email", key="register_email")
            password = st.text_input(
                "Password", type="password", key="register_password"
            )
            confirm_password = st.text_input(
                "Confirm Password", type="password", key="confirm_password"
            )
            submit_button = st.form_submit_button("Register")
            if submit_button:
                with st.spinner("Registering..."):
                    if not email or not password or not first_name or not last_name:
                        st.error("Please enter all fields")
                        return
                    if password != confirm_password:
                        st.error("Passwords do not match")
                    if len(password) < 8:
                        st.error("Password must be at least 8 characters long")

                    success, message = register_user(
                        email, password, first_name, last_name
                    )
                    if success:
                        st.success("User registered successfully! Please log in.")
                    else:
                        st.error(message)
