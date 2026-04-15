import streamlit as st

from src.database import (
    add_email_to_whitelist,
    get_users,
    get_whitelisted_emails,
    remove_email_from_whitelist,
    remove_user,
)


def admin_panel():
    """
    Display admin panel for managing users and whitelist.
    Only accessible to admin users.
    """
    if not st.session_state.user_data or not st.session_state.user_data["is_admin"]:
        st.error("Please login as admin to access admin panel")
        return

    st.header("👑 Admin Panel")

    # Tabs for different admin functions
    tab1, tab2 = st.tabs(
        [
            "Whitelist Management",
            "User Management",
        ]
    )

    with tab1:
        st.subheader("Email Whitelist")
        with st.spinner("Loading whitelist..."):
            whitelist_df = get_whitelisted_emails()
        if not whitelist_df.empty:
            st.dataframe(whitelist_df, width="stretch")

            # Remove from whitelist
            email_to_remove = st.text_input("Email to remove from whitelist:")
            if st.button("Remove from Whitelist"):
                if email_to_remove:
                    success, message = remove_email_from_whitelist(email_to_remove)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please enter an email address")
        else:
            st.info("No whitelisted emails found")

        st.subheader("Add Email to Whitelist")
        with st.form("add_to_whitelist"):
            new_email = st.text_input("Email address:")
            created_by = st.text_input(
                "Added by:", value=st.session_state.user_data.get("email", "")
            )
            submit_button = st.form_submit_button("Add to Whitelist")

            if submit_button:
                if new_email:
                    success, message = add_email_to_whitelist(
                        email=new_email,
                        created_by=created_by,
                    )
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please enter an email address")

    with tab2:
        st.subheader("User Management")
        with st.spinner("Loading users..."):
            users_df = get_users()
        if not users_df.empty:
            st.dataframe(users_df, width="stretch")

            # Remove user
            email_to_remove = st.text_input("Email to remove:")
            if st.button("Remove User"):
                if email_to_remove:
                    success, message = remove_user(email_to_remove)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please enter an email address")
        else:
            st.info("No users found")
