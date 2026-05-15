import streamlit as st
import uuid

def dynamic_list_input(label: str, session_key: str, placeholder: str) -> list[str]:
    # Use a separate key internally to store the dictionary structure
    internal_key = f"_internal_layout_{session_key}"

    if internal_key not in st.session_state or not st.session_state[internal_key]:
        st.session_state[internal_key] = [
            {"id": str(uuid.uuid4()), "value": ""}
        ]

    def add_item_callback():
        st.session_state[internal_key].append(
            {"id": str(uuid.uuid4()), "value": ""}
        )

    def delete_item_callback(target_id):
        st.session_state[internal_key] = [
            item for item in st.session_state[internal_key] if item["id"] != target_id
        ]

    def update_value_callback(item_id, current_key):
        for item in st.session_state[internal_key]:
            if item["id"] == item_id:
                item["value"] = st.session_state[current_key]
                break

    st.markdown(f"**{label}**")
    
    for item in st.session_state[internal_key]:
        col1, col2 = st.columns([12, 1])
        item_id = item["id"]
        input_key = f"input_{session_key}_{item_id}"

        with col1:
            st.text_input(
                label,
                value=item["value"],
                placeholder=placeholder,
                key=input_key,
                label_visibility="collapsed",
                on_change=update_value_callback,
                args=(item_id, input_key)
            )

        with col2:
            st.button(
                "✕", 
                key=f"del_{session_key}_{item_id}", 
                on_click=delete_item_callback, 
                args=(item_id,)
            )

    st.button(
        f"➕ Add {label}", 
        key=f"add_{session_key}", 
        on_click=add_item_callback
    )

    # Return only the cleaned list of strings to forms.py
    return [
        item["value"].strip()
        for item in st.session_state[internal_key]
        if item["value"].strip()
    ]
