import streamlit as st
import os
import json
from PIL import Image
from Contract_generation import get_text_from_Pdf, rag_pipeline_with_prompt, Generation_of_Contract

# Ensure required folders exist
work_folder = "temp_contracts"
os.makedirs(work_folder, exist_ok=True)

st.set_page_config(page_title="Contract Generator", page_icon="📄")
st.title("📄 Contract Generator Assistant")

# Initialize session state
if "contract_template" not in st.session_state:
    st.session_state.contract_template = None
if "contract_placeholders" not in st.session_state:
    st.session_state.contract_placeholders = None
if "user_placeholder_values" not in st.session_state:
    st.session_state.user_placeholder_values = {}

# Sidebar UI
with st.sidebar:
    st.header("Step 1: Upload PDF")
    uploaded_pdf = st.file_uploader("Upload a contract PDF", type=["pdf"])

    if uploaded_pdf:
        file_path = os.path.join(work_folder, uploaded_pdf.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        st.success(f"Uploaded: {uploaded_pdf.name}")

        if st.button("🔍 Extract Placeholders"):
            with st.spinner("Analyzing contract..."):
                try:
                    pdf_text = get_text_from_Pdf(file_path)
                    result = rag_pipeline_with_prompt(pdf_text)
                    parsed = json.loads(result)

                    st.session_state.contract_template = parsed["Template"]
                    st.session_state.contract_placeholders = parsed["Placeholders"]
                    st.session_state.user_placeholder_values = {
                        k: "" for k in parsed["Placeholders"].keys()
                    }

                    st.success("✅ Placeholders extracted successfully!")

                except Exception as e:
                    st.error(f"Failed to process contract: {str(e)}")

# Main UI
if st.session_state.contract_placeholders:
    st.header("Step 2: Fill in Details")
    for key, desc in st.session_state.contract_placeholders.items():
        st.session_state.user_placeholder_values[key] = st.text_input(
            f"{desc} ({key})", value=st.session_state.user_placeholder_values.get(key, "")
        )

    if st.button("📄 Generate Final Contract"):
        with st.spinner("Generating final contract..."):
            try:
                final_output = Generation_of_Contract(
                    st.session_state.contract_template,
                    json.dumps(st.session_state.user_placeholder_values)
                )
                st.success("✅ Contract generated!")
                st.subheader("Generated Contract")
                st.text_area("Contract Output", final_output, height=400)
                st.download_button("💾 Download Contract", data=final_output, file_name="Final_Contract.txt")

            except Exception as e:
                st.error(f"Contract generation failed: {str(e)}")

elif uploaded_pdf is None:
    st.info("Please upload a contract PDF to begin.")
