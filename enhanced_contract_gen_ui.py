import streamlit as st
import os
import json
from PIL import Image
from contract_generation_enhanced import (
    get_text_from_Pdf, 
    rag_pipeline_with_prompt, 
    generate_contract_from_template,
    interactive_contract_modifier,
    batch_contract_modifications,
    get_contract_sections_summary,
    format_text_to_html_with_llm
)

# Ensure required folders exist
work_folder = "temp_contracts"
os.makedirs(work_folder, exist_ok=True)

st.set_page_config(page_title="Enhanced Contract Generator", page_icon="📄", layout="wide")
st.title("📄 Enhanced Contract Generator Assistant")

# Initialize session state
if "contract_template" not in st.session_state:
    st.session_state.contract_template = None
if "contract_placeholders" not in st.session_state:
    st.session_state.contract_placeholders = None
if "user_placeholder_values" not in st.session_state:
    st.session_state.user_placeholder_values = {}
if "generated_contract" not in st.session_state:
    st.session_state.generated_contract = None
if "final_contract" not in st.session_state:
    st.session_state.final_contract = None
if "contract_sections" not in st.session_state:
    st.session_state.contract_sections = None
if "modification_history" not in st.session_state:
    st.session_state.modification_history = []

# Create two columns for better layout
col1, col2 = st.columns([1, 2])

# Sidebar UI
with st.sidebar:
    st.header("🔧 Contract Generation Pipeline")
    
    # Step 1: Upload PDF
    st.subheader("Step 1: Upload PDF")
    uploaded_pdf = st.file_uploader("Upload a contract PDF", type=["pdf"])

    if uploaded_pdf:
        file_path = os.path.join(work_folder, uploaded_pdf.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        st.success(f"✅ Uploaded: {uploaded_pdf.name}")

        # Step 2: Extract Template
        if st.button("🔍 Extract Template & Placeholders"):
            with st.spinner("Analyzing contract structure..."):
                try:
                    pdf_text = get_text_from_Pdf(file_path)
                    result = rag_pipeline_with_prompt(pdf_text)
                    parsed = json.loads(result)

                    st.session_state.contract_template = parsed["Template"]
                    st.session_state.contract_placeholders = parsed["Placeholders"]
                    st.session_state.user_placeholder_values = {
                        k: v.get("original_value", "") if isinstance(v, dict) else str(v)
                        for k, v in parsed["Placeholders"].items()
                    }

                    st.success("✅ Template extracted successfully!")
                    st.info(f"Found {len(st.session_state.contract_placeholders)} placeholders")

                except Exception as e:
                    st.error(f"Failed to process contract: {str(e)}")
    
    # Reset button
    if st.button("🔄 Reset All", type="secondary"):
        for key in ["contract_template", "contract_placeholders", "user_placeholder_values", 
                   "generated_contract", "final_contract", "contract_sections", "modification_history"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Main content area
with col1:
    st.header("📝 Contract Configuration")
    
    # Step 3: Fill in Placeholders
    if st.session_state.contract_placeholders:
        st.subheader("Step 2: Fill in Details")
        
        # Create a form for better organization
        with st.form("placeholder_form"):
            for key, placeholder_info in st.session_state.contract_placeholders.items():
                if isinstance(placeholder_info, dict):
                    description = placeholder_info.get("description", key)
                    original_value = placeholder_info.get("original_value", "")
                else:
                    description = str(placeholder_info)
                    original_value = ""
                
                # Use text_area for longer content, text_input for shorter
                if len(original_value) > 50:
                    st.session_state.user_placeholder_values[key] = st.text_area(
                        f"**{description}**",
                        value=st.session_state.user_placeholder_values.get(key, original_value),
                        height=100,
                        help=f"Placeholder: {key}"
                    )
                else:
                    st.session_state.user_placeholder_values[key] = st.text_input(
                        f"**{description}**",
                        value=st.session_state.user_placeholder_values.get(key, original_value),
                        help=f"Placeholder: {key}"
                    )
            
            # Generate contract button
            if st.form_submit_button("📄 Generate Contract", type="primary"):
                with st.spinner("Generating contract..."):
                    try:
                        # Prepare placeholders dict for generation
                        placeholders_for_generation = {
                            k: {"value": v} for k, v in st.session_state.user_placeholder_values.items()
                        }
                        
                        st.session_state.generated_contract = generate_contract_from_template(
                            st.session_state.contract_template,
                            placeholders_for_generation
                        )
                        st.session_state.final_contract = st.session_state.generated_contract
                        
                        # Get contract sections for modification reference
                        st.session_state.contract_sections = get_contract_sections_summary(
                            st.session_state.generated_contract
                        )
                        
                        st.success("✅ Contract generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Contract generation failed: {str(e)}")

    elif uploaded_pdf is None:
        st.info("👆 Please upload a contract PDF to begin.")
    else:
        st.info("Click 'Extract Template & Placeholders' to analyze the uploaded PDF.")

# Contract display and modification area
with col2:
    st.header("📋 Contract Review & Modification")
    
    if st.session_state.generated_contract:
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Current Contract", "🔧 Modify Contract", "📊 Contract Sections", "📁 Export Options"])
        
        with tab1:
            st.subheader("Current Contract")
            st.text_area(
                "Contract Content",
                value=st.session_state.final_contract,
                height=600,
                disabled=True
            )
            
            # Quick stats
            contract_stats = {
                "Characters": len(st.session_state.final_contract),
                "Words": len(st.session_state.final_contract.split()),
                "Lines": len(st.session_state.final_contract.split('\n')),
                "Modifications Applied": len(st.session_state.modification_history)
            }
            
            cols = st.columns(4)
            for i, (stat, value) in enumerate(contract_stats.items()):
                cols[i].metric(stat, value)
        
        with tab2:
            st.subheader("Interactive Contract Modifications")
            
            # Show modification history
            if st.session_state.modification_history:
                st.write("**Modification History:**")
                for i, mod in enumerate(st.session_state.modification_history, 1):
                    st.write(f"{i}. {mod}")
                st.divider()
            
            # Modification input
            modification_request = st.text_area(
                "**Describe the modification you want to make:**",
                placeholder="Example: Add a confidentiality section, Remove the penalty clause, Change payment terms to quarterly payments, etc.",
                height=100
            )
            
            col_modify, col_batch = st.columns(2)
            
            with col_modify:
                if st.button("🔄 Apply Modification", type="primary", disabled=not modification_request.strip()):
                    with st.spinner("Applying modification..."):
                        try:
                            st.session_state.final_contract = interactive_contract_modifier(
                                st.session_state.final_contract,
                                modification_request
                            )
                            st.session_state.modification_history.append(modification_request)
                            st.success("✅ Modification applied successfully!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Modification failed: {str(e)}")
            
            with col_batch:
                if st.button("↩️ Revert to Original", type="secondary"):
                    st.session_state.final_contract = st.session_state.generated_contract
                    st.session_state.modification_history = []
                    st.success("✅ Reverted to original generated contract")
                    st.rerun()
            
            # Batch modification section
            st.divider()
            st.subheader("Batch Modifications")
            st.write("Apply multiple modifications at once:")
            
            # Allow users to add multiple modification requests
            if "batch_modifications" not in st.session_state:
                st.session_state.batch_modifications = [""]
            
            for i, mod in enumerate(st.session_state.batch_modifications):
                col_input, col_remove = st.columns([4, 1])
                with col_input:
                    st.session_state.batch_modifications[i] = st.text_input(
                        f"Modification {i+1}",
                        value=mod,
                        key=f"batch_mod_{i}"
                    )
                with col_remove:
                    if len(st.session_state.batch_modifications) > 1:
                        if st.button("❌", key=f"remove_{i}"):
                            st.session_state.batch_modifications.pop(i)
                            st.rerun()
            
            col_add, col_apply_batch = st.columns(2)
            with col_add:
                if st.button("➕ Add Another Modification"):
                    st.session_state.batch_modifications.append("")
                    st.rerun()
            
            with col_apply_batch:
                valid_mods = [mod for mod in st.session_state.batch_modifications if mod.strip()]
                if st.button("🔄 Apply All Modifications", disabled=not valid_mods):
                    with st.spinner("Applying batch modifications..."):
                        try:
                            st.session_state.final_contract = batch_contract_modifications(
                                st.session_state.final_contract,
                                valid_mods
                            )
                            st.session_state.modification_history.extend(valid_mods)
                            st.success(f"✅ Applied {len(valid_mods)} modifications successfully!")
                            st.session_state.batch_modifications = [""]
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Batch modification failed: {str(e)}")
        
        with tab3:
            st.subheader("Contract Structure Analysis")
            if st.session_state.contract_sections:
                st.write(st.session_state.contract_sections)
            else:
                if st.button("🔍 Analyze Contract Sections"):
                    with st.spinner("Analyzing contract structure..."):
                        try:
                            st.session_state.contract_sections = get_contract_sections_summary(
                                st.session_state.final_contract
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
        
        with tab4:
            st.subheader("Export Options")
            
            # Text download
            st.download_button(
                "💾 Download as Text",
                data=st.session_state.final_contract,
                file_name="contract.txt",
                mime="text/plain"
            )
            
            # HTML formatting and download
            st.write("**HTML Export Options:**")
            html_style = st.selectbox(
                "Choose HTML styling:",
                [
                    "Use professional legal document styling with clear section headers and proper spacing",
                    "Use modern corporate styling with elegant fonts and layouts",
                    "Use minimal clean styling with good readability",
                    "Use generic professional styling"
                ]
            )
            
            if st.button("🌐 Generate HTML Version"):
                with st.spinner("Formatting to HTML..."):
                    try:
                        html_content = format_text_to_html_with_llm(
                            st.session_state.final_contract,
                            html_style
                        )
                        
                        st.download_button(
                            "💾 Download HTML",
                            data=html_content,
                            file_name="contract.html",
                            mime="text/html"
                        )
                        
                        # Show preview
                        with st.expander("📱 HTML Preview"):
                            st.components.v1.html(html_content, height=400, scrolling=True)
                            
                    except Exception as e:
                        st.error(f"HTML generation failed: {str(e)}")
            
            # JSON export of the complete session
            if st.button("📊 Export Session Data (JSON)"):
                session_data = {
                    "template": st.session_state.contract_template,
                    "placeholders": st.session_state.contract_placeholders,
                    "user_values": st.session_state.user_placeholder_values,
                    "generated_contract": st.session_state.generated_contract,
                    "final_contract": st.session_state.final_contract,
                    "modification_history": st.session_state.modification_history,
                    "contract_sections": st.session_state.contract_sections
                }
                
                st.download_button(
                    "💾 Download Session JSON",
                    data=json.dumps(session_data, indent=2),
                    file_name="contract_session.json",
                    mime="application/json"
                )

    elif st.session_state.contract_template:
        st.info("👈 Fill in the contract details and click 'Generate Contract' to proceed.")
    else:
        st.info("Upload and process a PDF contract to start generating contracts.")

# Footer
st.divider()
st.markdown("")
# st.markdown("""
# **Enhanced Contract Generator Features:**
# - 🔍 Intelligent template extraction from PDF contracts
# - 📝 Dynamic placeholder identification and filling
# - 🔄 Interactive contract modifications using AI
# - 📊 Contract structure analysis
# - 🌐 HTML formatting and export options
# - 📁 Multiple export formats (Text, HTML, JSON)
# - 📚 Batch modification support
# """)