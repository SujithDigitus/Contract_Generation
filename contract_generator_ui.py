import streamlit as st
import os
import json
import tempfile
from PIL import Image
from Contract_Generation import (
    get_text_from_Pdf, 
    rag_pipeline_with_prompt, 
    clean_text_for_llm,
    format_text_to_html_with_llm
)

# Page configuration
st.set_page_config(
    page_title="Contract Generator Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .step-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Ensure required folders exist
work_folder = "temp_contracts"
templates_folder = "Contract_templates"
os.makedirs(work_folder, exist_ok=True)
os.makedirs(templates_folder, exist_ok=True)

# Initialize session state
def initialize_session_state():
    if "contract_template" not in st.session_state:
        st.session_state.contract_template = None
    if "contract_placeholders" not in st.session_state:
        st.session_state.contract_placeholders = None
    if "user_placeholder_values" not in st.session_state:
        st.session_state.user_placeholder_values = {}
    if "generated_contract" not in st.session_state:
        st.session_state.generated_contract = None
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None
    if "processing_step" not in st.session_state:
        st.session_state.processing_step = 1

initialize_session_state()

# Main title
st.markdown('<h1 class="main-header">📄 Contract Generator Pro</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. **Upload** your contract PDF
    2. **Extract** placeholders from the document
    3. **Fill in** the required information
    4. **Generate** your customized contract
    5. **Download** in your preferred format
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Progress Tracker")
    
    # Progress indicator
    progress_steps = [
        "Upload PDF",
        "Extract Placeholders", 
        "Fill Information",
        "Generate Contract"
    ]
    
    for i, step in enumerate(progress_steps, 1):
        if st.session_state.processing_step > i:
            st.markdown(f"✅ {step}")
        elif st.session_state.processing_step == i:
            st.markdown(f"🔄 {step}")
        else:
            st.markdown(f"⏳ {step}")

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="step-header">📁 Step 1: Upload Contract PDF</div>', unsafe_allow_html=True)
    
    uploaded_pdf = st.file_uploader(
        "Choose a contract PDF file",
        type=["pdf"],
        help="Upload a PDF contract to extract placeholders and generate templates"
    )
    
    if uploaded_pdf:
        # Save uploaded file
        file_path = os.path.join(work_folder, uploaded_pdf.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        st.session_state.uploaded_filename = uploaded_pdf.name
        st.session_state.processing_step = max(st.session_state.processing_step, 2)
        
        st.markdown(f'<div class="success-box">✅ Successfully uploaded: <strong>{uploaded_pdf.name}</strong></div>', unsafe_allow_html=True)
        
        # File info
        file_size = len(uploaded_pdf.getbuffer()) / 1024  # Size in KB
        st.markdown(f"**File size:** {file_size:.1f} KB")

with col2:
    if uploaded_pdf:
        st.markdown('<div class="step-header">🔍 Step 2: Extract Placeholders</div>', unsafe_allow_html=True)
        
        col2a, col2b = st.columns(2)
        
        with col2a:
            if st.button("🔍 Extract Placeholders", type="primary", use_container_width=True):
                with st.spinner("🤖 Analyzing contract with AI..."):
                    try:
                        # Extract PDF text
                        pdf_text = get_text_from_Pdf(file_path)
                        
                        if not pdf_text.strip():
                            st.error("⚠️ Could not extract text from PDF. Please ensure the PDF contains readable text.")
                        else:
                            # Clean text for LLM processing
                            cleaned_text = clean_text_for_llm(pdf_text)
                            
                            # Process with LLM
                            result = rag_pipeline_with_prompt(cleaned_text)
                            parsed = json.loads(result)
                            
                            # Store in session state
                            st.session_state.contract_template = parsed["Template"]
                            st.session_state.contract_placeholders = parsed["Placeholders"]
                            st.session_state.user_placeholder_values = {
                                k: "" for k in parsed["Placeholders"].keys()
                            }
                            st.session_state.processing_step = max(st.session_state.processing_step, 3)
                            
                            # Save template to file
                            template_path = os.path.join(templates_folder, f"{uploaded_pdf.name}.json")
                            with open(template_path, 'w', encoding='utf-8') as f:
                                json.dump(parsed, f, indent=2)
                            
                            st.success("✅ Placeholders extracted successfully!")

                    except json.JSONDecodeError as e:
                        st.error(f"❌ Failed to parse AI response: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Failed to process contract: {str(e)}")
        
        with col2b:
            # Load existing template if available
            template_path = os.path.join(templates_folder, f"{uploaded_pdf.name}.json")
            if os.path.exists(template_path) and st.button("📂 Load Existing Template", use_container_width=True):
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        parsed = json.load(f)
                    
                    st.session_state.contract_template = parsed["Template"]
                    st.session_state.contract_placeholders = parsed["Placeholders"]
                    st.session_state.user_placeholder_values = {
                        k: "" for k in parsed["Placeholders"].keys()
                    }
                    st.session_state.processing_step = max(st.session_state.processing_step, 3)
                    
                    st.success("✅ Template loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to load template: {str(e)}")

# Step 3: Fill in placeholders
if st.session_state.contract_placeholders:
    st.markdown('<div class="step-header">📝 Step 3: Fill in Contract Details</div>', unsafe_allow_html=True)
    
    # Display placeholder count
    placeholder_count = len(st.session_state.contract_placeholders)
    st.markdown(f'<div class="info-box">Found <strong>{placeholder_count}</strong> placeholders to fill</div>', unsafe_allow_html=True)
    
    # Create tabs for better organization
    if placeholder_count > 5:
        # Split placeholders into chunks for better UX
        placeholder_items = list(st.session_state.contract_placeholders.items())
        chunk_size = 5
        chunks = [placeholder_items[i:i + chunk_size] for i in range(0, len(placeholder_items), chunk_size)]
        
        tab_names = [f"Fields {i*chunk_size + 1}-{min((i+1)*chunk_size, placeholder_count)}" for i in range(len(chunks))]
        tabs = st.tabs(tab_names)
        
        for tab, chunk in zip(tabs, chunks):
            with tab:
                for key, desc in chunk:
                    # Show original value if available
                    if isinstance(desc, dict):
                        description = desc.get("description", key)
                        original_value = desc.get("original_value", "")
                        help_text = f"Original value: {original_value}" if original_value else "No original value found"
                    else:
                        description = str(desc)
                        help_text = None
                    
                    st.session_state.user_placeholder_values[key] = st.text_input(
                        f"**{description}**",
                        value=st.session_state.user_placeholder_values.get(key, ""),
                        key=f"input_{key}",
                        help=help_text
                    )
    else:
        # Show all placeholders in a single view
        for key, desc in st.session_state.contract_placeholders.items():
            # Show original value if available
            if isinstance(desc, dict):
                description = desc.get("description", key)
                original_value = desc.get("original_value", "")
                help_text = f"Original value: {original_value}" if original_value else "No original value found"
            else:
                description = str(desc)
                help_text = None
            
            st.session_state.user_placeholder_values[key] = st.text_input(
                f"**{description}**",
                value=st.session_state.user_placeholder_values.get(key, ""),
                key=f"input_{key}",
                help=help_text
            )

# Step 4: Generate contracts
if st.session_state.contract_placeholders:
    st.markdown('<div class="step-header">🎯 Step 4: Generate Your Contract</div>', unsafe_allow_html=True)
    
    # Generation options
    col4a, col4b = st.columns(2)
    
    with col4a:
        if st.button("📄 Generate Plain Text Contract", type="primary", use_container_width=True):
            with st.spinner("📝 Generating contract..."):
                try:
                    # Fill placeholders in the template
                    final_output = st.session_state.contract_template
                    
                    # Sort keys by length (longest first) to avoid partial replacements
                    sorted_keys = sorted(st.session_state.contract_placeholders.keys(), key=len, reverse=True)
                    
                    for key in sorted_keys:
                        user_value = st.session_state.user_placeholder_values.get(key, "")
                        if user_value:
                            replacement_value = user_value
                        else:
                            # Use original value if available
                            placeholder_data = st.session_state.contract_placeholders[key]
                            if isinstance(placeholder_data, dict):
                                replacement_value = placeholder_data.get("original_value", key)
                            else:
                                replacement_value = key
                        
                        final_output = final_output.replace(key, replacement_value)
                    
                    st.session_state.generated_contract = final_output
                    st.session_state.processing_step = 4
                    st.success("✅ Contract generated successfully!")

                except Exception as e:
                    st.error(f"❌ Contract generation failed: {str(e)}")
    
    with col4b:
        # Style options for HTML generation
        style_options = {
            "Professional (Default)": "Use generic professional styling with clean fonts and proper spacing",
            "Legal Document": "Format as a formal legal document with traditional styling, serif fonts, and proper legal formatting",
            "Modern Business": "Use modern business styling with blue accents, sans-serif fonts, and clean layout",
            "Minimalist": "Apply minimalist styling with lots of white space and simple typography",
            "Corporate": "Use corporate styling with professional colors and structured layout"
        }
        
        selected_style = st.selectbox("Choose HTML Style", list(style_options.keys()))
        
        if st.button("🎨 Generate Styled HTML Contract", type="secondary", use_container_width=True):
            with st.spinner("🎨 Generating styled HTML contract..."):
                try:
                    # Fill placeholders in the template
                    final_output = st.session_state.contract_template
                    
                    # Sort keys by length (longest first) to avoid partial replacements
                    sorted_keys = sorted(st.session_state.contract_placeholders.keys(), key=len, reverse=True)
                    
                    for key in sorted_keys:
                        user_value = st.session_state.user_placeholder_values.get(key, "")
                        if user_value:
                            replacement_value = user_value
                        else:
                            # Use original value if available
                            placeholder_data = st.session_state.contract_placeholders[key]
                            if isinstance(placeholder_data, dict):
                                replacement_value = placeholder_data.get("original_value", key)
                            else:
                                replacement_value = key
                        
                        final_output = final_output.replace(key, replacement_value)
                    
                    # Generate HTML with selected styling
                    html_output = format_text_to_html_with_llm(
                        plain_text_content=final_output,
                        style_instructions=style_options[selected_style]
                    )
                    
                    st.session_state.generated_html_contract = html_output
                    st.success("✅ Styled HTML contract generated successfully!")

                except Exception as e:
                    st.error(f"❌ HTML contract generation failed: {str(e)}")

# Display generated contracts
if hasattr(st.session_state, 'generated_contract') and st.session_state.generated_contract:
    st.markdown("---")
    st.markdown("## 📋 Generated Plain Text Contract")
    
    # Contract preview
    st.text_area(
        "Contract Preview",
        st.session_state.generated_contract,
        height=400,
        key="contract_preview"
    )
    
    # Download options
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        st.download_button(
            "💾 Download as TXT",
            data=st.session_state.generated_contract,
            file_name=f"Contract_{st.session_state.uploaded_filename.replace('.pdf', '')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col_dl2:
        # Convert to basic HTML for download
        basic_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Generated Contract</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
        pre {{ white-space: pre-wrap; }}
    </style>
</head>
<body>
    <pre>{st.session_state.generated_contract}</pre>
</body>
</html>"""
        
        st.download_button(
            "🌐 Download as HTML",
            data=basic_html,
            file_name=f"Contract_{st.session_state.uploaded_filename.replace('.pdf', '')}.html",
            mime="text/html",
            use_container_width=True
        )
    
    with col_dl3:
        # Save as JSON with metadata
        contract_json = {
            "original_file": st.session_state.uploaded_filename,
            "generated_at": str(st.datetime.now()),
            "placeholders_filled": st.session_state.user_placeholder_values,
            "contract_content": st.session_state.generated_contract
        }
        
        st.download_button(
            "📊 Download as JSON",
            data=json.dumps(contract_json, indent=2),
            file_name=f"Contract_{st.session_state.uploaded_filename.replace('.pdf', '')}.json",
            mime="application/json",
            use_container_width=True
        )

# Display styled HTML contract
if hasattr(st.session_state, 'generated_html_contract') and st.session_state.generated_html_contract:
    st.markdown("---")
    st.markdown("## 🎨 Generated Styled HTML Contract")
    
    # HTML preview
    st.components.v1.html(st.session_state.generated_html_contract, height=600, scrolling=True)
    
    # Download HTML
    st.download_button(
        "🎨 Download Styled HTML Contract",
        data=st.session_state.generated_html_contract,
        file_name=f"Styled_Contract_{st.session_state.uploaded_filename.replace('.pdf', '')}.html",
        mime="text/html",
        use_container_width=True
    )

# Help section
if not uploaded_pdf:
    st.markdown("---")
    st.markdown("## 🤔 How it works")
    
    col_help1, col_help2, col_help3 = st.columns(3)
    
    with col_help1:
        st.markdown("""
        ### 1. 📄 Upload
        Upload your contract PDF file. The system will extract the text content for analysis.
        """)
    
    with col_help2:
        st.markdown("""
        ### 2. 🤖 AI Analysis
        Our AI analyzes the contract and identifies placeholders - parts that need to be customized.
        """)
    
    with col_help3:
        st.markdown("""
        ### 3. ✨ Generate
        Fill in your details and generate a customized contract in multiple formats.
        """)

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; font-size: 0.8rem;">Built with ❤️ using Streamlit and Google Gemini AI</div>',
    unsafe_allow_html=True
)