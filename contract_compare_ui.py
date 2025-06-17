import streamlit as st
import os
import json
import PyPDF2
import tempfile
from io import BytesIO
from dotenv import load_dotenv
import base64

# Import functions from the contract comparison module
from contract_compare import compare_contracts_with_llm, generate_html_report

# --- Configuration ---
# Load environment variables
load_dotenv()

# --- Helper Functions ---

def extract_text_from_pdf_bytes(pdf_bytes):
    """Extracts text from PDF bytes."""
    text = ""
    try:
        pdf_file = BytesIO(pdf_bytes)
        reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() or ""
        if not text.strip():
            st.warning("No text extracted from PDF. The PDF might be image-based or empty.")
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def compare_contracts_with_llm_streamlit(contract_texts, contract_labels, api_key):
    """Wrapper function that adapts the original compare_contracts_with_llm for Streamlit."""
    if not api_key:
        st.error("API key is missing.")
        return None

    # Set the API key in environment for the imported function
    original_api_key = os.environ.get("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    
    try:
        # Show progress messages
        max_len = 30000 
        for i, contract_text in enumerate(contract_texts):
            if len(contract_text) > max_len:
                st.warning(f"Contract {i+1} text truncated to {max_len} characters for LLM.")
        
        # Call the imported function
        result = compare_contracts_with_llm(contract_texts, contract_labels)
        
        # Handle the results with Streamlit-specific messaging
        if result is None:
            st.error("LLM output structure is not a recognized list or single item.")
            return None
        elif not result:
            st.info("LLM returned an empty response. Assuming no differences found.")
            return []
        
        return result
        
    except Exception as e:
        st.error(f"An error occurred during LLM comparison: {e}")
        return None
    finally:
        # Restore original API key
        if original_api_key:
            os.environ["GOOGLE_API_KEY"] = original_api_key
        elif "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]

def generate_html_report_content(comparison_data, contract_labels):
    """Generates HTML content for the comparison report."""
    if not comparison_data:
        return "<div class='alert alert-info'>No comparison data was generated. This could be due to an error in processing, the LLM did not return valid data, or no differences were identified by the LLM.</div>"

    # Generate dynamic table headers based on number of contracts
    contract_columns = ""
    for label in contract_labels:
        contract_columns += f"<th>{label} Detail</th>\n                    "

    html_content = f"""
    <style>
        .comparison-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 2px 15px rgba(0,0,0,0.1); background-color: #fff; }}
        .comparison-table th, .comparison-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; }}
        .comparison-table th {{ background-color: #007bff; color: white; font-weight: bold; }}
        .comparison-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .comparison-table tr:hover {{ background-color: #f1f1f1; }}
        .category {{ font-weight: bold; }}
        .difference {{ color: #d9534f; }}
        .no-difference {{ color: #5cb85c; }}
        .detail-cell {{ white-space: pre-wrap; word-wrap: break-word; }}
    </style>
    
    <h3>Contract Comparison Report - Identified Differences ({len(contract_labels)} Contracts)</h3>
    <table class="comparison-table">
        <thead>
            <tr>
                <th>Differing Aspect / Clause Category</th>
                {contract_columns}<th>Analysis of Difference</th>
            </tr>
        </thead>
        <tbody>
    """
    
    added_rows = 0
    for item in comparison_data:
        category = item.get("clause_category", "N/A")
        analysis = item.get("analysis_of_difference", "N/A")

        # Conditions to skip adding a row
        skip_row = False
        analysis_lower = analysis.lower()

        # Check if all contract details are "not found" or similar
        all_not_found = True
        contract_details = []
        for i, label in enumerate(contract_labels):
            detail_key = f"contract_{chr(65+i).lower()}_detail"
            detail = item.get(detail_key, "N/A")
            contract_details.append(detail)
            
            detail_lower = detail.lower()
            not_found_phrases = ["not specified", "not found", "n/a"]
            if detail_lower not in not_found_phrases:
                all_not_found = False

        if analysis_lower == "not found in any contract.":
            skip_row = True
        elif all_not_found and analysis_lower in ["not specified", "not found", "n/a"]:
            skip_row = True
        
        if skip_row:
            continue 

        analysis_class = "difference"
        if "no significant difference" in analysis_lower or "similar" in analysis_lower:
            analysis_class = "no-difference"

        # Generate contract detail cells
        contract_detail_cells = ""
        for detail in contract_details:
            contract_detail_cells += f'<td class="detail-cell">{detail}</td>\n                    '
        
        html_content += f"""
                <tr>
                    <td class="category">{category}</td>
                    {contract_detail_cells}<td class="{analysis_class} detail-cell">{analysis}</td>
                </tr>
        """
        added_rows += 1

    html_content += """
            </tbody>
        </table>
    """
    
    if added_rows == 0:
        html_content += "<div class='alert alert-info'>No significant differences were identified by the LLM for comparison, or all identified items were filtered out.</div>"
    
    return html_content

def create_download_link(html_content, filename):
    """Creates a download link for the HTML report."""
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">Download HTML Report</a>'
    return href

# --- Streamlit UI ---

def main():
    st.set_page_config(
        page_title="Contract Comparison Tool", 
        page_icon="📋", 
        layout="wide"
    )
    
    st.title("📋 Contract Comparison Tool")
    st.markdown("Upload 2-10 PDF contracts to compare and identify key differences using AI.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google API Key", 
            type="password", 
            help="Enter your Google Gemini API key",
            value=os.getenv("GOOGLE_API_KEY", "")
        )
        
        if not api_key:
            st.warning("Please enter your Google API key to proceed.")
        
        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.markdown("""
        1. Enter your Google Gemini API key
        2. Upload 2-10 PDF contract files
        3. Click 'Compare Contracts'
        4. Review the differences found
        5. Download the HTML report
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload Contract Files")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload 2-10 PDF contract files for comparison"
        )
        
        # Validate number of files
        if uploaded_files:
            num_files = len(uploaded_files)
            if num_files < 2:
                st.error("Please upload at least 2 PDF files for comparison.")
            elif num_files > 10:
                st.error("Maximum 10 PDF files are supported. Please remove some files.")
            else:
                st.success(f"✅ {num_files} files uploaded successfully")
                
                # Display uploaded files
                with st.expander("📋 Uploaded Files", expanded=True):
                    for i, file in enumerate(uploaded_files, 1):
                        st.write(f"**Contract {chr(64+i)}:** {file.name} ({file.size:,} bytes)")
    
    with col2:
        st.header("🔍 Analysis")
        
        if uploaded_files and len(uploaded_files) >= 2 and len(uploaded_files) <= 10 and api_key:
            if st.button("🚀 Compare Contracts", type="primary"):
                with st.spinner("Processing contracts and analyzing differences..."):
                    # Extract text from all PDFs
                    contract_texts = []
                    contract_labels = []
                    failed_extractions = []
                    
                    progress_bar = st.progress(0)
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        contract_text = extract_text_from_pdf_bytes(uploaded_file.read())
                        if contract_text:
                            contract_texts.append(contract_text)
                            contract_labels.append(chr(65 + len(contract_texts) - 1))
                            st.success(f"✅ Extracted text from {uploaded_file.name}")
                        else:
                            failed_extractions.append(f"Contract {chr(65 + i)} ({uploaded_file.name})")
                            st.error(f"❌ Failed to extract text from {uploaded_file.name}")
                    
                    progress_bar.empty()
                    
                    if failed_extractions:
                        st.warning(f"⚠️ Failed to extract text from: {', '.join(failed_extractions)}")
                    
                    if len(contract_texts) >= 2:
                        # Perform comparison
                        with st.spinner("Analyzing contracts with AI..."):
                            comparison_results = compare_contracts_with_llm_streamlit(contract_texts, contract_labels, api_key)
                        
                        # Store results in session state
                        st.session_state.comparison_results = comparison_results
                        st.session_state.contract_labels = contract_labels
                        st.session_state.uploaded_file_names = [f.name for f in uploaded_files if contract_texts]
                        
                        if comparison_results is not None:
                            if comparison_results:
                                st.success(f"✅ Analysis complete! Found {len(comparison_results)} differences.")
                            else:
                                st.info("ℹ️ No material differences found between the contracts.")
                        else:
                            st.error("❌ Analysis failed. Please check your API key and try again.")
                    else:
                        st.error("❌ Need at least 2 valid contracts for comparison.")
        else:
            if not api_key:
                st.info("🔑 Please enter your API key in the sidebar.")
            elif not uploaded_files:
                st.info("📁 Please upload PDF files to begin.")
            elif len(uploaded_files) < 2:
                st.info("📁 Please upload at least 2 PDF files.")
            elif len(uploaded_files) > 10:
                st.info("📁 Please upload no more than 10 PDF files.")
    
    # Display results
    if hasattr(st.session_state, 'comparison_results') and st.session_state.comparison_results is not None:
        st.markdown("---")
        st.header("📊 Comparison Results")
        
        # Generate and display HTML content
        html_content = generate_html_report_content(
            st.session_state.comparison_results, 
            st.session_state.contract_labels
        )
        
        # Display the comparison table
        st.markdown(html_content, unsafe_allow_html=True)
        
        # Download section
        st.markdown("---")
        st.header("💾 Download Report")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Use the imported generate_html_report function to create a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
                # Generate HTML report using the imported function
                generate_html_report(
                    st.session_state.comparison_results, 
                    st.session_state.contract_labels, 
                    tmp_file.name
                )
                
                # Read the generated file
                with open(tmp_file.name, 'r', encoding='utf-8') as f:
                    full_html = f.read()
                
                # Clean up the temporary file
                os.unlink(tmp_file.name)
            
            st.download_button(
                label="📥 Download HTML Report",
                data=full_html,
                file_name="contract_comparison_report.html",
                mime="text/html"
            )
        
        with col2:
            # JSON download
            if st.session_state.comparison_results:
                json_data = json.dumps(st.session_state.comparison_results, indent=2)
                st.download_button(
                    label="📥 Download JSON Data",
                    data=json_data,
                    file_name="contract_comparison_data.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()