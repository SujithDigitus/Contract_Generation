import os
import json
import PyPDF2
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import argparse

# --- Configuration ---
# Attempt to load the API key from environment variables
# Canvas will provide the GOOGLE_API_KEY if it's available in the environment
# For local execution, ensure your .env file is set up or the env var is exported.
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    # Fallback for local development if .env is used and GOOGLE_API_KEY is not directly in env
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("Error: GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")
    # In a real application, you might exit here or raise an error.
    # For this example, we'll let it proceed, but Langchain will fail if the key is truly missing.

# --- Helper Functions ---

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or "" # Add empty string if extract_text returns None
        if not text.strip():
            print(f"Warning: No text extracted from {pdf_path}. The PDF might be image-based or empty.")
        return text
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None

def generate_html_report(comparison_data, contract_labels, output_path="contract_comparison_report.html"):
    """Generates an HTML report from the comparison data, filtering out 'Not found' entries."""
    if not comparison_data:
        html_content = "<html><head><title>Contract Comparison Report</title></head>"
        html_content += "<body><h1>Contract Comparison Report</h1>"
        html_content += "<p>No comparison data was generated. This could be due to an error in processing, the LLM did not return valid data, or no differences were identified by the LLM.</p>"
        html_content += "</body></html>"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Generated basic HTML report (no data or no differences found) at {output_path}")
        return

    # Generate dynamic table headers based on number of contracts
    contract_columns = ""
    for label in contract_labels:
        contract_columns += f"<th>{label} Detail</th>\n                    "

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Contract Comparison Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f4f4f9; color: #333; }}
            h1 {{ color: #333; text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 2px 15px rgba(0,0,0,0.1); background-color: #fff; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; }}
            th {{ background-color: #007bff; color: white; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            tr:hover {{ background-color: #f1f1f1; }}
            .category {{ font-weight: bold; }}
            .difference {{ color: #d9534f; }} /* Highlight differences */
            .no-difference {{ color: #5cb85c; }} /* Highlight similarities - less likely with new prompt */
            .detail-cell {{ white-space: pre-wrap; word-wrap: break-word; }} /* Preserve formatting and wrap long text */
        </style>
    </head>
    <body>
        <h1>Contract Comparison Report - Identified Differences ({len(contract_labels)} Contracts)</h1>
        <table>
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

        # Conditions to skip adding a row (though less likely with the new prompt focusing on differences)
        skip_row = False
        analysis_lower = analysis.lower()

        # Check if all contract details are "not found" or similar
        all_not_found = True
        contract_details = []
        for i, label in enumerate(contract_labels):
            detail_key = f"contract_{chr(65+i).lower()}_detail"  # contract_a_detail, contract_b_detail, etc.
            detail = item.get(detail_key, "N/A")
            contract_details.append(detail)
            
            detail_lower = detail.lower()
            not_found_phrases = ["not specified", "not found", "n/a"]
            if detail_lower not in not_found_phrases:
                all_not_found = False

        # This filtering might be less necessary if the LLM only returns differences,
        # but it's good to keep as a safeguard.
        if analysis_lower == "not found in any contract.":
            skip_row = True
        elif all_not_found and analysis_lower in ["not specified", "not found", "n/a"]:
            skip_row = True
        
        if skip_row:
            continue 

        analysis_class = "difference" # Default to difference as per new prompt focus
        if "no significant difference" in analysis_lower or "similar" in analysis_lower:
            analysis_class = "no-difference" # Should be rare with new prompt

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
        added_rows +=1

    html_content += """
            </tbody>
        </table>
    """
    if added_rows == 0:
         html_content += "<p>No significant differences were identified by the LLM for comparison, or all identified items were filtered out.</p>"
    
    html_content += """
    </body>
    </html>
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML report generated successfully at {output_path}")
        if added_rows == 0 and comparison_data: 
            print("Note: The LLM returned data, but no items met the criteria for display after filtering, or no differences were reported.")
    except Exception as e:
        print(f"Error writing HTML report: {e}")


def compare_contracts_with_llm(contract_texts, contract_labels):
    """Compares multiple contract texts using Gemini LLM and returns structured JSON focusing on differences."""
    if not API_KEY:
        print("LLM comparison skipped: API key is missing.")
        return None

    max_len = 30000 
    truncated_texts = []
    for i, contract_text in enumerate(contract_texts):
        if len(contract_text) > max_len:
            print(f"Warning: Contract {i+1} text truncated to {max_len} characters for LLM.")
            truncated_texts.append(contract_text[:max_len])
        else:
            truncated_texts.append(contract_text)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=API_KEY)

    # Build dynamic contract sections for the prompt
    contract_sections = ""
    contract_variables = []
    for i, label in enumerate(contract_labels):
        contract_sections += f"\n    Contract {label}:\n    ---\n    {{contract_{chr(65+i).lower()}_text}}\n    ---\n"
        contract_variables.append(f"contract_{chr(65+i).lower()}_text")

    # Build dynamic response format description
    response_format_examples = ""
    for i, label in enumerate(contract_labels):
        response_format_examples += f'"contract_{chr(65+i).lower()}_detail": (string, the relevant detail or excerpt from Contract {label} pertaining to this differing aspect. If the aspect is missing in {label} but present in others, state "Not present in Contract {label}" or similar.)\n    '

    # New prompt: Asks LLM to identify differing aspects across multiple contracts.
    prompt_template_str = f"""
    You are an expert legal assistant specializing in contract review and comparison.
    Your task is to meticulously review the {len(contract_labels)} contracts provided below.

    First, thoroughly read and understand all contracts.
    Then, identify the key clauses, terms, or aspects where there are material differences between any of the contracts.
    Consider aspects such as (but not limited to, and only if they differ):
    - Parties involved
    - Effective Dates or Execution Dates
    - Contract Duration or Term
    - Governing Law and Jurisdiction
    - Payment Terms (amounts, schedules, methods)
    - Scope of Work, Supply, or Services
    - Confidentiality obligations
    - Limitations of Liability
    - Termination rights and procedures
    - Dispute resolution mechanisms
    - Force Majeure provisions
    - Assignment rights
    - Notice requirements
    - Any unique or non-standard clauses that show variation.

    For EACH identified material difference, provide a concise summary.
    If a clause is present in some contracts but entirely absent or significantly different in others, highlight this as a key difference.
    Do NOT list aspects that are identical or substantially similar across all contracts. Focus only on the differences.

    Format your response as a JSON array of objects. Each object in the array should represent one identified difference and must have the following keys:
    "clause_category": (string, e.g., "Effective Date Discrepancy", "Parties - Purchaser Identity", "Governing Law Variation")
    {response_format_examples}"analysis_of_difference": (string, a brief explanation of the nature and potential implication of this difference across the contracts.)

    If, after careful review, you find NO material differences between the contracts, return an empty JSON array: [].
    {contract_sections}
    JSON Output (ensure this is a valid JSON array, focusing only on differences):
    """
    
    prompt = PromptTemplate(
        input_variables=contract_variables,
        template=prompt_template_str,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        print(f"Sending request to LLM for comparison of {len(contract_texts)} contracts (focusing on differences)...")
        
        # Build the input dictionary dynamically
        chain_input = {}
        for i, contract_text in enumerate(truncated_texts):
            chain_input[f"contract_{chr(65+i).lower()}_text"] = contract_text
        
        response = chain.invoke(chain_input)
        
        llm_output_text = response.get('text', '').strip()
        
        if llm_output_text.startswith("```json"):
            llm_output_text = llm_output_text[7:]
        if llm_output_text.endswith("```"):
            llm_output_text = llm_output_text[:-3]
        llm_output_text = llm_output_text.strip()

        print("LLM Raw Output (first 500 chars):", llm_output_text[:500]) 

        if not llm_output_text: # Handle empty string from LLM
            print("LLM returned an empty response. Assuming no differences found.")
            return []

        comparison_data = json.loads(llm_output_text)
        if not isinstance(comparison_data, list):
            print("Error: LLM output was not a JSON list as expected.")
            if isinstance(comparison_data, dict) and "clause_category" in comparison_data:
                 print("Attempting to wrap single dictionary into a list.")
                 comparison_data = [comparison_data]
            else: # If it's not a list and not a single item we can wrap, treat as error
                print("LLM output structure is not a recognized list or single item. Treating as no valid data.")
                return None # Or [] if you prefer to show "no differences"
        return comparison_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM: {e}")
        print("LLM Output that caused error:", llm_output_text)
        return None # Or [] to indicate no valid differences parsed
    except Exception as e:
        print(f"An error occurred during LLM comparison: {e}")
        return None # Or []

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare 2-10 PDF contracts and generate an HTML report on their differences.")
    parser.add_argument("pdf_paths", nargs='+', help="Paths to the PDF contract files (minimum 2, maximum 10).")
    parser.add_argument("--output", default="contract_comparison_report.html", help="Path for the output HTML report.")
    
    args = parser.parse_args()

    # Validate number of contracts
    num_contracts = len(args.pdf_paths)
    if num_contracts < 2:
        print("Error: At least 2 PDF files are required for comparison.")
        exit(1)
    elif num_contracts > 10:
        print("Error: Maximum 10 PDF files are supported for comparison.")
        exit(1)

    if not API_KEY:
        print("Exiting: GOOGLE_API_KEY is not configured.")
    else:
        print(f"Comparing {num_contracts} contracts:")
        for i, pdf_path in enumerate(args.pdf_paths, 1):
            print(f"  Contract {i}: {pdf_path}")

        # Extract text from all PDFs
        contract_texts = []
        contract_labels = []
        failed_extractions = []
        
        for i, pdf_path in enumerate(args.pdf_paths):
            contract_text = extract_text_from_pdf(pdf_path)
            if contract_text:
                contract_texts.append(contract_text)
                contract_labels.append(chr(65 + i))  # A, B, C, D, etc.
            else:
                failed_extractions.append(f"Contract {chr(65 + i)} ({pdf_path})")

        if failed_extractions:
            print(f"Warning: Failed to extract text from: {', '.join(failed_extractions)}")

        comparison_results = None
        if len(contract_texts) >= 2:
            # Update labels to match successfully extracted contracts
            contract_labels = [chr(65 + i) for i in range(len(contract_texts))]
            comparison_results = compare_contracts_with_llm(contract_texts, contract_labels)
        else:
            print("Could not proceed with comparison due to insufficient valid contracts (need at least 2).")

        # generate_html_report will handle None or empty list for comparison_results
        generate_html_report(comparison_results, contract_labels, args.output)
        
        if comparison_results is None:
             print("Comparison failed or LLM did not return valid data.")
        elif not comparison_results: # Empty list
             print("LLM indicated no material differences found, or the response was empty.")