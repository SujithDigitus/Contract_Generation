import os
from langchain_google_genai import ChatGoogleGenerativeAI # Changed import
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from dotenv import load_dotenv # Ensure this is here
import fitz  # PyMuPDF for PDF text extraction
import re

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv() # Call it here for this module
import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
# Ensure you have your GOOGLE_API_KEY set in your environment
# or configure it directly if not using environment variables.

def strip_markdown_json(json_string: str) -> str:
    """
    Removes Markdown code block fences (```json ... ``` or ``` ... ```)
    from a string, which LLMs sometimes add around JSON output.
    Handles cases where fences might be incomplete due to truncation.
    """
    # print(f"\n--- DEBUG (strip_markdown_json): Received input string (first 100 chars): '{str(json_string)[:100]}'") # Optional: uncomment for deep debugging
    if json_string is None:
        # print("--- DEBUG (strip_markdown_json): Input is None, returning empty string.") # Optional: uncomment for deep debugging
        return ""
    
    # Normalize line breaks and strip leading/trailing whitespace from the whole string first
    text = str(json_string).replace('\r\n', '\n').replace('\r', '\n').strip()
    # print(f"--- DEBUG (strip_markdown_json): Text after initial normalization and strip (first 100 chars): '{text[:100]}'") # Optional: uncomment for deep debugging

    # Check for and remove common complete fence patterns first
    if text.startswith("```json\n") and text.endswith("\n```"):
        # print("--- DEBUG (strip_markdown_json): Matched ```json\\n ... \\n```") # Optional: uncomment for deep debugging
        return text[7:-4].strip() 
    if text.startswith("```json") and text.endswith("```"): # Handles no newline after ```json
        # print("--- DEBUG (strip_markdown_json): Matched ```json ... ```") # Optional: uncomment for deep debugging
        return text[7:-3].strip()
    if text.startswith("```\n") and text.endswith("\n```"): # For generic ``` blocks with newlines
        # print("--- DEBUG (strip_markdown_json): Matched ```\\n ... \\n```") # Optional: uncomment for deep debugging
        return text[4:-4].strip()
    if text.startswith("```") and text.endswith("```"): # For generic ``` blocks without newlines
        # print("--- DEBUG (strip_markdown_json): Matched ``` ... ```") # Optional: uncomment for deep debugging
        return text[3:-3].strip()

    # If complete fences weren't matched, aggressively strip known prefixes
    # This helps if the LLM output was truncated and the closing fence is missing.
    original_text_before_prefix_strip = text # For debugging
    stripped_prefix = False
    if text.startswith("```json\n"):
        # print("--- DEBUG (strip_markdown_json): Stripping prefix ```json\\n") # Optional: uncomment for deep debugging
        text = text[7:]
        stripped_prefix = True
    elif text.startswith("```json"): 
        # print("--- DEBUG (strip_markdown_json): Stripping prefix ```json") # Optional: uncomment for deep debugging
        text = text[7:]
        stripped_prefix = True
    elif text.startswith("```\n"):
        # print("--- DEBUG (strip_markdown_json): Stripping prefix ```\\n") # Optional: uncomment for deep debugging
        text = text[4:]
        stripped_prefix = True
    elif text.startswith("```"): 
        # print("--- DEBUG (strip_markdown_json): Stripping prefix ```") # Optional: uncomment for deep debugging
        text = text[3:]
        stripped_prefix = True
    
    # if stripped_prefix: # Optional: uncomment for deep debugging
        # print(f"--- DEBUG (strip_markdown_json): Text after prefix strip (first 100 chars): '{text[:100]}'") # Optional: uncomment for deep debugging
    # else: # Optional: uncomment for deep debugging
        # print(f"--- DEBUG (strip_markdown_json): No prefix stripped. Original was (first 100 chars): '{original_text_before_prefix_strip[:100]}'") # Optional: uncomment for deep debugging
        
    final_stripped_text = text.strip()
    # print(f"--- DEBUG (strip_markdown_json): Returning final stripped text (first 100 chars): '{final_stripped_text[:100]}'") # Optional: uncomment for deep debugging
    return final_stripped_text

def rag_pipeline_with_prompt(context: str) -> str:
    """
    Analyzes text in contracts using a Gemini LLM to extract a template,
    placeholders, their descriptions, and their original values.
    Placeholders are bare words (e.g., Party_Name).

    Returns a string, which should be a JSON object.
    """
    # Initialize the LLM with Gemini
    # Ensure GOOGLE_API_KEY is set in your environment.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0, convert_system_message_to_human=True)

    template = """
Your **absolute primary objective** is to meticulously analyze the ENTIRE provided contract text and identify **EVERY SINGLE specific, replaceable piece of information.** Do not omit any relevant details. This is crucial for creating a versatile and comprehensive template.

**Step 1: Comprehensive Identification - Find ALL Changeable Data**
Examine the document thoroughly. Your foremost task is to locate ALL text segments that are specific to this particular instance of the document and would likely change if the document were for different parties, a different subject, a different date, or a different context. Think broadly and be exhaustive. Examples include:
- Names (individuals, companies, organizations, signatories, beneficiaries)
- Dates (effective dates, execution dates, birth dates, start/end dates, deadlines)
- Locations and Addresses (physical addresses, email addresses, notice addresses)
- Contact Information (phone numbers, URLs, social media links)
- Monetary amounts, percentages, quantities, payment terms, itemized lists
- Specific legal or technical terms if they are instance-specific (e.g., a specific project code name, a unique case ID)
- Detailed descriptions or statements (e.g., scope of work, project goals, qualifications, duties, specific obligations or covenants)
- Titles (of people, sections if instance-specific, documents, projects)
- Any other unique identifiers or data points specific to this version of the document.
This information might appear as standalone text (e.g., a name mentioned in a paragraph, a descriptive clause) OR be associated with a clear label (e.g., "Party A:" followed by a name, "Notice Period:" followed by a duration). IDENTIFY ALL SUCH INSTANCES.

**Step 2: For EACH Identified Piece of Information - Define Placeholder Details**
For EVERY piece of information you identified in Step 1:
1.  **Placeholder Name:** Create a unique, descriptive name using only letters, numbers, and underscores (e.g., `Primary_Party_Name`, `Agreement_Effective_Date`).
    * If the information clearly followed a label in the text (e.g., "Effective Date: January 1st, 2025"), use the label to help form the placeholder name (e.g., `Effective_Date`).
2.  **'original_value' (Data ONLY, Complete & Verbatim - ABSOLUTELY CRITICAL):**
    * This field MUST contain **ONLY the actual data content/value** that was identified.
    * **If the information was preceded by a label in the source text** (e.g., "Effective Date: January 1st, 2025"), the `original_value` MUST BE **ONLY THE DATA PART** (e.g., "January 1st, 2025").
    * **CRUCIAL CLARIFICATION: The label itself (like "Effective Date:") MUST NOT, under any circumstances, be part of the `original_value` string.** The `original_value` is the pure data that followed the label.
    * **If the information was standalone text** (not part of an obvious 'Label: Value' structure, e.g., a paragraph detailing specific terms), the `original_value` is that ENTIRE standalone piece of text.
    * **NON-NEGOTIABLE COMPLETENESS AND ACCURACY:** The `original_value` MUST BE THE E X A C T, V E R B A T I M, C O M P L E T E, AND U N A B R I D G E D text for its corresponding data part.
        * ABSOLUTELY NO TRUNCATION. NO SUMMARIZATION. NO ELLIPSES (...). This applies equally to short values, long lists, multi-sentence paragraphs, etc. If a list of deliverables has ten items, all ten items (the complete string, e.g., "Report A, Analysis B, Prototype C") must be the `original_value`. If a clause is five sentences long, that entire five-sentence text is the `original_value`. Any omission or shortening is a failure.
3.  **`description`:** Briefly explain what the placeholder represents (e.g., "The legal name of the first contracting party", "The specific date the agreement comes into force"). Make descriptions general enough if the type of information could appear in various documents.

**Step 3: Constructing the `Template` String**
The `Template` string must be the full original input text, modified as follows:
1.  For each piece of information identified:
    * **If it had a label** (e.g., "Effective Date: January 1st, 2025" or "Selected Coursework: Web Tech"), the `Template` string MUST **preserve the original label** (e.g., "Effective Date: " or "Selected Coursework: ") and then replace ONLY THE VALUE PART with the placeholder. Example: "Effective Date: `Effective_Date_Placeholder`" or "Selected Coursework: `Selected_Coursework_Placeholder`".
    * If it was standalone text, that entire segment of standalone text is replaced by its placeholder in the `Template`. Example: "`Scope_Of_Work_Paragraph_Placeholder`".
2.  Preserve all original formatting, indentation, and line breaks around the placeholders and any remaining static text.

**Step 4: JSON Output Format**
Return a single, valid JSON object with the following exact structure. Crucially, ensure **ALL** identified placeholders from Step 1 are included in the "Placeholders" object.

```json
{{
  "Template": "<The full input text, with ONLY THE VALUE PARTS of identified information replaced by their respective placeholders. Labels that preceded values in the original text must remain as static text in the template. Example: 'This Agreement (the "Agreement") is entered into by and between First_Party_Legal_Name ("Party A") and Second_Party_Legal_Name ("Party B"), effective as of Agreement_Effective_Date. The primary services to be rendered are detailed in Service_Description_Paragraph. All notices shall be sent to Notice_Recipient_Email_Address. List of Attachments: List_Of_Attachment_Names.'>",
  "Placeholders": {{
    "Example_Contract_Party_Name": {{
      "description": "The full legal name of a contracting party.",
      "original_value": "<E.g., 'Acme Innovations LLC' - This value must be EXACTLY as it appears in the source, COMPLETE, and must NOT include any preceding label like 'Party A:'. NO TRUNCATION.>"
    }},
    "Example_Specific_Date_After_Label": {{
      "description": "A specific date mentioned in the document, such as an effective date or a deadline, that followed a label.",
      "original_value": "<E.g., if source was 'Effective Date: March 15, 2026', the original_value is EXACTLY 'March 15, 2026'. The label 'Effective Date:' IS NOT part of this value. EXACT, COMPLETE. NO TRUNCATION.>"
    }},
    "Example_Descriptive_Clause_Or_Paragraph_Standalone": {{
      "description": "A standalone paragraph or clause detailing specific terms, an abstract, a scope of work, or a detailed description (not immediately following a 'Label: Value' structure).",
      "original_value": "<The entire, multi-sentence paragraph or clause text, verbatim and complete. E.g., 'The Consultant shall perform the services with a professional degree of skill and care, consistent with industry standards. All intellectual property developed hereunder shall be the property of the Client upon full payment.' - This ENTIRE text must be the value. NO TRUNCATION.>"
    }}
    // ... and so on for ALL identified placeholders. Do not omit any.
  }}
}}
```

**FINAL INSTRUCTIONS - CRITICAL PRIORITIES (Reiteration for Emphasis):**
1.  **COMPREHENSIVE RECALL (Most Important):** You MUST identify and include placeholders for **EVERY** relevant, specific, and replaceable piece of information from the **ENTIRE** document. Do not be overly conservative; if it's specific data that could change, extract it.
2.  **`original_value` INTEGRITY - DATA ONLY & COMPLETE:**
    * The `original_value` must be the **PURE DATA VALUE ONLY**.
    * It must **NOT** contain any part of the preceding label from the source text (e.g., if source is "Client Name: XYZ Corp", `original_value` is "XYZ Corp", NOT "Client Name: XYZ Corp").
    * It must be **100% COMPLETE and VERBATIM**. Absolutely **NO TRUNCATION or ellipses (...)** are permitted, regardless of length.
3.  **ACCURATE TEMPLATE AND LABEL HANDLING:** The `Template` string must correctly preserve original labels (where they existed in the source text) and correctly substitute placeholders for **only the value parts**.
4.  Respond ONLY with a single, valid JSON object. No other text or explanations.

Input context:
{context}
"""

    prompt_final = PromptTemplate(
        template=template,
        input_variables=["context"]
    )

    # Define the RAG chain
    rag_chain = (
        {"context": itemgetter("context")}
        | prompt_final
        | llm
        | StrOutputParser()
    )

    # Invoke the chain with the context
    print("--- DEBUG (rag_pipeline_with_prompt): About to invoke LLM chain with Gemini...")
    result_str = rag_chain.invoke({"context": context})
    print(f"--- DEBUG (rag_pipeline_with_prompt): Gemini LLM chain returned (first 100 chars): '{str(result_str)[:100]}'")
    
    # Clean the LLM output (remove markdown fences if present)
    cleaned_result_str = strip_markdown_json(result_str)
    print(f"--- DEBUG (rag_pipeline_with_prompt): After strip_markdown_json (first 100 chars): '{cleaned_result_str[:100]}'")
    
    return cleaned_result_str

def generate_contract_from_template(template: str, placeholders_dict: dict) -> str:
    """
    Generates a final contract by replacing placeholders in the template with actual values.
    
    Args:
        template (str): The contract template with placeholders
        placeholders_dict (dict): Dictionary mapping placeholder names to their replacement values
    
    Returns:
        str: The final contract with all placeholders replaced
    """
    final_contract = template
    
    print("--- DEBUG (generate_contract_from_template): Starting placeholder replacement...")
    
    for placeholder_name, placeholder_info in placeholders_dict.items():
        # Get the actual value to replace with
        if isinstance(placeholder_info, dict) and 'value' in placeholder_info:
            replacement_value = placeholder_info['value']
        elif isinstance(placeholder_info, dict) and 'original_value' in placeholder_info:
            replacement_value = placeholder_info['original_value']
        else:
            replacement_value = str(placeholder_info)
        
        # Replace the placeholder in the template
        final_contract = final_contract.replace(placeholder_name, replacement_value)
        print(f"--- DEBUG: Replaced {placeholder_name} with: {replacement_value[:50]}...")
    
    print("--- DEBUG (generate_contract_from_template): Placeholder replacement completed")
    return final_contract

def interactive_contract_modifier(current_contract: str, modification_request: str) -> str:
    """
    Uses Gemini LLM to interactively modify contract sections based on user requests.
    This function works on the FINAL generated contract (after placeholders have been filled).
    Allows users to add, remove, or modify specific sections of the contract.
    
    Args:
        current_contract (str): The current fully generated contract text (with placeholders filled)
        modification_request (str): User's request for modifications (e.g., "Add a confidentiality section", 
                                  "Remove the penalty clause", "Modify the payment terms to include quarterly payments")
    
    Returns:
        str: The modified contract text
    """
    # Initialize the LLM with Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.1, convert_system_message_to_human=True)

    modification_template = """
You are an expert contract modification assistant. Your task is to intelligently modify a contract based on the user's specific request while maintaining legal coherence, professional language, and proper contract structure.

**MODIFICATION INSTRUCTIONS:**

1. **Analyze the Request**: Carefully understand what the user wants to:
   - ADD: Insert new sections, clauses, or terms
   - REMOVE: Delete specific sections, clauses, or terms
   - MODIFY: Change existing content, terms, or conditions

2. **Maintain Contract Integrity**: 
   - Preserve the overall structure and flow of the contract
   - Ensure all modifications use appropriate legal language
   - Keep consistent formatting and style with the existing contract
   - Maintain logical section numbering if present

3. **Smart Modifications**:
   - For ADDITIONS: Insert new content in the most appropriate location within the contract structure
   - For REMOVALS: Clean up any references to removed sections and adjust numbering if needed
   - For MODIFICATIONS: Update the specified content while preserving the intent and legal validity

4. **Legal Considerations**:
   - Ensure modifications are legally sound and properly worded
   - Maintain consistency with other contract terms
   - Use standard legal terminology and phrasing
   - Preserve essential contract elements (parties, consideration, terms, etc.)

5. **Output Requirements**:
   - Return ONLY the complete modified contract text
   - Do not include explanations, comments, or additional text
   - Preserve all original formatting, spacing, and structure where not modified
   - Ensure the final contract is coherent and professionally formatted

**Current Contract:**
{current_contract}

**User's Modification Request:**
{modification_request}

**CRITICAL**: Respond with ONLY the complete modified contract text. No explanations, comments, or additional text should be included.
"""

    prompt = PromptTemplate(
        template=modification_template,
        input_variables=["current_contract", "modification_request"]
    )

    # Define the modification chain
    modification_chain = (
        {"current_contract": itemgetter("current_contract"), "modification_request": itemgetter("modification_request")}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Invoke the chain
    print("--- DEBUG (interactive_contract_modifier): About to invoke LLM chain with Gemini for contract modification...")
    print(f"--- DEBUG (interactive_contract_modifier): Modification request: '{modification_request[:100]}...'")
    
    result_str = modification_chain.invoke({
        "current_contract": current_contract,
        "modification_request": modification_request
    })
    
    print(f"--- DEBUG (interactive_contract_modifier): Gemini LLM chain returned modified contract (first 100 chars): '{str(result_str)[:100]}'")
    
    # Clean any potential markdown formatting
    cleaned_result = strip_markdown_json(result_str)
    
    return cleaned_result

def batch_contract_modifications(current_contract: str, modification_requests: list) -> str:
    """
    Applies multiple modifications to a contract in sequence.
    This works on the FINAL generated contract (after placeholders have been filled).
    
    Args:
        current_contract (str): The initial fully generated contract text (with placeholders filled)
        modification_requests (list): List of modification requests to apply in order
    
    Returns:
        str: The final modified contract after all modifications
    """
    modified_contract = current_contract
    
    for i, request in enumerate(modification_requests):
        print(f"--- DEBUG (batch_contract_modifications): Applying modification {i+1}/{len(modification_requests)}")
        modified_contract = interactive_contract_modifier(modified_contract, request)
    
    return modified_contract

def get_contract_sections_summary(contract_text: str) -> str:
    """
    Uses Gemini LLM to analyze and summarize the main sections of a contract.
    Useful for users to understand what sections are available for modification.
    
    Args:
        contract_text (str): The contract text to analyze
    
    Returns:
        str: A summary of the contract's main sections and their purposes
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0, convert_system_message_to_human=True)

    summary_template = """
You are a contract analysis expert. Analyze the provided contract and create a clear, concise summary of its main sections and their purposes. This will help users understand the contract structure for potential modifications.

**Instructions:**
1. Identify all major sections, clauses, and subsections in the contract
2. Provide a brief description of what each section covers
3. Use a clear, numbered or bulleted format
4. Focus on the key areas that users might want to modify
5. Keep descriptions concise but informative

**Contract Text:**
{contract_text}

**Required Output Format:**
Provide a structured summary like this:

CONTRACT SECTIONS SUMMARY:
1. [Section Name] - [Brief description of what this section covers]
2. [Section Name] - [Brief description of what this section covers]
...

Include any notable clauses, terms, or special provisions that users might want to modify.
"""

    prompt = PromptTemplate(
        template=summary_template,
        input_variables=["contract_text"]
    )

    chain = prompt | llm | StrOutputParser()
    
    print("--- DEBUG (get_contract_sections_summary): Analyzing contract sections...")
    result = chain.invoke({"contract_text": contract_text})
    
    return result.strip()

def get_text_from_Pdf(file_path: str) -> str:
    """
    Extracts all text from a PDF file.
    """
    try:
        doc = fitz.open(file_path)
        full_text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            full_text += page.get_text()
        doc.close()
        return full_text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        raise

def clean_text_for_llm(text: str) -> str:
    """
    Cleans text by removing or replacing control characters and other
    potentially problematic characters before sending to the LLM.
    """
    if text is None:
        return ""
    # Normalize line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove most ASCII control characters (0x00-0x1F, 0x7F)
    # We keep \n (newline) and \t (tab) as they are common and should be
    # handled (escaped) by the LLM when generating JSON strings.
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    return text

def format_text_to_html_with_llm(plain_text_content: str, style_instructions: str = "Use generic professional styling") -> str:
    """
    Uses a Gemini LLM to format plain text content into a styled HTML document.
    """
    # Initialize the LLM with Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.2, convert_system_message_to_human=True)

    html_styler_prompt_template = """
You are an expert document stylist. Your task is to take the following plain text content and reformat it into a single, complete, and well-structured HTML document string.
Apply rich text styling such as headings (H1, H2, H3), subheadings, bold text, italic text, paragraphs, and lists (bulleted or numbered).
The goal is to make the document highly readable and professional-looking.

Input Plain Text Content:
---------------------------
{plain_text_content}
---------------------------

Styling Instructions:
{style_instructions}

Specific Formatting Tasks:
1.  Analyze the structure of the plain text to identify titles, headings (try to discern different levels like H1, H2, H3 based on prominence or common patterns like all-caps lines or lines with few words), subheadings, paragraphs, and any lists.
2.  Generate a complete HTML document (must include `<!DOCTYPE html>`, `<html>`, `<head>`, `<style>`, and `<body>` tags).
3.  In the `<style>` section of the HTML `<head>`, define appropriate CSS rules.
    * If specific 'Styling Instructions' are provided above and are not "Use generic professional styling", adhere to them as closely as possible for font families, sizes, colors, alignments, margins, etc.
    * If 'Styling Instructions' are "Use generic professional styling" or are minimal/absent, apply the following default professional styling:
        - `body {{ font-family: Arial, Helvetica, sans-serif; line-height: 1.6; margin: 30px; color: #333333; background-color: #fdfdfd; }}`
        - `.document-title {{ text-align: center; font-size: 28px; font-weight: bold; margin-bottom: 30px; color: #1a1a1a; }}`
        - `h1 {{ font-size: 22px; font-weight: bold; margin-top: 25px; margin-bottom: 15px; color: #2c2c2c; border-bottom: 2px solid #eeeeee; padding-bottom: 8px; }}`
        - `h2 {{ font-size: 20px; font-weight: bold; margin-top: 22px; margin-bottom: 12px; color: #2c2c2c; }}`
        - `h3 {{ font-size: 18px; font-weight: bold; margin-top: 20px; margin-bottom: 10px; color: #333333; }}`
        - `p {{ margin-bottom: 15px; text-align: left; }}` /* Consider 'text-align: justify;' for very formal documents if appropriate */
        - `strong {{ font-weight: bold; }}`
        - `em {{ font-style: italic; }}`
        - `ul, ol {{ margin-left: 20px; padding-left: 20px; margin-bottom: 15px; }}`
        - `li {{ margin-bottom: 6px; }}`
        - `table {{ width: 100%; border-collapse: collapse; margin-bottom: 15px; }}`
        - `th, td {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}`
        - `th {{ background-color: #f2f2f2; font-weight: bold; }}`
4.  In the `<body>`, use appropriate HTML semantic tags (e.g., `<header>`, `<article>`, `<section>`, `<h1>`, `<h2>`, `<h3>`, `<p>`, `<strong>`, `<em>`, `<ul>`, `<ol>`, `<li>`, `<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>`).
    * Try to identify a main title for the document and wrap it in a `<div>` with class `document-title` or an `<h1>` if it's the primary heading.
5.  Ensure the output is ONLY the complete HTML document string. Do not include any other explanatory text, apologies, or conversational remarks before or after the HTML code.

Produce the HTML output now.
"""
    prompt = PromptTemplate.from_template(html_styler_prompt_template)
    
    chain = prompt | llm | StrOutputParser()
    
    print("--- DEBUG (format_text_to_html_with_llm): About to invoke LLM chain with Gemini for HTML styling...")
    html_output = chain.invoke({
        "plain_text_content": plain_text_content,
        "style_instructions": style_instructions
    })
    print(f"--- DEBUG (format_text_to_html_with_llm): Gemini LLM chain returned for HTML (first 100 chars): '{str(html_output)[:100]}'")

    # Basic check to ensure it looks like HTML
    # Also, clean potential markdown fences from HTML output, as LLMs might add them.
    cleaned_html_output = strip_markdown_json(html_output) # Clean markdown like ```html ... ```

    if cleaned_html_output and "<html>" in cleaned_html_output.lower() and "</html>" in cleaned_html_output.lower():
        return cleaned_html_output
    else:
        # Fallback or error handling if LLM didn't produce expected HTML
        print(f"Warning: Gemini LLM output for HTML styling might be invalid or missing HTML tags after cleaning. Output preview: {cleaned_html_output[:500]}")
        # Return a simple HTML representation of the plain text as a fallback
        escaped_plain_text = plain_text_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<!DOCTYPE html><html><head><title>Formatted Document (Fallback)</title><style>body{{font-family:sans-serif;}} pre{{white-space:pre-wrap;}}</style></head><body><pre>{escaped_plain_text}</pre></body></html>"


# # Complete workflow example for contract generation and modification
# def complete_contract_workflow_example():
#     """
#     Complete example showing the full workflow:
#     1. Extract template and placeholders from source
#     2. Fill placeholders to generate final contract
#     3. Interactively modify the generated contract
#     """
#     print("=== Complete Contract Generation and Modification Workflow ===\n")
    
#     # Step 1: Simulate extracting template from a source document
#     print("1. EXTRACTING TEMPLATE FROM SOURCE DOCUMENT")
#     print("-" * 50)
    
#     # Example source contract (this would come from your PDF extraction)
#     source_contract = """
#     SERVICE AGREEMENT
    
#     This Service Agreement ("Agreement") is entered into on January 1, 2024, between ABC Corp ("Client") and XYZ Services ("Provider").
    
#     1. SCOPE OF WORK
#     Provider agrees to deliver web development services including website design, backend development, and testing.
    
#     2. PAYMENT TERMS
#     Client agrees to pay $5,000 upon completion of services.
    
#     3. TERM
#     This agreement shall remain in effect for 6 months from the effective date.
    
#     4. TERMINATION
#     Either party may terminate this agreement with 30 days written notice.
#     """
    
#     # Simulate the template extraction (this would use your rag_pipeline_with_prompt function)
#     # For demo purposes, here's what the extracted template and placeholders would look like:
#     extracted_template = """
#     SERVICE AGREEMENT
    
#     This Service Agreement ("Agreement") is entered into on Agreement_Date, between Client_Company_Name ("Client") and Provider_Company_Name ("Provider").
    
#     1. SCOPE OF WORK
#     Provider agrees to deliver Service_Description.
    
#     2. PAYMENT TERMS
#     Client agrees to pay Payment_Amount upon completion of services.
    
#     3. TERM
#     This agreement shall remain in effect for Agreement_Duration from the effective date.
    
#     4. TERMINATION
#     Either party may terminate this agreement with Termination_Notice_Period written notice.
#     """
    
#     extracted_placeholders = {
#         "Agreement_Date": {
#             "description": "The date when the agreement is entered into",
#             "original_value": "January 1, 2024",
#             "value": "March 15, 2025"  # New value for the new contract
#         },
#         "Client_Company_Name": {
#             "description": "The legal name of the client company",
#             "original_value": "ABC Corp",
#             "value": "TechStart LLC"  # New value
#         },
#         "Provider_Company_Name": {
#             "description": "The legal name of the service provider company", 
#             "original_value": "XYZ Services",
#             "value": "Digital Solutions Inc"  # New value
#         },
#         "Service_Description": {
#             "description": "Detailed description of services to be provided",
#             "original_value": "web development services including website design, backend development, and testing",
#             "value": "mobile app development services including iOS and Android applications, API integration, and user testing"  # New value
#         },
#         "Payment_Amount": {
#             "description": "The total payment amount for services",
#             "original_value": "$5,000",
#             "value": "$15,000"  # New value
#         },
#         "Agreement_Duration": {
#             "description": "The duration for which the agreement remains in effect",
#             "original_value": "6 months", 
#             "value": "12 months"  # New value
#         },
#         "Termination_Notice_Period": {
#             "description": "The notice period required for contract termination",
#             "original_value": "30 days",
#             "value": "60 days"  # New value
#         }
#     }
    
#     print("Template extracted with placeholders identified.")
#     print(f"Found {len(extracted_placeholders)} placeholders to fill.")
    
#     # Step 2: Generate the final contract with updated placeholder values
#     print("\n2. GENERATING FINAL CONTRACT WITH NEW VALUES")
#     print("-" * 50)
    
#     generated_contract = generate_contract_from_template(extracted_template, extracted_placeholders)
#     print("Generated Contract:")
#     print(generated_contract)
    
#     # Step 3: Get contract sections summary for user reference
#     print("\n3. ANALYZING CONTRACT SECTIONS")
#     print("-" * 50)
    
#     sections_summary = get_contract_sections_summary(generated_contract)
#     print("Available sections for modification:")
#     print(sections_summary)
    
#     # Step 4: Apply interactive modifications to the generated contract
#     print("\n4. APPLYING INTERACTIVE MODIFICATIONS")
#     print("-" * 50)
    
#     # Example modifications that a user might request
#     print("Modification 1: Adding confidentiality section...")
#     modified_contract = interactive_contract_modifier(
#         generated_contract, 
#         "Add a comprehensive confidentiality section that requires both parties to keep all business information, technical details, and financial terms confidential for 3 years after the agreement ends"
#     )
    
#     print("Modification 2: Adding intellectual property rights...")
#     modified_contract = interactive_contract_modifier(
#         modified_contract,
#         "Add a section stating that all intellectual property created during this project will be owned by the client, but the provider retains the right to use general methodologies and non-proprietary techniques"
#     )
    
#     print("Modification 3: Adding liability limitation...")
#     modified_contract = interactive_contract_modifier(
#         modified_contract,
#         "Add a liability limitation clause that limits each party's liability to the total amount paid under this agreement, except in cases of willful misconduct or gross negligence"
#     )
    
#     print("Modification 4: Updating payment terms...")
#     modified_contract = interactive_contract_modifier(
#         modified_contract,
#         "Change the payment structure to 30% upfront, 40% at project milestone completion, and 30% upon final delivery and acceptance"
#     )
    
#     # Step 5: Show final result
#     print("\n5. FINAL MODIFIED CONTRACT")
#     print("=" * 70)
#     print(modified_contract)
#     print("=" * 70)
    
#     # Step 6: Convert to HTML if needed
#     print("\n6. FORMATTING TO HTML")
#     print("-" * 50)
    
#     final_html = format_text_to_html_with_llm(
#         modified_contract, 
#         "Use professional legal document styling with clear section headers and proper spacing"
#     )
#     print("HTML version generated successfully.")
#     print(f"HTML length: {len(final_html)} characters")
    
#     return {
#         'original_contract': source_contract,
#         'template': extracted_template,
#         'placeholders': extracted_placeholders,
#         'generated_contract': generated_contract,
#         'final_modified_contract': modified_contract,
#         'html_version': final_html
#     }

# # Example usage function for the interactive features (updated)
# def example_interactive_workflow():
#     """
#     Updated example showing how to use interactive modifications AFTER contract generation.
#     """
#     print("=== Interactive Contract Modification After Generation ===\n")
    
#     # Assume you already have a generated contract (after placeholders were filled)
#     generated_contract = """
#     SERVICE AGREEMENT
    
#     This Service Agreement ("Agreement") is entered into on March 15, 2025, between TechStart LLC ("Client") and Digital Solutions Inc ("Provider").
    
#     1. SCOPE OF WORK
#     Provider agrees to deliver mobile app development services including iOS and Android applications, API integration, and user testing.
    
#     2. PAYMENT TERMS
#     Client agrees to pay $15,000 upon completion of services.
    
#     3. TERM
#     This agreement shall remain in effect for 12 months from the effective date.
    
#     4. TERMINATION
#     Either party may terminate this agreement with 60 days written notice.
#     """
    
#     print("Generated Contract (with placeholders filled):")
#     print("-" * 60)
#     print(generated_contract)
#     print("-" * 60)
    
#     # Now apply interactive modifications
#     print("\n1. Getting contract sections summary...")
#     sections_summary = get_contract_sections_summary(generated_contract)
#     print("Contract Sections:")
#     print(sections_summary)
    
#     print("\n2. Applying modifications to the generated contract...")
    
#     # Add new sections and modify existing ones
#     print("\nAdding confidentiality section...")
#     modified_contract = interactive_contract_modifier(
#         generated_contract, 
#         "Add a confidentiality section that states both parties must keep all project information confidential for 2 years after the agreement ends"
#     )
    
#     print("Modifying payment terms...")
#     modified_contract = interactive_contract_modifier(
#         modified_contract,
#         "Change the payment terms to include milestone-based payments: 40% upfront, 30% at beta release, and 30% upon final delivery"
#     )
    
#     print("Adding warranty section...")
#     modified_contract = interactive_contract_modifier(
#         modified_contract,
#         "Add a warranty section where the provider warrants the delivered applications will be free from defects for 90 days after delivery"
#     )
    
#     print("\nFinal Modified Contract (after interactive modifications):")
#     print("=" * 70)
#     print(modified_contract)
#     print("=" * 70)
    
#     return modified_contract

# Uncomment the line below to test the complete workflow
# complete_contract_workflow_example()