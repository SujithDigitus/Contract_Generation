import os
import json
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse # Added HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any

# Ensure Contract_Generation.py is in the same directory or accessible via PYTHONPATH
# Updated import list
from Contract_Generation import (
    rag_pipeline_with_prompt, 
    get_text_from_Pdf, 
    clean_text_for_llm,
    format_text_to_html_with_llm # NEWLY ADDED FUNCTION
    # fill_placeholders_with_llm # Add this if you implement the optional LLM data filler
)


app = FastAPI(title="Contract Generation API")

# Ensure necessary directories exist
os.makedirs("./original_contracts", exist_ok=True)
os.makedirs("./Contract_templates", exist_ok=True)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000", # Corrected the link format
    # "http://your-frontend-domain.com", # Replace with your actual frontend domain
    "*"  # Allow all origins (consider restricting in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FilenameData(BaseModel):
    filename: str # Expects the original PDF filename, e.g., "my_contract.pdf"

class UserInputData(BaseModel):
    filename: str # Expects the original PDF filename, e.g., "my_contract.pdf"
    Placeholders: str # A JSON string representing user-filled placeholder values.

# --- NEW REQUEST MODEL FOR STYLED HTML ENDPOINT ---
class GenerateStyledHTMLRequest(BaseModel):
    filename: str # Original PDF filename to fetch the template JSON
    user_placeholders: str = "{}" # JSON string of placeholder values from user, defaults to empty JSON object
    style_instructions: str = "Use generic professional styling" # Optional: textual description of desired HTML style

# --- Endpoint to process PDF to template JSON (existing) ---
@app.post("/contract_template/")
async def process_document_to_template(file: UploadFile = File(...)):
    original_filename = file.filename
    file_location = f"./original_contracts/{original_filename}"
    
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    finally:
        file.file.close()

    try:
        raw_context = get_text_from_Pdf(file_location)
        if not raw_context or not raw_context.strip():
            raise HTTPException(status_code=400, detail="Extracted text from PDF is empty or could not be read. Cannot process.")
            
        context = clean_text_for_llm(raw_context)
        
        if not context.strip():
            raise HTTPException(status_code=400, detail="Text became empty after cleaning. Original PDF might contain only control characters or unsupported content.")

        llm_output_str = rag_pipeline_with_prompt(context)
        
        try:
            json_data = json.loads(llm_output_str) 
        except json.JSONDecodeError as e:
            error_msg = f"LLM output was not valid JSON. Error: {str(e)}"
            context_window = 40
            start_index = max(0, e.pos - context_window)
            end_index = min(len(llm_output_str), e.pos + context_window)
            problematic_snippet = llm_output_str[start_index:end_index]
            
            def repr_non_printable(s):
                return "".join(repr(c)[1:-1] if len(repr(c)) > 3 or ord(c) < 32 or ord(c) == 127 else c for c in s)

            detailed_error_msg = (
                f"{error_msg}\n"
                f"Error near character {e.pos} (line {e.lineno}, column {e.colno}).\n"
                f"Snippet (raw): '{problematic_snippet}'\n"
                f"Snippet (with non-printables escaped): '{repr_non_printable(problematic_snippet)}'"
            )
            print(f"--- DETAILED JSON PARSE ERROR ---\n{detailed_error_msg}\n--- END DETAILED JSON PARSE ERROR ---") 

            error_output_path = f"./Contract_templates/{original_filename}.error.txt"
            with open(error_output_path, 'w', encoding='utf-8') as f_err:
                f_err.write(f"--- Detailed Error ---\n{detailed_error_msg}\n\n--- Full LLM Output ---\n{llm_output_str}")
            raise HTTPException(status_code=500, detail=f"{error_msg}. Problematic snippet logged. Check server logs and '{original_filename}.error.txt'.")

        output_json_path = f"./Contract_templates/{original_filename}.json"
        with open(output_json_path, 'w', encoding='utf-8') as f_json:
            json.dump(json_data, f_json, indent=2)
            
        return {"summary": f"Contract '{original_filename}' processed. Template saved to '{original_filename}.json'."}

    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        # Log the full traceback for unexpected errors
        import traceback
        print(f"--- UNEXPECTED ERROR in /contract_template/ for {original_filename} ---")
        traceback.print_exc()
        print(f"--- END UNEXPECTED ERROR ---")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing the document: {str(e)}")

# --- Endpoint to get placeholders (existing) ---
@app.post("/template_placeholders/")
async def get_template_placeholders(data: FilenameData):
    json_filename = f"{data.filename}.json"
    path = f"./Contract_templates/{json_filename}"

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Template file '{json_filename}' not found.")

    try:
        with open(path, 'r', encoding='utf-8') as file:
            template_data = json.load(file) 
        
        placeholders_info = template_data.get('Placeholders', {})
        if not isinstance(placeholders_info, dict):
            raise HTTPException(status_code=500, detail=f"Invalid format for 'Placeholders' in template file '{json_filename}'. Expected a dictionary.")

        return {"Placeholders_Info": placeholders_info}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Invalid JSON format in template file '{json_filename}'.")
    except Exception as e:
        print(f"Error reading template placeholders for {json_filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve placeholder information: {str(e)}")

# --- Endpoint to generate filled plain text contract (existing) ---
@app.post("/generate_contract/")
async def generate_filled_contract(data: UserInputData):
    json_filename = f"{data.filename}.json"
    path = f"./Contract_templates/{json_filename}"

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Template file '{json_filename}' not found.")

    try:
        with open(path, 'r', encoding='utf-8') as file:
            template_data = json.load(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read or parse template file '{json_filename}': {str(e)}")

    template_text = template_data.get('Template')
    extracted_placeholders_details = template_data.get('Placeholders') 

    if not template_text or not isinstance(extracted_placeholders_details, dict):
        raise HTTPException(status_code=500, detail=f"Template file '{json_filename}' is missing 'Template' or 'Placeholders' data, or 'Placeholders' is not a dictionary.")

    try:
        user_provided_values: Dict[str, str] = json.loads(data.Placeholders) if data.Placeholders and data.Placeholders.strip() else {}
        if not isinstance(user_provided_values, dict):
            # This case should ideally not be reached if JSON parsing is strict,
            # but as a safeguard:
            raise ValueError("Placeholders input must be a JSON object string that deserializes to a dictionary.")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON string format in 'Placeholders' field: {str(e)}")
    except ValueError as e: # Catch our custom ValueError
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e: # Catch other unexpected errors
        raise HTTPException(status_code=400, detail=f"Error processing 'Placeholders' field: {str(e)}")


    final_text = template_text
    # Sort keys by length in reverse to avoid replacing substrings of longer keys
    # e.g., "Party_Name" before "Party_Name_Address"
    sorted_placeholder_keys = sorted(extracted_placeholders_details.keys(), key=len, reverse=True)

    for placeholder_key in sorted_placeholder_keys:
        details_dict = extracted_placeholders_details.get(placeholder_key) # Use .get for safety
        
        replacement_value = None # Initialize
        user_value = user_provided_values.get(placeholder_key)

        if user_value is not None: # User provided a value (even if it's an empty string)
            replacement_value = str(user_value)
        elif isinstance(details_dict, dict) and details_dict.get('original_value') is not None:
            replacement_value = str(details_dict.get('original_value'))
        else: 
            # If no user value and no original_value, use the placeholder key itself or an empty string
            replacement_value = placeholder_key # Or "" if you prefer empty
            print(f"Warning: No user value or original_value for placeholder '{placeholder_key}' in template '{json_filename}'. Using placeholder name as fallback.")
        
        # Ensure the placeholder_key is treated as a literal string for replacement
        # This is important if placeholder_key contains special regex characters,
        # though our generation logic for keys (letters, numbers, underscores) makes this less likely.
        final_text = final_text.replace(placeholder_key, replacement_value)


    return {"Generated_Contract": final_text}

# --- NEW ENDPOINT TO GENERATE STYLED HTML DOCUMENT ---
@app.post("/generate_styled_html_document/", response_class=HTMLResponse)
async def generate_styled_html_document_endpoint(data: GenerateStyledHTMLRequest):
    json_filename = f"{data.filename}.json" # Based on original PDF filename
    template_json_path = f"./Contract_templates/{json_filename}"

    if not os.path.exists(template_json_path):
        raise HTTPException(status_code=404, detail=f"Template JSON file '{json_filename}' not found.")

    try:
        with open(template_json_path, 'r', encoding='utf-8') as file:
            template_data_from_json = json.load(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read or parse template JSON file '{json_filename}': {str(e)}")

    plain_text_template_from_json = template_data_from_json.get('Template')
    placeholders_details_from_json = template_data_from_json.get('Placeholders')

    if not plain_text_template_from_json or not isinstance(placeholders_details_from_json, dict):
        raise HTTPException(status_code=500, detail=f"Template JSON file '{json_filename}' is missing 'Template' or 'Placeholders' data, or 'Placeholders' is not a dictionary.")

    try:
        # Ensure user_placeholders is treated as a JSON string representing a dictionary
        user_values_for_filling: Dict[str, Any] = json.loads(data.user_placeholders) if data.user_placeholders and data.user_placeholders.strip() else {}
        if not isinstance(user_values_for_filling, dict):
            # This indicates that the JSON string did not parse into a dictionary
            user_values_for_filling = {} # Default to empty dict
            # Optionally, raise an error if strict parsing to dict is required:
            # raise ValueError("User placeholders must be a JSON object string that deserializes to a dictionary.")
    except json.JSONDecodeError:
        if data.user_placeholders and data.user_placeholders.strip() and data.user_placeholders.strip() != '{}':
             raise HTTPException(status_code=400, detail="Invalid JSON string format for user_placeholders.")
        user_values_for_filling = {} # Default to empty dict if not provided or invalid non-empty string
    except Exception as e: 
        raise HTTPException(status_code=400, detail=f"Error processing user_placeholders: {str(e)}")


    # --- Logic to fill placeholders ---
    filled_plain_text = plain_text_template_from_json
    
    all_placeholder_keys = sorted(placeholders_details_from_json.keys(), key=len, reverse=True)

    for placeholder_key in all_placeholder_keys:
        details_dict = placeholders_details_from_json.get(placeholder_key) # Use .get for safety
        
        replacement_value_str = "" # Default to empty string if no value found
        user_value = user_values_for_filling.get(placeholder_key)

        if user_value is not None: # User provided a value (even if it's an empty string)
            replacement_value_str = str(user_value)
        elif isinstance(details_dict, dict) and details_dict.get('original_value') is not None:
            replacement_value_str = str(details_dict.get('original_value'))
        else:
            replacement_value_str = "" # Or placeholder_key if you prefer to see the key
            print(f"Info: No user value or original_value for '{placeholder_key}'. Using empty string as fallback for HTML generation.")
        
        filled_plain_text = filled_plain_text.replace(placeholder_key, replacement_value_str)
    # --- End of placeholder filling logic ---

    # --- Call LLM to format the filled plain text to HTML ---
    try:
        styled_html_output = format_text_to_html_with_llm(
            plain_text_content=filled_plain_text,
            style_instructions=data.style_instructions # CORRECTED KEYWORD
        )
    except Exception as e:
        print(f"Error during HTML formatting by LLM: {str(e)}")
        import traceback
        traceback.print_exc() # For more detailed server logs during dev
        raise HTTPException(status_code=500, detail=f"Failed to format text to HTML using LLM: {str(e)}")

    if not styled_html_output:
        raise HTTPException(status_code=500, detail="LLM returned empty output for HTML formatting.")
        
    if not styled_html_output.strip().lower().startswith("<!doctype html>") and \
       not styled_html_output.strip().lower().startswith("<html>"):
        print(f"Warning: LLM output does not appear to be a full HTML document. Preview: {styled_html_output[:200]}")

    return HTMLResponse(content=styled_html_output)


# To run: uvicorn main:app --reload
# Example for testing /generate_styled_html_document/ with curl:
# curl -X POST "http://127.0.0.1:8000/generate_styled_html_document/" \
# -H "Content-Type: application/json" \
# -d '{"filename": "your_contract.pdf", "user_placeholders": "{\"Placeholder_Name_1\": \"User Value 1\", \"Date_Placeholder\": \"June 1, 2025\"}", "style_instructions": "Make it look very formal with blue headings."}' \
# -o output.html
# (Ensure your_contract.pdf.json exists in ./Contract_templates/)
