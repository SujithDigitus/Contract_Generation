import os
import json
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import tempfile
import uuid
from pathlib import Path

# Import functions from contract_compare.py
from contract_compare import (
    extract_text_from_pdf,
    compare_contracts_with_llm,
    generate_html_report
)

app = FastAPI(title="Contract Comparison API", description="API for comparing multiple PDF contracts")

# Ensure necessary directories exist
os.makedirs("./uploaded_contracts", exist_ok=True)
os.makedirs("./comparison_reports", exist_ok=True)

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "*"  # Allow all origins (consider restricting in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ComparisonJobStatus(BaseModel):
    job_id: str
    status: str  # "processing", "completed", "failed"
    message: str
    report_url: str = None
    contracts_processed: int = 0
    total_contracts: int = 0

class ContractComparisonResponse(BaseModel):
    job_id: str
    status: str
    message: str
    report_url: str = None
    comparison_data: List[Dict[str, Any]] = None

# In-memory storage for job status (in production, use Redis or database)
job_storage: Dict[str, Dict] = {}

@app.get("/")
async def root():
    return {
        "message": "Contract Comparison API",
        "version": "1.0.0",
        "endpoints": {
            "compare_contracts": "/compare-contracts/",
            "job_status": "/job-status/{job_id}",
            "download_report": "/download-report/{job_id}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "contract-comparison-api"}

@app.post("/compare-contracts/", response_model=ContractComparisonResponse)
async def compare_contracts(
    files: List[UploadFile] = File(..., description="PDF contract files to compare (2-10 files)"),
    return_html: bool = Form(default=True, description="Whether to return HTML report"),
    return_json: bool = Form(default=False, description="Whether to return JSON comparison data")
):
    """
    Compare multiple PDF contracts and generate a comparison report.
    
    Args:
        files: List of PDF files to compare (minimum 2, maximum 10)
        return_html: Whether to generate and return HTML report
        return_json: Whether to return JSON comparison data
    
    Returns:
        ComparisonJobStatus with job details and report URL
    """
    
    # Validate number of files
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="At least 2 PDF files are required for comparison.")
    elif len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 PDF files are supported for comparison.")
    
    # Validate file types
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF file.")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    job_storage[job_id] = {
        "status": "processing",
        "message": "Starting contract comparison...",
        "total_contracts": len(files),
        "contracts_processed": 0,
        "contract_names": [file.filename for file in files],
        "comparison_data": None,
        "report_path": None
    }
    
    try:
        # Save uploaded files temporarily
        temp_dir = tempfile.mkdtemp()
        saved_files = []
        
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            try:
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                saved_files.append(file_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to save file {file.filename}: {str(e)}")
        
        # Update job status
        job_storage[job_id]["message"] = "Files uploaded, extracting text..."
        
        # Extract text from PDFs
        contract_texts = []
        contract_labels = []
        failed_extractions = []
        
        for i, file_path in enumerate(saved_files):
            try:
                contract_text = extract_text_from_pdf(file_path)
                if contract_text and contract_text.strip():
                    contract_texts.append(contract_text)
                    contract_labels.append(chr(65 + len(contract_texts) - 1))  # A, B, C, etc.
                    job_storage[job_id]["contracts_processed"] = len(contract_texts)
                else:
                    failed_extractions.append(f"Contract {chr(65 + i)} ({files[i].filename})")
            except Exception as e:
                failed_extractions.append(f"Contract {chr(65 + i)} ({files[i].filename}): {str(e)}")
        
        if len(contract_texts) < 2:
            error_msg = f"Could not extract text from enough contracts. Failed: {', '.join(failed_extractions)}"
            job_storage[job_id]["status"] = "failed"
            job_storage[job_id]["message"] = error_msg
            raise HTTPException(status_code=400, detail=error_msg)
        
        if failed_extractions:
            job_storage[job_id]["message"] = f"Text extraction completed with warnings: {', '.join(failed_extractions)}"
        else:
            job_storage[job_id]["message"] = "Text extraction completed successfully"
        
        # Compare contracts using LLM
        job_storage[job_id]["message"] = "Comparing contracts using AI..."
        
        try:
            comparison_results = compare_contracts_with_llm(contract_texts, contract_labels)
            job_storage[job_id]["comparison_data"] = comparison_results
        except Exception as e:
            error_msg = f"Failed to compare contracts: {str(e)}"
            job_storage[job_id]["status"] = "failed"
            job_storage[job_id]["message"] = error_msg
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Generate HTML report if requested
        report_path = None
        if return_html:
            job_storage[job_id]["message"] = "Generating HTML report..."
            report_filename = f"comparison_report_{job_id}.html"
            report_path = os.path.join("./comparison_reports", report_filename)
            
            try:
                generate_html_report(comparison_results, contract_labels, report_path)
                job_storage[job_id]["report_path"] = report_path
            except Exception as e:
                error_msg = f"Failed to generate HTML report: {str(e)}"
                job_storage[job_id]["status"] = "failed"
                job_storage[job_id]["message"] = error_msg
                raise HTTPException(status_code=500, detail=error_msg)
        
        # Update job status to completed
        job_storage[job_id]["status"] = "completed"
        job_storage[job_id]["message"] = "Contract comparison completed successfully"
        
        # Clean up temporary files
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory: {e}")
        
        # Prepare response
        response_data = {
            "job_id": job_id,
            "status": "completed",
            "message": job_storage[job_id]["message"]
        }
        
        if return_html and report_path:
            response_data["report_url"] = f"/download-report/{job_id}"
        
        if return_json and comparison_results:
            response_data["comparison_data"] = comparison_results
        
        return ComparisonJobStatus(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        # Handle unexpected errors
        job_storage[job_id]["status"] = "failed"
        job_storage[job_id]["message"] = f"Unexpected error: {str(e)}"
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/job-status/{job_id}", response_model=ComparisonJobStatus)
async def get_job_status(job_id: str):
    """Get the status of a comparison job"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = job_storage[job_id]
    response_data = {
        "job_id": job_id,
        "status": job_data["status"],
        "message": job_data["message"],
        "contracts_processed": job_data.get("contracts_processed", 0),
        "total_contracts": job_data.get("total_contracts", 0)
    }
    
    if job_data["status"] == "completed" and job_data.get("report_path"):
        response_data["report_url"] = f"/download-report/{job_id}"
    
    return ComparisonJobStatus(**response_data)

@app.get("/download-report/{job_id}")
async def download_report(job_id: str):
    """Download the HTML comparison report for a job"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = job_storage[job_id]
    
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    report_path = job_data.get("report_path")
    if not report_path or not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report file not found")
    
    return FileResponse(
        path=report_path,
        media_type="text/html",
        filename=f"contract_comparison_report_{job_id}.html"
    )

@app.get("/view-report/{job_id}", response_class=HTMLResponse)
async def view_report(job_id: str):
    """View the HTML comparison report directly in the browser"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = job_storage[job_id]
    
    if job_data["status"] != "completed":
        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head><title>Report Not Ready</title></head>
        <body>
            <h1>Report Not Ready</h1>
            <p>Job Status: {job_data['status']}</p>
            <p>Message: {job_data['message']}</p>
            <p><a href="/job-status/{job_id}">Check Job Status</a></p>
        </body>
        </html>
        """)
    
    report_path = job_data.get("report_path")
    if not report_path or not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report file not found")
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read report file: {str(e)}")

@app.get("/comparison-data/{job_id}")
async def get_comparison_data(job_id: str):
    """Get the raw JSON comparison data for a job"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = job_storage[job_id]
    
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    comparison_data = job_data.get("comparison_data")
    if comparison_data is None:
        raise HTTPException(status_code=404, detail="Comparison data not found")
    
    return {
        "job_id": job_id,
        "contract_labels": [chr(65 + i) for i in range(job_data["total_contracts"])],
        "contract_names": job_data.get("contract_names", []),
        "comparison_data": comparison_data,
        "total_differences": len(comparison_data) if comparison_data else 0
    }

@app.delete("/cleanup-job/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up job data and associated files"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = job_storage[job_id]
    
    # Remove report file if it exists
    report_path = job_data.get("report_path")
    if report_path and os.path.exists(report_path):
        try:
            os.remove(report_path)
        except Exception as e:
            print(f"Warning: Failed to remove report file {report_path}: {e}")
    
    # Remove job from storage
    del job_storage[job_id]
    
    return {"message": f"Job {job_id} cleaned up successfully"}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Example usage:
# 1. Start the server: uvicorn compare_main:app --reload
# 2. Compare contracts:
#    curl -X POST "http://localhost:8000/compare-contracts/" \
#    -F "files=@contract1.pdf" \
#    -F "files=@contract2.pdf" \
#    -F "files=@contract3.pdf" \
#    -F "return_html=true" \
#    -F "return_json=false"
# 3. Check job status:
#    curl "http://localhost:8000/job-status/{job_id}"
# 4. Download report:
#    curl "http://localhost:8000/download-report/{job_id}" -o report.html
# 5. View report in browser:
#    http://localhost:8000/view-report/{job_id}