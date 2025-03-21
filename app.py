import streamlit as st
import os
import tempfile
from pathlib import Path
import base64
import io
import docx2txt
import PyPDF2

def main():
    st.title("Paper Upload System")
    st.write("Upload your paper document to proceed to the next stage.")
    
    # Set up the session state to track if a file has been uploaded
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = "upload"
    if 'file_content' not in st.session_state:
        st.session_state.file_content = None
    if 'file_type' not in st.session_state:
        st.session_state.file_type = None
    if 'file_name' not in st.session_state:
        st.session_state.file_name = None
    
    # Display different content based on the current stage
    if st.session_state.current_stage == "upload":
        upload_stage()
    elif st.session_state.current_stage == "confirmation":
        confirmation_stage()
    elif st.session_state.current_stage == "next_stage":
        next_stage()

def upload_stage():
    st.subheader("Stage 1: Upload Your Paper")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"], help="Please upload a PDF, Word document, or text file.")
    
    if uploaded_file is not None:
        # Display file details
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "File type": uploaded_file.type
        }
        
        st.write("File Details:")
        for key, value in file_details.items():
            st.write(f"- {key}: {value}")
        
        # Save the uploaded file to a temporary location
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir) / uploaded_file.name
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.uploaded_file_path = str(temp_path)
        st.session_state.file_uploaded = True
        st.session_state.file_type = uploaded_file.type
        st.session_state.file_name = uploaded_file.name
        
        # Preview section based on file type
        st.subheader("Document Preview:")
        
        if uploaded_file.type == "application/pdf":
            # PDF Preview
            preview_pdf(uploaded_file)
            
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Word document preview
            text = docx2txt.process(uploaded_file)
            st.session_state.file_content = text
            st.text_area("Document Content (Preview)", text, height=300, disabled=True)
            
        elif uploaded_file.type == "text/plain":
            # Display a preview of the text file
            text_content = uploaded_file.getvalue().decode("utf-8")
            st.session_state.file_content = text_content
            st.text_area("Content", text_content, height=300, disabled=True)
    
    # Submit button
    if st.button("Submit Paper", disabled=not st.session_state.file_uploaded):
        st.session_state.current_stage = "confirmation"
        st.experimental_rerun()
    
    # Warning message if no file is uploaded
    if not st.session_state.file_uploaded:
        st.warning("⚠️ You must upload a paper to proceed to the next stage.")

def preview_pdf(pdf_file):
    # Read PDF file
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    
    # Display number of pages
    st.write(f"PDF document with {num_pages} pages")
    
    # Extract and display text from the first few pages
    all_text = ""
    max_pages_to_show = min(3, num_pages)  # Limit to first 3 pages for preview
    
    for i in range(max_pages_to_show):
        page = pdf_reader.pages[i]
        page_text = page.extract_text()
        all_text += f"\n--- Page {i+1} ---\n{page_text}\n"
    
    st.session_state.file_content = all_text
    st.text_area("PDF Content (First 3 pages preview)", all_text, height=300, disabled=True)
    
    # Embed PDF for direct viewing
    base64_pdf = base64_encode_pdf(pdf_file)
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def base64_encode_pdf(pdf_file):
    pdf_file.seek(0)
    pdf_bytes = pdf_file.read()
    base64_encoded = base64.b64encode(pdf_bytes).decode('utf-8')
    return base64_encoded

def confirmation_stage():
    st.subheader("Stage 2: Confirm Your Submission")
    
    # Display confirmation information
    st.success("✅ File uploaded successfully!")
    st.write(f"You've uploaded: {st.session_state.file_name}")
    
    # Show preview again
    st.subheader("Document Preview:")
    
    if st.session_state.file_type == "application/pdf":
        # For PDFs, offer to view again
        if st.button("View PDF Content Again"):
            with open(st.session_state.uploaded_file_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        
    else:
        # For text and Word docs, show the content again
        st.text_area("Document Content", st.session_state.file_content, height=200, disabled=True)
    
    # Options to go back or proceed
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Go Back and Upload Again"):
            st.session_state.current_stage = "upload"
            st.experimental_rerun()
    
    with col2:
        if st.button("Proceed to Next Stage →"):
            st.session_state.current_stage = "next_stage"
            st.experimental_rerun()

def next_stage():
    st.subheader("Stage 3: Paper Processing")
    st.write("Your paper has been successfully uploaded and is ready for processing.")
    
    # Display a success message
    st.success("🎉 Congratulations! Your paper has been accepted.")
    
    # Display file information again
    st.write(f"**File name:** {st.session_state.file_name}")
    
    # Here you would typically add the functionality for the next stage
    st.write("In this stage, you could implement features such as:")
    st.write("- Paper analysis")
    st.write("- Metadata extraction")
    st.write("- Peer review assignment")
    st.write("- Additional information collection")
    
    # Option to start over
    if st.button("Start Over"):
        st.session_state.file_uploaded = False
        st.session_state.uploaded_file_path = None
        st.session_state.file_content = None
        st.session_state.file_type = None
        st.session_state.file_name = None
        st.session_state.current_stage = "upload"
        st.experimental_rerun()

if __name__ == "__main__":
    main()