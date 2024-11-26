import os
from groq import Groq
from typing import List
import json
import ast
import fitz
class PrescriptionParser:
    def __init__(self):
        """Initialize the parser with Groq API key."""
        self.api_key = "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"
        if not self.api_key:
            raise Exception("GROQ_API_KEY environment variable not set.")
        self.client = Groq(api_key=self.api_key)
        
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file using PyMuPDF.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Extracted text from the PDF.
        """
        try:
            # Open the PDF file
            pdf_document = fitz.open(pdf_path)
            text = ""

            # Iterate through each page
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text += page.get_text()  # Extract text from the page
            pdf_document.close()  # Close the PDF file
            return text

        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def parse_medication_details(self, text: str) -> List[List]:
        print("""Use Groq AI to extract medication details from text.""")
        prompt = f"""
        Extract medication details from the following prescription text. For each medication, provide:
        1. Medicine name (as a string)
        2. Price per item (as a number)
        3. Quantity (as a number)
        4. Total Price(as a number)

        Format the response as a list of lists, where each inner list contains exactly these 3 elements:
        [["Medicine Name", price_per_item, quantity, total_price], ...]

        For example:
        [["Aspirin", 5.99, 30, 179.70], ...........]

        Prescription text:
        {text}

        Return only the Python list format, nothing else. Ensure all medicine names are in quotes.
        """

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="mixtral-8x7b-32768",
                temperature=0.1
            )
            
            # Get the response content
            content = response.choices[0].message.content.strip()
            
            # Clean the response to ensure proper Python list format
            content = content.replace("'", "")
            
            try:
                # First try parsing as JSON
                result = json.loads(content)
            except json.JSONDecodeError:
                try:
                    # If JSON fails, try using ast.literal_eval
                    result = ast.literal_eval(content)
                except:
                    raise Exception("Failed to parse Groq AI response")
            
            # Validate the response format
            if not isinstance(result, list) or not all(isinstance(item, list) and len(item) == 4 for item in result):
                raise Exception("Invalid response format from Groq AI")
            return result
        
        except Exception as e:
            raise Exception(f"Error processing with Groq AI: {str(e)}")

def call(pdf1):
    try:
        # Initialize parser
        parser = PrescriptionParser()
        # Extract text from PDF
        print("Extracting text from PDF...")
        text1 = parser.extract_text_from_pdf(pdf1)
        # Parse medication details
        print("Processing with Groq AI...")
        medication_details = parser.parse_medication_details(text1)
        return medication_details
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("An unexpected error occurred. Please check your inputs and try again.")