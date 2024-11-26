from flask import Flask, render_template, request, redirect, send_file
from werkzeug.utils import secure_filename
import os
from geni import call
import pandas as pd
from twotest import admi
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib import utils
from textwrap import wrap
from io import BytesIO
import pandas as pd

SUSPICIOUS_THRESHOLD = 30
THRESHOLD = 30
file_path = r"compressed_data.csv"
df = pd.read_csv(file_path, low_memory=False)
df['name'] = df['name'].str.strip().str.lower()  # Normalize dataset names

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder to save uploaded files
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'doc', 'docx','jpg'}  # Allowed file types

medicines = []
admissions = ""
prob = 71

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'document' not in request.files or 'admission' not in request.files:
        print("No document part in request files")
        return redirect(request.url)
    
    file = request.files['document']
    file1 = request.files['admission']
    if file.filename == '' or file1.filename == '':
        print("No selected file")
        return redirect(request.url)

    if file and file1 and allowed_file(file.filename) and allowed_file(file1.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        filename1 = secure_filename(file1.filename)
        file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
        print(f"Files {filename} and {filename1} uploaded successfully.")
        global medicines
        global admissions
        global prob
        medicines = call('uploads/'+filename)
        adm = admi('uploads/'+filename1)
        admissions = adm[0]
        prob = adm[1]*100
        print("*"*100)
        print(adm[0])
        print("*"*100)
        print(adm[1])
        return render_template('report_index.html')

    print("File not allowed")
    return redirect(request.url)

def find_cheapest_substitute(medicine_name,oc):
    # Normalize input medicine name
    medicine_name = medicine_name.strip().lower()
    # Check if the medicine exists in the dataset
    if medicine_name not in df['name'].values:
        return oc

    # Get the row corresponding to the input medicine
    medicine_row = df[df['name'] == medicine_name].iloc[0]
    
    # Gather substitutes and their costs
    substitutes = []
    for i in range(5):  # Up to 5 substitutes
        substitute_name = medicine_row.get(f'substitute{i}', None)
        if pd.notna(substitute_name):
            substitute_name = substitute_name.strip().lower()
            substitute_row = df[df['name'] == substitute_name]
            if not substitute_row.empty:
                substitute_cost = substitute_row['Cost'].values[0]
                substitutes.append((substitute_name, substitute_cost))
    
    # Find the cheapest substitute
    if not substitutes :
        return oc

    cheapest_substitute = min(substitutes, key=lambda x: x[1])
    return min(oc,cheapest_substitute[1])

@app.route('/report', methods=['GET'])
def report():
    try:
        # Generate PDF Report
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)

        # Title
        c.setFont("Helvetica-Bold", 24)
        c.setFillColorRGB(0.0, 0.48, 0.51)  # Dark blue from theme (#007B83)
        c.drawString(200, 750, "Medicine Report")

        # Draw a Decorative Line Below Title
        c.setStrokeColorRGB(0.65, 0.82, 0.88)  # Light blue from theme (#A7D0CD)
        c.setLineWidth(2)
        c.line(50, 745, 550, 745)

        # Table Header with Styling
        c.setFillColorRGB(0.65, 0.82, 0.88)  # Light blue background for header
        c.rect(50, 700, 500, 25, fill=1, stroke=0)  # Header background
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(colors.white)
        c.drawString(60, 710, "Medicine Name")
        c.drawString(200, 710, "Original Rate")
        c.drawString(300, 710, "Cheaper Price")
        c.drawString(400, 710, "Quantity")

        # Table Content with Decoration
        y = 690
        c.setFont("Helvetica", 10)

        total_original_amount = 0
        total_cheaper_amount = 0
        i = 0
        for med in medicines:
            # Calculate the amounts
            original_rate = float(med[1])
            cheaper_rate = float(find_cheapest_substitute(med[0], float(med[1])))
            quantity = int(med[2])
            original_amount = original_rate * quantity
            cheaper_amount = cheaper_rate * quantity

            # Append to total amounts
            total_original_amount += original_amount
            total_cheaper_amount += cheaper_amount

            # Alternating row colors for better readability (light gray or white)
            row_color = (0.97, 0.97, 0.97) if i % 2 == 0 else (1, 1, 1)  # Light gray or white
            c.setFillColorRGB(*row_color)
            c.rect(50, y - 15, 500, 20, fill=1, stroke=0)  # Draw row background

            # Draw text in each column with proper column widths
            c.setFillColor(colors.black)
            c.drawString(60, y-7.5, med[0][:30])  # Display only part of the medicine name to prevent overflow
            c.drawString(200, y-7.5, f"{original_rate}")
            c.drawString(300, y-7.5, f"{cheaper_rate}")
            c.drawString(400, y-7.5, str(quantity))
            y -= 20
            i += 1

            # Create a new page if necessary
            if y < 50:
                c.showPage()
                y = 750
                # Redraw the header on the new page
                c.setFillColorRGB(0.65, 0.82, 0.88)  # Light blue background
                c.rect(50, 700, 500, 25, fill=1, stroke=0)
                c.setFont("Helvetica-Bold", 12)
                c.setFillColor(colors.white)
                c.drawString(60, 710, "Medicine Name")
                c.drawString(200, 710, "Rate")
                c.drawString(300, 710, "Cheaper Substitue Rate")
                c.drawString(400, 710, "Quantity")

        # Display totals and percentage increase
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(colors.black)
        y -= 30
        c.drawString(50, y, f"Total Original Amount: {total_original_amount:.2f}")
        c.drawString(50, y - 20, f"Total Cheaper Amount: {total_cheaper_amount:.2f}")

        if total_cheaper_amount != 0:
            percentage_increase = ((total_original_amount - total_cheaper_amount) / total_cheaper_amount) * 100
        else:
            percentage_increase = 0
        c.drawString(50, y - 40, f"Percentage Increase in Bill: {percentage_increase:.2f}%")

        # Check if the bill is suspicious
        if percentage_increase > SUSPICIOUS_THRESHOLD:
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(colors.red)
            c.drawString(50, y - 70, "!!! Suspicious Bill Detected !!!")
            c.setFont("Helvetica", 10)
            c.setFillColor(colors.black)
            c.drawString(50, y - 120, f"The percentage increase in the bill is {percentage_increase:.2f}%, which is greater than the threshold of {SUSPICIOUS_THRESHOLD}%.")
            c.drawString(50, y - 140, "This could indicate that the medicines are being sold at inflated prices.")
            c.drawString(50, y - 160, "It is advised to double-check the prices to ensure the bill is correct.")

        def draw_wrapped_text(canvas, text, x, y, max_width, line_height):
            lines = wrap(text, width=max_width // 6)  # Approximation for character width in points
            for line in lines:
                canvas.drawString(x, y, line)
                y -= line_height
            return y

        c.showPage()

        # Admission Sheet Report (Page 2)
        c.setFont("Helvetica-Bold", 24)
        c.setFillColorRGB(0.0, 0.48, 0.51)  # Dark blue from theme
        c.drawString(180, 750, "Admission Sheet Report")

        # Draw a Decorative Line Below Title
        c.setStrokeColorRGB(0.65, 0.82, 0.88)
        c.setLineWidth(2)
        c.line(50, 740, 550, 740)

        # Add paragraph text with wrapping
        c.setFont("Helvetica", 12)
        c.setFillColor(colors.black)
        y = draw_wrapped_text(c, admissions, x=50, y=700, max_width=450, line_height=15)

        if prob >= THRESHOLD:
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(colors.red)
            c.drawString(50, y - 70, "!!! Suspicious Bill Detected !!!")
            c.setFont("Helvetica", 10)
            c.setFillColor(colors.black)
            c.drawString(50, y - 120, f"The probability of the admission sheet to be fake is {prob:.2f}%, which is greater than the threshold of {THRESHOLD}%.")
            c.drawString(50, y - 140, "This could indicate that the patient is admitted unecessarily.")
            c.drawString(50, y - 160, "It is advised to double-check the details.")

        # Save the PDF to the buffer
        c.save()
        pdf_buffer.seek(0)

        # Send the generated PDF as a downloadable file
        return send_file(pdf_buffer, as_attachment=True, download_name="medicine_report.pdf", mimetype="application/pdf")
    except Exception as e:
        return f"Error: {e}"
    


if __name__ == '__main__':
    app.run(debug=True)