import re
from collections import defaultdict

""" # Path to the OCR text file
path = "D:/FlipKart GRiD 6.0 Robotic Track/prototype/project/ocr/ocr.txt"

# Open and read the OCR text file
with open(path, "r") as file:
    ocr_text = file.read() """



def nlpresult(ocr_text):
    print(ocr_text)
    # Regular expressions for different fields
    date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'  # To find dates
    price_pattern = r'Rs\.\d+(\.\d{2})?'  # To find price like Rs.0.15 per g
    batch_pattern = r'Batch No\.:?\s?([A-Za-z0-9]+)'  # To find batch number
    net_qty_pattern = r'NetQty\.:?\s?(\d+ml|\d+g)'  # To find net quantity like 275ml or 100g
    ingredients_pattern = r'INGREDIENTS?:\s?([A-Za-z0-9\s%,\.\-]+)'  # To find ingredients
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'  # Email regex
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'  # Phone numbers
    manufacturer_pattern = r'MKT[D|D]\s*BY:\s?([A-Za-z0-9\s\.\-]+)'  # To find manufacturer

    # Helper function to extract information
    def extract_info(text, pattern):
        matches = re.findall(pattern, text)
        return matches[0] if matches else None

    # Extract information using regex
    extracted_data = defaultdict(str)

    # Product Name (you can improve this part based on how you want to identify the product)
    extracted_data['Product Name'] = "All New Denver HAM"  # Static example

    # Extract ingredients
    ingredients = extract_info(ocr_text, ingredients_pattern)
    extracted_data['Ingredients'] = ingredients

    # Extract dates (Manufacturing Date and Expiry Date)
    dates = re.findall(date_pattern, ocr_text)
    if len(dates) >= 2:
        extracted_data['Manufacturing Date'] = dates[0]
        extracted_data['Expiry Date'] = dates[1]

    # Extract Batch Number
    batch_number = extract_info(ocr_text, batch_pattern)
    extracted_data['Batch Number'] = batch_number

    # Extract Price (if applicable)
    price = extract_info(ocr_text, price_pattern)
    extracted_data['Price'] = price

    # Extract Net Quantity
    net_qty = extract_info(ocr_text, net_qty_pattern)
    extracted_data['Net Quantity'] = net_qty

    # Extract Manufacturer and Address
    manufacturer = extract_info(ocr_text, manufacturer_pattern)
    extracted_data['Manufacturer'] = manufacturer

    # Extract Email
    email = extract_info(ocr_text, email_pattern)
    extracted_data['Email'] = email

    # Extract Phone
    phone = extract_info(ocr_text, phone_pattern)
    extracted_data['Phone'] = phone

    print(extracted_data)
    return extracted_data


""" # Writing the extracted data into a new file
file2="product_classifier.txt"
with open(file2, "w") as output_file:
    for key, value in extracted_data.items():
        output_file.write(f"{key}: {value}\n")

print("Data has been successfully written to 'product_classifier.txt'")
 """