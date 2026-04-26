import os
import csv
from email import policy
from email.parser import BytesParser

# Converts raw SpamAssassin dataset folders into a single structured CSV file
# Each email becomes one row with selected header fields + label

def makeCSVfolders():
    src = "../../spamAssassinDataset"
    folder_labels = ["easyham", "hardham", "spam"]
    output_file = "../../data/pre_processed.csv"
    
    fields = [
        "Return-Path", "Received", "X-Authentication-Warning",
        "From", "To", "Cc", "Subject", "In-Reply-To", "References", 
        "Message-Id", "Sender","Errors-To", "Date"
    ]
    fields = ["Label"] + fields
    
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()

        for label in folder_labels:
            input_dir = f"{src}/{label}"

            for root, _, files in os.walk(input_dir):
                for filename in files:
                    file_path = f"{root}/{filename}"

                    with open(file_path, "rb") as f:
                        msg = BytesParser(policy=policy.default).parse(f)

                    row = {"Label": label}
                    for field in fields[1:]:
                        if(field == 'Received'):
                            value = msg.get_all(field)
                            row[field] = " || ".join(value) if value else ""
                        else:
                            value = msg.get(field)
                            row[field] = value if value else ""

                    writer.writerow(row)
def main():
    print("Creating pre_processed.csv ...")
    makeCSVfolders()
    print("pre_processed.csv created")

if __name__ == "__main__":
    main()