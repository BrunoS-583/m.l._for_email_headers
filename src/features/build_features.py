import pandas as pd
import re
from email.utils import parseaddr, getaddresses

# Converts raw email data into engineered numerical features for ML models
# Outputs full feature set and a numeric-only dataset for training

def extractEmail(email):
    if email and "@" in email:
        return email.split("@")[-1].lower()
    else:
        return ""

def safeLength(x):
    if x:
        return len(x)
    else:
        return 0

def countNumWords(x):
    totalWords = x.split(" ")
    return len(totalWords)

def countNumCapital(x):
    numCapital = 0
    for i in x:
        if i.isupper():
            numCapital += 1
    return numCapital

def countDigit(x):
    numDigits = 0
    for i in x:
        if i.isdigit():
            numDigits += 1
    return numDigits

def countNumSpecial(x):
    numSpecial = 0
    specialChar = {'~','!','@','#','$','%','^','&','*','(',')','-','_','+','=','[',']','<','>','?'}
    for i in x:
        if i in specialChar:
            numSpecial += 1
    return numSpecial

def ifSpecial(x):
    specialChar = {'~','!','@','#','$','%','^','&','*','(',')','-','_','+','=','[',']','<','>','?'}
    for i in x:
        if i in specialChar:
            return 1
    return 0

def maxWordLength(x):
    maxLen = 0; currentLength = 0

    for char in x:
        if(char != ' '):
            currentLength += 1
        else:
            maxLen = max(maxLen, currentLength)
            currentLength = 0
    
    return max(maxLen, currentLength)

def extractFrom(x):
    name, email = parseaddr(str(x))
    domain = extractEmail(email)
    return pd.Series({
        "From_Domain": domain,
        "From_Has_Name": int(name != ""),
        "From_Is_Empty": int(email == "")
    })

def extractReturn(x):
    _, returnEmail = parseaddr(str(x))
    domain = extractEmail(returnEmail)
    return pd.Series({
        "Return_Domain": domain,
        "Return_Is_Empty": int(returnEmail == "")
    })

def extractSender(x):
    _, senderEmail = parseaddr(str(x))
    domain = extractEmail(senderEmail)
    return pd.Series({
        "Sender_Domain": domain,
        "Sender_Exists": int(senderEmail != "")
    })

def extractDomainMatches(row):
    return pd.Series({
        "From_Return_Match": int(row["From_Domain"] == row["Return_Domain"] and row["From_Domain"] != ""),
        "From_Sender_Match": int(row["From_Domain"] == row["Sender_Domain"] and row["From_Domain"] != "")
    })

def extractReceived(x):
    received = str(x)
    hops = received.split("||") if received else []
    IPS = re.findall(r'\d+\.\d+\.\d+\.\d+', received)
    return pd.Series({
        "Num_Received": len(hops),
        "Num_IPS": len(IPS),
        "Num_Unique_IPS": len(set(IPS))
    })

def extractRecipient(to_field, cc_field):
    To_Addresses = getaddresses([str(to_field)])
    Cc_Addresses = getaddresses([str(cc_field)])
    return pd.Series({
        "Num_To": len(To_Addresses),
        "Num_Cc": len(Cc_Addresses),
        "Num_Recipients": len(To_Addresses) + len(Cc_Addresses)
    })

def extractWarning(auth_field, errors_field):
    return pd.Series({
        "Has_Auth_Warning": int(str(auth_field) != ""),
        "Has_Errors_To": int(str(errors_field) != "")
    })

def extractThreading(reply_field, references_field):
    return pd.Series({
        "Has_Reply": int(str(reply_field) != ""),
        "Has_References": int(str(references_field) != "")
    })

def extractMsgId(x):
    msg_id = str(x)
    return pd.Series({
        "Msg_Id_Length": len(msg_id),
        "Msg_Id_Has_At": int("@" in msg_id)
    })

def extractLabel(x):
    mapping = {'easyham': 0, 'hardham': 0, 'spam': 1}
    return mapping.get(x)

def featureExtraction(df):
    df['Subject'] = df['Subject'].fillna("")

    df['Words_In_Subject'] = df['Subject'].apply(countNumWords)
    df['Num_Capital_Subject'] = df['Subject'].apply(countNumCapital)
    df['Num_Digits_Subject'] = df['Subject'].apply(countDigit)
    df['Num_Special_Subject'] = df['Subject'].apply(countNumSpecial)
    df['Special_Exists_Subject'] = df['Subject'].apply(ifSpecial)
    df['Max_Word_Length_Subject'] = df['Subject'].apply(maxWordLength)

    df[["From_Domain","From_Has_Name","From_Is_Empty"]] = df["From"].apply(extractFrom)
    df[["Return_Domain","Return_Is_Empty"]] = df["Return-Path"].apply(extractReturn)
    df[["Sender_Domain","Sender_Exists"]] = df["Sender"].apply(extractSender)
    df[["From_Return_Match","From_Sender_Match"]] = df.apply(extractDomainMatches, axis=1)
    df[["Num_Received","Num_IPS","Num_Unique_IPS"]] = df["Received"].apply(extractReceived)
    df[["Num_To","Num_Cc","Num_Recipients"]] = df.apply(lambda r: extractRecipient(r["To"], r["Cc"]), axis=1)
    df[["Has_Auth_Warning","Has_Errors_To"]] = df.apply(lambda r: extractWarning(r["X-Authentication-Warning"], r["Errors-To"]), axis=1)
    df[["Has_Reply","Has_References"]] = df.apply(lambda r: extractThreading(r["In-Reply-To"], r["References"]), axis=1)
    df[["Msg_Id_Length","Msg_Id_Has_At"]] = df["Message-Id"].apply(extractMsgId)
    
    df['Label'] = df['Label'].apply(extractLabel)
    
    df.to_csv("../../data/data_with_features.csv", index = False)

    numeric_df = df.select_dtypes(include=["number", "bool"])
    numeric_df.to_csv("../../data/data_numeric_only.csv", index=False)
    
def main():
    print("Starting feature extraction")
    df = pd.read_csv("../../data/pre_processed.csv")
    featureExtraction(df)
    print("Feature extraction completed")

if __name__ == "__main__":
    main()