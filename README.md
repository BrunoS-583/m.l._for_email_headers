### **M.L. For Email Headers**

### Prerequisites
Download SpamAssassin Dataset here: https://spamassassin.apache.org/old/publiccorpus/
 - [20030228_easy_ham.tar.bz2]
 - [20030228_easy_ham_2.tar.bz2]
 - [20021010_hard_ham.tar.bz2]
 - [20030228_spam.tar.bz2]
 - [20030228_spam_2.tar.bz2]


Unpack and move to `data` folder.

### Running Project
From the root of the project run:

`pip install -r requirements.txt`

`python3 src/data/make_dataset.py`

`python3 src/features/build_features.py`

`python3 src/models/prediction_model.py`
