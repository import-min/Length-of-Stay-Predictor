import pandas as pd
import random


icd_df = pd.read_excel("section111validicd10-jan2025_0.xlsx")
valid_icd_codes = icd_df['CODE'].tolist()



dummy_data = []
for i in range(100): #making 100 records to start
    subject_id = 10000000 + i
    gender = random.choice(['M', 'F'])
    anchor_age = random.randint(18, 90)
    anchor_year = random.randint(2000, 2025)
    admission_type = random.choice([
        'AMBULATORY OBSERVATION', 'DIRECT EMER.', 'DIRECT OBSERVATION',
        'ELECTIVE', 'EU OBSERVATION', 'EW EMER.', 'OBSERVATION ADMIT',
        'SURGICAL SAME DAY ADMISSION', 'URGENT'
    ])
    admission_location = random.choice([
        'EMERGENCY ROOM ADMIT', 'CLINIC REFERRAL/PREMATURE', 'TRANSFER FROM HOSP/EXTRAM',
        'TRANSFER FROM SKILLED NUR', 'PHYS REFERRAL/NORMAL DELI', 'TRANSFER FROM OTHER HEALT',
        '** INFO NOT AVAILABLE **', 'HMO REFERRAL/SICK', 'TRSF WITHIN THIS FACILITY'
    ])
    icd_code = random.choice(valid_icd_codes)
    dummy_data.append({
        "subject_id": subject_id,
        "gender": gender,
        "anchor_age": anchor_age,
        "anchor_year": anchor_year,
        "admission_type": admission_type,
        "admission_location": admission_location,
        "icd_code": icd_code
    })
dummy_df = pd.DataFrame(dummy_data)
dummy_df.to_csv("dummy_mimic_data.csv", index=False)
