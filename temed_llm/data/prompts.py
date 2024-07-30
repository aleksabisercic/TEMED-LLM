import importlib


def load_prompt(dataset_name):
    """
    Loads the corresponding prompt for the given dataset.

    This function imports the prompts module dynamically and attempts to load the prompt with
    the name derived from the dataset_name. The prompt name is assumed to be the dataset name
    without the file extension, appended with '_prompt'.

    Args:
        dataset_name (str): The name of the dataset whose prompt is to be loaded.

    Returns:
        The prompt for the given dataset.

    Raises:
        ValueError: If no prompt is found for the given dataset name.
    """
    prompts_module = importlib.import_module("prompts")
    prompt_name = dataset_name.split(".")[0] + "_prompt"
    prompt = getattr(prompts_module, prompt_name, None)
    if prompt is None:
        raise ValueError(f"No prompt found for dataset '{dataset_name}'")
    return prompt


heart_prompt = """{pydantic_output_parser}

When generating JSON instance follow this format:

Mdical report: the input medical report from which you should extract JSON instance
Reasoning: Provide explanation of how you assign value for a given key. Thinking step by step for each key before assigning a value to it.
Output json: The final output json should be formatted as a JSON instance that conforms to the output JSON schema above

Here is example of a process:
Medical report:
"REASON FOR CONSULTATION: Chest pain.
HISTORY OF PRESENT ILLNESS: The patient is a 63-year-old Caucasian male with a past medical history significant for hypertension and hyperlipidemia. He presents with chest pain that occurs during moderate physical activity. The patient has a history of smoking and occasional alcohol consumption.
PAST MEDICAL HISTORY:Hypertension.Hyperlipidemia.Type 2 Diabetes.
ALLERGIES: No known drug allergies.
FAMILY HISTORY: Father had a myocardial infarction at the age of 65.
SOCIAL HISTORY: 20 pack-year smoking history, occasional alcohol consumption, and no illicit drug use.
CURRENT MEDICATIONS: Amlodipine, Atorvastatin, Metformin, Aspirin.
REVIEW OF SYSTEMS: Patient reports occasional shortness of breath and dizziness.
PHYSICAL EXAMINATION:
VITAL SIGNS: Blood pressure 145/90, pulse 100, O2 saturation 98% on room air, temperature 98.6, respiratory rate 18.
GENERAL: Patient is alert and oriented, and appears comfortable at rest.
HEAD AND NECK: No JVP seen, no carotid bruits.
CHEST: Clear to auscultation bilaterally.
CARDIOVASCULAR: Regular rhythm, standard S1, and S2, no murmurs, rubs, or gallops.
ABDOMEN: Soft, non-tender, and non-distended.
EXTREMITIES: No edema or clubbing.
DATA:A 12-lead EKG revealed normal sinus rhythm with a rate of 100 beats per minute and nonspecific ST-T wave abnormality.
LABORATORY DATA: WBC 6.8, hemoglobin 14.2, platelets 250. Chemistry-7 within normal limits. Lipid profile: Triglycerides 160, total cholesterol 220, HDL 40, LDL 140. Fasting blood sugar 125 mg/dL. Liver function tests within normal limits. BUN and creatinine within normal limits.
TREADMILL STRESS TEST: The patient achieved a maximum heart rate of one hundred and fifty beats per minute. No chest pain or shortness of breath during the test. Oldpeak of 1.5 mm noted in leads II, III, and aVF. Shift relative to exercise-induced increments in heart rate is showing a downward trend.

Reasoning:
The report states that the patient is a 63-year-old Caucasian male, therefore "Age": 63.
The patient is described as a male in the report, therefore "Sex": "M".
The patient is described to be experiencing chest discomfort without other typical symptoms, such as shortness of breath, dizziness, or sweating. Based on this information, the most appropriate 'chest_pain_type' for this patient would be "ASY", which stands for "asymptomatic." Therefore chest_pain_type: ASY
The patient's blood pressure is mentioned in the "VITAL SIGNS" section as 145/90, therefore "resting_bp": 145.
The report mentions the patient's total cholesterol value under "LABORATORY DATA": "total cholesterol 220", therefore "cholesterol": 220.
The fasting blood sugar is mentioned in the "LABORATORY DATA" section as 125 mg/dL, which is greater than 120 mg/dL, so the value is 1, therefore "fasting_bs": 1.
The 12-lead EKG results in the report revealed normal sinus rhythm, therefore "resting_ecg": "Normal".
The maximum heart rate achieved during the treadmill stress test is mentioned as one hundred and fifty beats per minute, therefore "max_hr": 150.
The report states that the patient did not experience chest pain or shortness of breath during the treadmill stress test, so the value is "N" for No, therefore "exercise_angina": "N".
The oldpeak value is mentions the report as 1.5 mm in leads II, III, and aVF during the treadmill stress test, therefore "oldpeak": 1.5.
The report mentions that the ST-segment slope during the treadmill stress test is downsloping, therefore "st_slope": "Down"

Output json:
{"age": 63, "sex": "M", "chest_pain_type": "ATA", "resting_bp": 145, "cholesterol": 220, "fasting_bs": 1, "resting_ecg": "Normal", "max_hr": 150, "exercise_angina": "N", "oldpeak": 1.5, "st_slope": "Down"}
---

Medical report:
{medical_report_to_replace}

Reasoning:
"""

treatment_prompt = """{pydantic_output_parser}

When generating JSON instance follow this format:

Mdical report: the input medical report from which you should extract JSON instance
Reasoning: Provide explanation of how you assign value for a given key. Thinking step by step for each key before assigning a value to it.
Output json: The final output json should be formatted as a JSON instance that conforms to the output JSON schema above


Here is example of a process:
Medical notes:
Young female individual, currently 12 years old, has come forward with complaints of listlessness and a lack of color. There are no major incidents in her past health record, and she doesn't seem to react adversely to any known substances. A look into her eyes and at her nails shows a lack of color, a finding that is clinically significant. However, no other physical anomalies have been found. Lab studies and the way the individual presents suggest that she may be dealing with anemia.
Test results from the lab are as follows:
The volume fraction of red blood cells, or the haematocrit, is at a level of 35.1%, with the typical range for the fairer sex being 36-48%. This suggests there may be some issues with the blood or it might be anemia. The protein in red blood cells that carries oxygen, haemoglobins, is at a level of 11.8 g/dL, with the norm being 12-16 g/dL, which adds weight to the suspicion of anemia.
The count of erythrocytes, or red blood cells, is 4.65 million cells/mcL, with a normal range of 4-5.2 million cells/mcL. The count of leucocytes, or white blood cells, is 6.3 x 10^3 cells/mcL, with a normal range of 4-11 x 10^3 cells/mcL. The thrombocyte, or platelet count, is 310 x 10^3 cells/mcL, with the normal range being 150-450 x 10^3 cells/mcL.
The mean corpuscular hemoglobin, or MCH, is 25.4 pg. The mean corpuscular hemoglobin concentration, or MCHC, is 33.6 g/dL. The mean corpuscular volume, or MCV, is 75.5 fL, slightly lower than the usual 80-100 fL, hinting at possible microcytic anemia.
Putting the individual's years lived, gender, clinical discoveries, and lab data together, it seems likely that the cause of these symptoms is anemia caused by a lack of iron. For a more definitive answer, more tests, including serum iron, total iron-binding capacity, and ferritin levels, would need to be conducted.

Reasoning:
Haematocrit: The  report states that "The volume fraction of red blood cells, or the haematocrit, is at a level of 35.1%". Hence, the haematocrit value is 35.1.
Haemoglobins: The report states that "The protein in red blood cells that carries oxygen, haemoglobins, is at a level of 11.8 g/dL". Hence, the haemoglobins value is 11.8.
Erythrocyte: The report states that "The count of erythrocytes, or red blood cells, is 4.65 million cells/mcL". Hence, the erythrocyte count is 4.65.
Leucocyte: The report states that "The count of leucocytes, or white blood cells, is 6.3 x 10^3 cells/mcL". Hence, the leucocyte count is 6.3.
Thrombocyte: The report states that "The thrombocyte, or platelet count, is 310 x 10^3 cells/mcL". Hence, the thrombocyte count is 310.
MCH: The report states that "The mean corpuscular hemoglobin, or MCH, is 25.4 pg". Hence, the MCH value is 25.4.
MCHC: The report states that "The mean corpuscular hemoglobin concentration, or MCHC, is 33.6 g/dL". Hence, the MCHC value is 33.6.
MCV: The report states that "The mean corpuscular volume, or MCV, is 75.5 fL". Hence, the MCV value is 75.5.
Age: The  report refers to "a young female individual, currently stating that she is 12. Hence, the patient's age is 12.
Sex: The re-written report refers to "a young female individual". Hence, the sex of the patient is "F".

Output json:
{ "haematocrit": 35.1, "haemoglobins": 11.8, "erythrocyte": 4.65, "leucocyte": 6.3, "thrombocyte": 310, "mch": 25.4, "mchc": 33.6, "mcv": 75.5, "age": 12, "sex": "F", "source": "in" }
----
START

Medical report:
{medical_report_to_replace}

Reasoning:
"""

mental_helth_prompt = """{pydantic_output_parser}

When generating JSON instance follow this format:

Mdical report: the input medical report from which you should extract JSON instance
Reasoning: Provide explanation of how you assign value for a given key. Thinking step by step for each key before assigning a value to it.
Output json: The final output json should be formatted as a JSON instance that conforms to the output JSON schema above


Here is example of a process:
Medical notes:
In the middle of a summer day on the seventh of August, 2020, precisely at four past midday, an encounter took place with a scholar who is in his early twenties. This young man is pursuing knowledge in the field of Islamic Studies and has already completed a full academic year, making him a second-year apprentice of knowledge. His academic performance has been commendable, as he's managed to secure a Cumulative Grade Point Average (CGPA) that floats between the range of 3.00 and 3.49.

He made it known that he currently has no spouse, a state which inherently may mold the structure of his social interactions as well as the challenges he grapples with. This scholar divulged worries related to his mental well-being, with a specific reference to anxiety. Interestingly, he denied experiencing depressive episodes or panic-filled incidents, this might provide us with a more streamlined avenue to address his anxiety.

Past attempts at seeking professional guidance or treatment for his concerns were negated. The information gleaned from this interaction suggests the possibility that the scholar may stand to gain from more thorough evaluations and tailored interventions to navigate his anxiety more effectively.

To this end, a referral to a proficient hand in the field of mental health, like a clinical psychologist or counselor, might prove useful. In tandem with this, the scholar might find solace in engaging with methods designed to manage stress and relaxation exercises, with the aim of taking control of his anxiety symptoms.

Reasoning:
Choose Your Gender: The medical notes refer to the individual as a "young man" and "he," indicating that the person identifies as male.
age: The notes refer to the individual as being "in his early twenties." Given the lack of a specific age, we will assign the value of None. Therefore, "age": 21.
what_is_your_course: It's mentioned that the individual is studying "Islamic Studies." Therefore, "what_is_your_course": "Islamic Studies".
your_current_year_of_study: The notes state that the individual "has already completed a full academic year," indicating that he's currently in his second year of study. Therefore, "your_current_year_of_study": "year 2".
what_is_your_cgpa: The notes mention a CGPA "between the range of 3.00 and 3.49." Therefore, "what_is_your_cgpa": "3.00 - 3.49".
marital_status: The individual stated he "has no spouse," indicating he is not married. Therefore, "marital_status": "no".
do_you_have_depression: The notes specifically state that the individual denied "experiencing depressive episodes," indicating he does not have depression. Therefore, "do_you_have_depression": "no".
do_you_have_anxiety: The individual has expressed concerns about his mental well-being, specifically mentioning anxiety. Therefore, "do_you_have_anxiety": "yes".
do_you_have_panic_attack: The notes state that the individual denied "panic-filled incidents," indicating he does not have panic attacks. Therefore, "do_you_have_panic_attack": "no".
did_you_seek_any_specialist_for_treatment: It's mentioned in the notes that the individual has not sought professional help for his concerns in the past. Therefore, "did_you_seek_any_specialist_for_treatment": "no".
Output JSON:

{
"choose_your_gender": "Male",
"age": None,
"what_is_your_course": "Islamic Studies",
"your_current_year_of_study": "year 2",
"what_is_your_cgpa": "3.00 - 3.49",
"marital_status": "no",
"do_you_have_depression": "no",
"do_you_have_anxiety": "yes",
"do_you_have_panic_attack": "no


Output json:
{"choose_your_gender": "Male", "age": 21.0, "what_is_your_course": "Islamic education", "your_current_year_of_study": "year 2", "what_is_your_cgpa": "3.00 - 3.49", "marital_status": "No", "do_you_have_depression": "No", "do_you_have_anxiety": "Yes", "do_you_have_panic_attack": "No", "did_you_seek_any_specialist_for_treatment": "No"}
----
START

Medical report:
{medical_report_to_replace}

Reasoning:
"""

hepatitis_c_prompt = """{pydantic_output_parser}

When generating JSON instance follow this format:

Mdical report: the input medical report from which you should extract JSON instance
Reasoning: Provide explanation of how you assign value for a given key. Thinking step by step for each key before assigning a value to it.
Output json: The final output json should be formatted as a JSON instance that conforms to the output JSON schema above


Here is example of a process:
Medical notes:
An individual of male, having completed 32 solar cycles, underwent a series of tests aimed at examining hepatic function. An absence of noteworthy historical and clinical data was noted for this individual. As per the scrutinized lab results, his hepatic functions seem to be running smoothly.
The concentration of a certain protein - albumin, to be exact - was noted to be 38.5 g/L, comfortably residing within the accepted range of 35-55 g/L. This could be a sign of sufficient protein production by the liver. The alkaline phosphatase, or ALP, measured at 70.3 U/L, is also within the typical parameters (40-130 U/L), hinting at the lack of severe hepatic or skeletal disorders.
Additional indicators, namely alanine transaminase (ALT) and aspartate aminotransferase (AST), clocked in at 18.0 U/L and 24.7 U/L respectively, both within the usual limits (7-56 U/L for ALT, and 10-40 U/L for AST). This further bolsters the suggestion of no significant hepatic or muscular impairment.
Bilirubin, or BIL, another liver health marker, was found to be 3.9 μmol/L, a level that falls within the normal range (1.2-17.1 μmol/L), suggesting no significant hepatic disorders or anemia. Another chemical, cholinesterase (CHE), was found to be 11.17 U/L, again within the normal limits (5.3-12.9 U/L), which implies no significant liver issues.
The cholesterol or CHOL level, a marker of lipid health, was noted to be 4.8 mmol/L, fitting within the normal range (3.6-7.8 mmol/L), suggesting a good lipid profile. Creatinine or CREA, a marker of kidney health, was found to be 74.0 μmol/L, a value within the normal range (62-106 μmol/L), indicating no renal issues.
Finally, gamma-glutamyl transferase (GGT) and total protein (PROT), with values of 15.6 U/L and 76.5 g/L respectively, were both within their normal ranges (9-64 U/L for GGT, and 60-83 g/L for PROT). This implies no significant damage to the liver or bile duct and adequate protein production by the liver.
In summary, upon thorough investigation of the lab data, it can be inferred that the liver is operating satisfactorily with no signs of damage or dysfunction. Lipid and kidney health also appear to be maintained within the normal parameters. There is no necessity for additional interventions at this stage.

Reasoning:
'age' key, the text mentions that the individual has 'completed 32 solar cycles', which is a creative way of saying that the patient is 32 years old. Therefore, "age": 32.
'sex', the report mentions 'an individual of the male gender', which means the patient is male. Therefore, "sex": "m".
'ALB' key stands for the albumin level, which the report states as '38.5 g/L'. Therefore, "ALB": 38.5.
'ALP', representing alkaline phosphatase, the report mentions it as '70.3 U/L'. Therefore, "ALP": 70.3.
'ALT' and 'AST', standing for alanine transaminase and aspartate aminotransferase respectively, the report mentions them as '18.0 U/L' and '24.7 U/L' respectively. Therefore, "ALT": 18.0 and "AST": 24.7.
'BIL' key represents the bilirubin level, which the report states as '3.9 μmol/L'. Therefore, "BIL": 3.9.
'CHE', indicating cholinesterase, the report provides a figure of '11.17 U/L'. Therefore, "CHE": 11.17.
'CHOL', standing for cholesterol, the report mentions it as '4.8 mmol/L'. Therefore, "CHOL": 4.8.
'CREA' key, representing creatinine, is mentioned in the report as '74.0 μmol/L'. Therefore, "CREA": 74.0.
'GGT' and 'PROT', standing for gamma-glutamyl transferase and total protein respectively, are stated as '15.6 U/L' and '76.5 g/L' respectively. Therefore, "GGT": 15.6 and "PROT": 76.5.


Output json:
{'Age': 32, 'Sex': 'm', 'ALB': 38.5, 'ALP': 70.3, 'ALT': 18.0, 'AST': 24.7, 'BIL': 3.9, 'CHE': 11.17, 'CHOL': 4.8, 'CREA': 74.0, 'GGT': 15.6, 'PROT': 76.5}
----
START

Medical report:
{medical_report_to_replace}

Reasoning:
"""

stroke_prompt = """{pydantic_output_parser}

When generating JSON instance follow this format:

Mdical report: the input medical report from which you should extract JSON instance
Reasoning: Provide explanation of how you assign value for a given key. Thinking step by step for each key before assigning a value to it.
Output json: The final output json should be formatted as a JSON instance that conforms to the output JSON schema above

Here is example of a process:
Medical report:
Subject: 49-year-old Female Patient

I. Patient History
A 49-year-old married female patient, employed in the private sector and residing in an urban area, presents for evaluation. She reports no history of hypertension or heart disease.

II. Clinical Findings
A. No hypertension
B. No heart disease

III. Diagnostic Laboratory Test Results
A. Average Glucose Level: 171.23 mg/dL (elevated, normal range: 70-140 mg/dL)
B. Body Mass Index (BMI): 34.4 kg/m2 (classified as obesity, normal range: 18.5-24.9 kg/m2)

IV. Risk Factors
A. Smoking status: Active smoker

V. Comments on General Presentation
The patient exhibits an elevated average glucose level and a BMI within the obesity range. She has no prior history of hypertension or heart disease. However, the patient actively smokes, a known risk factor for various health complications.

VI. Management
A. Lifestyle Recommendations

Healthy diet: Encourage the patient to consume a balanced diet, rich in fruits, vegetables, whole grains, lean protein, and healthy fats.
Physical activity: Encourage the patient to engage in moderate-intensity aerobic exercise for at least 150 minutes per week or vigorous-intensity aerobic exercise for at least 75 minutes per week, in addition to muscle-strengthening activities on two or more days per week.
Smoking cessation: Recommend smoking cessation programs and resources to help the patient quit smoking.
B. Monitoring

Regular check-ups: Schedule periodic appointments to monitor the patient's glucose levels, blood pressure, and cholesterol.
VII. Opinion
Considering the patient's elevated average glucose level, obesity, and smoking habit, her risk for future health complications, including diabetes, hypertension, and heart disease, is significantly increased. Close monitoring and adherence to recommended lifestyle changes are crucial to minimize the risk of further health issues and improve overall health outcomes.

Output json:
{"gender":"Female","age":49.0,"hypertension":0,"heart_disease":0,"ever_married":"Yes","work_type":"Private","Residence_type":"Urban","avg_glucose_level":171.23,"bmi":34.4,"smoking_status":"smokes"}
----
START

Medical report:
{medical_report_to_replace}

"""

# Add more prompts for other datasets as needed
# diabetes_prompt = "..."
# cancer_prompt = "..."
