from pydantic import BaseModel, Field, validator
from typing import Optional, Union
import importlib


def load_schema(dataset_name):
    """
    Loads the corresponding Pydantic schema for the given dataset.

    This function imports the data_schema module dynamically and attempts to load the schema with
    the name derived from the dataset_name. The schema name is assumed to be the dataset name
    without the file extension, appended with '_schema'. 

    Args:
        dataset_name (str): The name of the dataset whose schema is to be loaded.

    Returns:
        The Pydantic schema for the given dataset.

    Raises:
        ValueError: If no schema is found for the given dataset name.
        
    Examples:
        >>> load_schema("heart.csv")
        HeartData
    """
    prompts_module = importlib.import_module("data_schema")
    pydantic_schema_name = dataset_name.split(".")[0] + "_schema"
    pydantic_schema = getattr(prompts_module, pydantic_schema_name, None)
    if pydantic_schema is None:
        raise ValueError(f"No pydantic_schema found for dataset '{dataset_name}'")
    return pydantic_schema


class HeartData(BaseModel):
    """Chest dataset"""
    age: Optional[int] = Field(description="Age of the patient [int](years)")
    sex: Optional[str] = Field(description="Sex of the patient [M,F] where M: Male, F: Female")
    chest_pain_type: Optional[str] = Field(
        description="Chest pain type [ATA, NAP, ASY, TA] where TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic")
    resting_bp: Optional[int] = Field(description="Resting blood pressure [int](mm Hg)")
    cholesterol: Optional[int] = Field(description="Serum cholesterol [int[(mm/dl)")
    fasting_bs: Optional[int] = Field(
        description="Fasting blood sugar [1,0] where 1: if FastingBS > 120 mg/dl, 0: otherwise")
    resting_ecg: Optional[str] = Field(
        description="Resting electrocardiogram results [Normal, ST, LVH] where Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria")
    max_hr: Optional[int] = Field(description="Maximum heart rate achieved [Numeric value between 60 and 202]")
    exercise_angina: Optional[str] = Field(description="Exercise-induced angina [Y,N] where Y: Yes, N: No")
    oldpeak: Optional[Union[float, int]] = Field(description="Oldpeak = ST Numeric value measured in depression")
    st_slope: Optional[str] = Field(
        description="The slope of the peak exercise ST segment [Up, Flat, Down] where Up: upsloping, Flat: flat, Down: downsloping")

    @validator("sex")
    def validate_sex(cls, v):
        if v is None or v == "" or v == " ":
            return None
        if v not in ("M", "F"):
            raise ValueError("Invalid value for Sex, allowed values are 'M' and 'F'")
        return v

    @validator("chest_pain_type")
    def validate_chest_pain_type(cls, v):
        if v is None or v == "" or v == " ":
            return None
        allowed_values = ("ATA", "NAP", "ASY", "TA")
        if v not in allowed_values:
            raise ValueError(f"Invalid value for ChestPainType, allowed values are {allowed_values}")
        return v

    @validator("resting_ecg")
    def validate_resting_ecg(cls, v):
        if v is None or v == "" or v == " ":
            return None
        allowed_values = ("Normal", "ST", "LVH")
        if v not in allowed_values:
            raise ValueError(f"Invalid value for RestingECG, allowed values are {allowed_values}")
        return v

    @validator("exercise_angina")
    def validate_exercise_angina(cls, v):
        if v is None or v == "" or v == " ":
            return None
        if v not in ("N", "Y"):
            raise ValueError("Invalid value for ExerciseAngina, allowed values are 'N' and 'Y'")
        return v

    @validator("st_slope")
    def validate_st_slope(cls, v):
        if v is None or v == "" or v == " ":
            return None
        allowed_values = ("Up", "Flat", "Down")
        if v not in allowed_values:
            raise ValueError(f"Invalid value for ST_Slope, allowed values are {allowed_values}")
        return v


heart_schema = HeartData


class MentalHelthData(BaseModel):
    """Mental helth dataset"""
    choose_your_gender: Optional[str] = Field(None, description="Gender of the student [Female, Male]")
    age: Optional[float] = Field(None, description="Age of the student")
    what_is_your_course: Optional[str] = Field(None, description="Course the student is enrolled in")
    your_current_year_of_study: Optional[str] = Field(None,
                                                      description="Current year of study [year 1, year 2, year 3, year 4]")
    what_is_your_cgpa: Optional[str] = Field(None,
                                             description="Current CGPA range of the student [0 - 1.99, 2.00 - 2.49, 2.50 - 2.99, 3.00 - 3.49, 3.50 - 4.00]")
    marital_status: Optional[str] = Field(None, description="Marital status of the student [yes, no]")
    do_you_have_depression: Optional[str] = Field(None, description="Indicates if the student has depression [yes, no]")
    do_you_have_anxiety: Optional[str] = Field(None, description="Indicates if the student has anxiety [yes, no]")
    do_you_have_panic_attack: Optional[str] = Field(None,
                                                    description="Indicates if the student has panic attacks [yes, no]")
    did_you_seek_any_specialist_for_treatment: Optional[str] = Field(None,
                                                                     description="Indicates if the student has sought treatment from a specialist [yes, no]")

    @validator("choose_your_gender")
    def validate_gender(cls, v):
        if v is None or v == "" or v == " ":
            return None
        if v.lower() not in ("female", "male"):
            raise ValueError("Invalid value for Gender, allowed values are 'Female' and 'Male'")
        return v.capitalize()

    @validator("your_current_year_of_study")
    def validate_year_of_study(cls, v):
        if v is None or v == "" or v == " ":
            return None
        if v.lower() not in ("year 1", "year 2", "year 3", "year 4"):
            raise ValueError(
                "Invalid value for Year of Study, allowed values are 'year 1', 'year 2', 'year 3', 'year 4'")
        return v.capitalize()

    @validator("marital_status", "do_you_have_depression", "do_you_have_anxiety", "do_you_have_panic_attack",
               "did_you_seek_any_specialist_for_treatment")
    def validate_yes_no(cls, v):
        if v is None or v == "" or v == " ":
            return None
        if v.lower() not in ("yes", "no"):
            raise ValueError("Invalid value, allowed values are 'yes' and 'no'")
        return v.capitalize()


mental_helth_schema = MentalHelthData


class TreatmentData(BaseModel):
    """Treatment dataset"""
    haematocrit: Optional[float] = Field(description="Patient laboratory test result of haematocrit")
    haemoglobins: Optional[float] = Field(description="Patient laboratory test result of haemoglobins")
    erythrocyte: Optional[float] = Field(description="Patient laboratory test result of erythrocyte")
    leucocyte: Optional[float] = Field(description="Patient laboratory test result of leucocyte")
    thrombocyte: Optional[float] = Field(description="Patient laboratory test result of thrombocyte")
    mch: Optional[float] = Field(description="Patient laboratory test result of MCH")
    mchc: Optional[float] = Field(description="Patient laboratory test result of MCHC")
    mcv: Optional[float] = Field(description="Patient laboratory test result of MCV")
    age: Optional[int] = Field(description="Patient age")
    sex: Optional[str] = Field(description="Sex of the patient [M,F] where M: Male, F: Female")
    source: Optional[str] = Field(
        description="In-care patient or out-care patient [in, out] where in: in-care, out: out-care")

    @validator("sex")
    def validate_sex(cls, v):
        if v is None or v == "" or v == " ":
            return None
        if v not in ("M", "F"):
            raise ValueError("Invalid value for Sex, allowed values are 'M' and 'F'")
        return v

    @validator("source")
    def validate_source(cls, v):
        if v is None or v == "" or v == " ":
            return None
        if v not in ("in", "out"):
            raise ValueError("Invalid value for Source, allowed values are 'in' and 'out'")
        return v


treatment_schema = TreatmentData


class StrokeData(BaseModel):
    """Stroke Dataset"""
    gender: str = Field(None, description="Gender of the patient: 'Male', 'Female', or 'Other'")
    age: float = Field(None, description="Age of the patient")
    hypertension: int = Field(None,
                              description="0 if the patient doesn't have hypertension, 1 if the patient has hypertension")
    heart_disease: int = Field(None,
                               description="0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease")
    ever_married: str = Field(None, description="Marital status of the patient: 'No' or 'Yes'")
    work_type: str = Field(None,
                           description="Work type of the patient: 'children', 'Govt_jov', 'Never_worked', 'Private' or 'Self-employed'")
    Residence_type: str = Field(None, description="Residence type of the patient: 'Rural' or 'Urban'")
    avg_glucose_level: float = Field(None, description="Average glucose level in blood")
    bmi: float = Field(None, description="Body mass index")
    smoking_status: Optional[str] = Field(None,
                                          description="Smoking status of the patient: 'formerly smoked', 'never smoked', 'smokes' or 'Unknown'. 'Unknown', np.nan or None means that the information is unavailable for this patient")

    @validator("gender")
    def validate_gender(cls, v):
        if v not in ("Male", "Female", "Other"):
            raise ValueError("Invalid gender, allowed values are 'Male', 'Female', and 'Other'")
        return v

    @validator("hypertension", "heart_disease")
    def validate_binary(cls, v):
        if v is None:
            return v
        if v not in (0, 1):
            raise ValueError("Invalid value, allowed values are 0 and 1")
        return v

    @validator("ever_married")
    def validate_ever_married(cls, v):
        if v is None:
            return v
        if v not in ("No", "Yes"):
            raise ValueError("Invalid value, allowed values are 'No' and 'Yes'")
        return v

    @validator("work_type")
    def validate_work_type(cls, v):
        allowed_values = ["children", "Govt_jov", "Never_worked", "Private", "Self-employed"]
        if v not in allowed_values:
            raise ValueError(f"Invalid work type, allowed values are {', '.join(allowed_values)}")
        return v

    @validator("Residence_type")
    def validate_Residence_type(cls, v):
        if v is None:
            return v
        if v not in ("Rural", "Urban"):
            raise ValueError("Invalid residence type, allowed values are 'Rural' and 'Urban'")
        return v

    @validator("smoking_status", pre=True)
    def validate_smoking_status(cls, v):
        if v is None:
            return "Unknown"
        allowed_values = ["formerly smoked", "never smoked", "smokes", "Unknown"]
        if v not in allowed_values:
            raise ValueError(f"Invalid smoking status, allowed values are {', '.join(allowed_values)}")
        return v


stroke_schema = StrokeData


class HepatitisData(BaseModel):
    age: Optional[int] = Field(None, description="Age of the patient in years")
    sex: Optional[str] = Field(None, description="Sex of the patient [f, m] ('f'=female, 'm'=male)")
    ALB: Optional[float] = Field(None, description="Amount of albumin in patient's blood")
    ALP: Optional[float] = Field(None, description="Amount of alkaline phosphatase in patient's blood")
    ALT: Optional[float] = Field(None, description="Amount of alanine transaminase in patient's blood")
    AST: Optional[float] = Field(None, description="Amount of aspartate aminotransferase in patient's blood")
    BIL: Optional[float] = Field(None, description="Amount of bilirubin in patient's blood")
    CHE: Optional[float] = Field(None, description="Amount of cholinesterase in patient's blood")
    CHOL: Optional[float] = Field(None, description="Amount of cholesterol in patient's blood")
    CREA: Optional[float] = Field(None, description="Amount of creatine in patient's blood")
    GGT: Optional[float] = Field(None, description="Amount of gamma-glutamyl transferase in patient's blood")
    PROT: Optional[float] = Field(None, description="Amount of protein in patient's blood")
    diagnosis: Optional[str] = Field(None,
                                     description="Diagnosis of the patient: Is pateinet Healthy or has Hepatitis ['Healthy', 'Hepatitis'] (Healthy=patient does't have liver illness, Hepatitis=patient has liver illness or Hepatitis)")

    @validator("sex")
    def validate_sex(cls, v):
        if v not in ("f", "m"):
            raise ValueError("Invalid value for Sex, allowed values are 'f' and 'm'")
        return v

    @validator("diagnosis")
    def validate_diagnosis(cls, v):
        if v not in ("Healthy", "Hepatitis"):
            raise ValueError("Invalid value for Diagnosis, allowed values are 'Healthy' and 'Hepatitis'")
        return v


hepatitis_c_schema = HepatitisData
