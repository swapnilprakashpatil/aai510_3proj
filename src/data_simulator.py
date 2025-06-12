# data_simulator.py

import pandas as pd
import random

class DataSimulator:
    def __init__(self, general_no_show_reasons=None):
        self.general_no_show_reasons = general_no_show_reasons or self._default_no_show_reasons()

    def simulate(self, input_csv, output_csv, notes_col='PatientNotes', sentiment_col='PatientSentiment', reason_col='NoShowReason'):
        df = pd.read_csv(input_csv)
        df[notes_col] = df.apply(self._generate_patient_notes, axis=1)
        df[sentiment_col] = df.apply(self._generate_patient_sentiment, axis=1)
        df[reason_col] = df.apply(self._generate_no_show_reason, axis=1)
        df.to_csv(output_csv, index=False)
        return df

    def _default_no_show_reasons(self):
        return [
            "Patient reported forgetting the appointment.",
            "Missed due to lack of transportation.",
            "Appointment was too early in the morning.",
            "Family emergency prevented attendance.",
            "Patient confused the appointment date.",
            "Patient felt better and didn't think follow-up was necessary.",
            "Couldn't get time off from work.",
            "Patient decided to try alternative treatment.",
            "Bad weather on appointment day.",
            "Patient was anxious about the visit.",
        ]

    def _generate_no_show_reason(self, row):
        reasons = self.general_no_show_reasons + self._get_gender_based_reasons(row['Gender']) + self._get_age_based_reasons(row['Age'])
        return random.choice(reasons) if str(row['No-show']).strip().lower() == 'yes' else "NA"

    def _get_age_based_reasons(self, age):
        if age < 18:
            return [
                "Minor dependent on guardian who couldn’t bring them.",
                "School-related activities caused scheduling conflict."
            ]
        elif age >= 60:
            return [
                "Patient reported mobility issues.",
                "Health deterioration on the day of appointment.",
                "Elderly patient forgot the appointment time.",
            ]
        return []

    def _get_gender_based_reasons(self, gender):
        gender = str(gender).strip().upper()
        if gender == 'F':
            return [
                "Patient had childcare responsibilities.",
                "Couldn't leave home due to household duties.",
            ]
        elif gender == 'M':
            return [
                "Patient reported high workload at job.",
                "Missed due to extended shift or overtime.",
            ]
        return []

    def _generate_patient_notes(self, row):
        notes = []
        if row['Hypertension'] == 1:
            notes.append(
                "Patient has a known history of hypertension. Prescribed Amlodipine 5mg daily. Advised salt intake reduction."
            )
        if row['Diabetes'] == 1:
            notes.append(
                "Type 2 diabetes under monitoring. Continued Metformin 500mg BID. Scheduled next HbA1c test in 3 months."
            )
        if row['Alcoholism'] == 1:
            notes.append(
                "Patient reports ongoing alcohol consumption. Provided brief intervention and referral to support group."
            )
        if row['Handcap'] > 0:
            notes.append(f"Patient reports handicap level {row['Handcap']}. Needs assistance on visit.")

        if row.get('SMS_received', 0) == 1:
            notes.append("Patient received SMS reminder for appointment.")
        if row.get('Scholarship', 0) == 1:
            notes.append("Patient is enrolled in healthcare scholarship program.")
        if row.get('Age', 0) > 65:
            notes.append("Geriatric patient. Fall risk assessment recommended.")
        if row.get('Age', 0) < 12:
            notes.append("Pediatric patient. Parent/guardian present during consultation.")
            notes.append("Discussed vaccination schedule and growth milestones with parent/guardian.")
            if row.get('No-show', '').strip().lower() == 'yes':
                notes.append("Missed appointment discussed with parent/guardian. Emphasized importance of regular pediatric visits.")
        elif row.get('Age', 0) < 18:
            notes.append("Minor patient. Consent obtained from parent/guardian for treatment.")
            if row.get('No-show', '').strip().lower() == 'yes':
                notes.append("Missed appointment discussed with parent/guardian. Explored barriers to attendance for minors.")
        if row.get('No-show', '').strip().lower() == 'yes':
            notes.append("Patient previously missed appointments. Discussed barriers to attendance.")
        if row.get('Gender', '').strip().upper() == 'F' and row.get('Age', 0) > 18:
            notes.append("Discussed women's health screening and preventive care.")
        if row.get('Gender', '').strip().upper() == 'M' and row.get('Age', 0) > 18:
            notes.append("Discussed men's health and cardiovascular risk factors.")
        return " ".join(notes) if notes else "No ongoing chronic conditions noted. General checkup advised."

    def _generate_patient_sentiment(self, row):
        sentiments = []
        if row['Diabetes'] == 1:
            sentiments.append("Patient experiences stress and anxiety managing blood sugar levels and dietary restrictions.")
            if str(row['No-show']).strip().lower() == 'yes':
                sentiments.append("Patient feels overwhelmed and confused by frequent diabetes appointments, leading to missed visits.")
        if row['Hypertension'] == 1:
            sentiments.append("Patient expresses fear and anxiety about high blood pressure and possible complications.")
            if str(row['No-show']).strip().lower() == 'yes':
                sentiments.append("Patient is fearful of medication side effects and feels hopeless about long-term control, contributing to avoidance of follow-ups.")
        if row['Alcoholism'] == 1:
            sentiments.append("Patient feels stigma, stress, and anxiety discussing alcohol use with healthcare staff.")
            if str(row['No-show']).strip().lower() == 'yes':
                sentiments.append("Patient avoids appointments due to fear of judgment and hopelessness about recovery.")
        if row['Age'] < 18:
            sentiments.append("Patient (minor) is anxious and fearful about medical procedures, sometimes confused by instructions, and stressed by separation from family.")
        elif row['Age'] >= 60:
            sentiments.append("Elderly patient expresses fear of declining health, confusion about medications, and stress related to mobility issues.")
        if not sentiments:
            sentiments.append("Patient is hopeful and shows no significant anxiety, stress, or fear related to health conditions.")
        return " ".join(sentiments)
