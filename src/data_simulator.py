# data_simulator.py

import pandas as pd
import random

class DataSimulator:
    def __init__(self, general_no_show_reasons=None):
        if general_no_show_reasons is None:
            self.general_no_show_reasons = [
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
        else:
            self.general_no_show_reasons = general_no_show_reasons

    def get_age_based_reasons(self, age):
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
        else:
            return []

    def get_gender_based_reasons(self, gender):
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
        else:
            return []

    def generate_patient_notes(self, row):
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
        return " ".join(notes) if notes else "No ongoing chronic conditions noted. General checkup advised."

    def generate_patient_sentiment(self, row):
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

    def get_patient_notes(self, row):
        return self.generate_patient_notes(row)

    def get_patient_sentiment(self, row):
        return self.generate_patient_sentiment(row)

    def get_no_show_reason(self, row):
        reasons = self.general_no_show_reasons + \
                  self.get_gender_based_reasons(row['Gender']) + \
                  self.get_age_based_reasons(row['Age'])
        if str(row['No-show']).strip().lower() == 'yes':
            return random.choice(reasons)
        else:
            return "NA"

    def simulate(self, input_csv, output_csv, notes_col='PatientNotes', sentiment_col='PatientSentiment', reason_col='NoShowReason'):
        df = pd.read_csv(input_csv)
        df[notes_col] = df.apply(self.get_patient_notes, axis=1)
        df[sentiment_col] = df.apply(self.get_patient_sentiment, axis=1)
        df[reason_col] = df.apply(self.get_no_show_reason, axis=1)
        df.to_csv(output_csv, index=False)
        return df
