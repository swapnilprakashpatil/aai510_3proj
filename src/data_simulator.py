# data_simulator.py

import pandas as pd
import random
import os
import csv

class DataSimulator:
    def __init__(self, general_no_show_reasons=None):
        self.synthetic_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'synthetic_reasons')
        self.no_show_reasons = self._load_reasons_from_csv('no_show_reasons.csv')
        self.positive_attendance_reasons = self._load_reasons_from_csv('positive_attendance_reasons.csv')
        self.patient_notes_templates = self._load_reasons_from_csv('patient_notes_templates.csv', 'note')
        self.patient_sentiment_templates = self._load_reasons_from_csv('patient_sentiment_templates.csv', 'sentiment')
        self.age_based_reasons = self._load_age_based_reasons()
        self.gender_based_reasons = self._load_gender_based_reasons()
        self.sms_based_reasons = self._load_sms_based_reasons()
        self.rule_engine = self._load_rule_engine()
        self.clinical_terms_df = self._load_clinical_terms_rules()
        self.general_no_show_reasons = general_no_show_reasons or self.no_show_reasons

    # --- Data Loading ---
    def _load_reasons_from_csv(self, filename, colname='reason'):
        path = os.path.join(self.synthetic_dir, filename)
        df = pd.read_csv(path)
        return df[colname].dropna().tolist()

    def _load_age_based_reasons(self):
        path = os.path.join(self.synthetic_dir, 'age_based_reasons.csv')
        return pd.read_csv(path)

    def _load_gender_based_reasons(self):
        path = os.path.join(self.synthetic_dir, 'gender_based_reasons.csv')
        return pd.read_csv(path)

    def _load_sms_based_reasons(self):
        path = os.path.join(self.synthetic_dir, 'sms_based_reasons.csv')
        return pd.read_csv(path)

    def _load_rule_engine(self):
        path = os.path.join(self.synthetic_dir, 'rule_engine.csv')
        return pd.read_csv(path)

    def _load_clinical_terms_rules(self):
        path = os.path.join(self.synthetic_dir, 'clinical_terms_rules.csv')
        return pd.read_csv(path)

    # --- Rule Engine ---
    def _apply_rule_engine(self, row, template_type):
        matches = []
        for _, rule in self.rule_engine.iterrows():
            if rule['template_type'] != template_type:
                continue
            context = {
                'age': row.get('Age', 0),
                'gender': f"'{str(row.get('Gender', '')).strip().upper()}'",
                'sms_received': row.get('SMS_received', 0),
                'no_show': f"'{str(row.get('No-show', '')).strip().lower()}'",
                'diabetes': row.get('Diabetes', 0),
                'hypertension': row.get('Hypertension', 0),
                'alcoholism': row.get('Alcoholism', 0),
                'handcap': row.get('Handcap', 0),
                'scholarship': row.get('Scholarship', 0)
            }
            try:
                if rule['condition'] and eval(rule['condition'], {}, context):
                    matches.append(rule['template_ref'])
            except Exception:
                continue
        return matches

    # --- Simulation Main Entry ---
    def simulate(self, input_csv, output_csv, notes_col='PatientNotes', sentiment_col='PatientSentiment', reason_col='NoShowReason', parallel=True, n_jobs=-1):
        df = pd.read_csv(input_csv)
        if parallel:
            from joblib import Parallel, delayed
            def process_row(row):
                return (
                    self._generate_patient_notes(row),
                    self._generate_patient_sentiment(row),
                    self._generate_no_show_reason(row)
                )
            results = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(process_row)(row) for _, row in df.iterrows()
            )
            notes, sentiments, reasons = zip(*results)
            df[notes_col] = notes
            df[sentiment_col] = sentiments
            df[reason_col] = reasons
        else:
            df[notes_col] = df.apply(self._generate_patient_notes, axis=1)
            df[sentiment_col] = df.apply(self._generate_patient_sentiment, axis=1)
            df[reason_col] = df.apply(self._generate_no_show_reason, axis=1)
        df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
        return df

    # --- No Show Reason Generation ---
    def _generate_no_show_reason(self, row):
        template_type = 'no_show_reason' if str(row['No-show']).strip().lower() == 'yes' else 'positive_attendance_reason'
        matches = self._apply_rule_engine(row, template_type)
        ref_pools = {}
        for ref in set(matches):
            if ref in ['child', 'elderly', 'adult']:
                ref_pools[ref] = self.age_based_reasons[self.age_based_reasons['age_group'] == ref]['reason'].dropna().tolist()
            elif ref in ['F', 'M']:
                ref_pools[ref] = self.gender_based_reasons[self.gender_based_reasons['gender'] == ref]['reason'].dropna().tolist()
            elif ref in ['yes', 'no']:
                ref_pools[ref] = self.sms_based_reasons[self.sms_based_reasons['no_show'] == ref]['sms_reason'].dropna().tolist()
            elif ref:
                ref_str = str(ref) if not isinstance(ref, str) else ref
                ref_pools[ref] = [r for r in self.no_show_reasons if isinstance(r, str) and ref_str.lower() in r.lower()]
        if template_type == 'no_show_reason':
            reasons = set()
            used_refs = set()
            for ref in matches:
                if ref and ref not in used_refs:
                    pool = ref_pools.get(ref, [])
                    if pool:
                        sampled = random.sample(pool, min(2, len(pool)))
                        reasons.update(sampled)
                        used_refs.add(ref)
                if len(reasons) >= 4:
                    break
            if not reasons:
                reasons = set(random.sample([r for r in self.no_show_reasons if isinstance(r, str)], min(2, len([r for r in self.no_show_reasons if isinstance(r, str)]))))
            return " ".join(random.sample(list(reasons), min(3, len(reasons))))
        else:
            return " ".join(random.sample(self.positive_attendance_reasons, min(2, len(self.positive_attendance_reasons))))

    # --- Patient Notes Generation ---
    def _generate_patient_notes(self, row):
        matches = self._apply_rule_engine(row, 'patient_notes')
        clinical_terms_df = self.clinical_terms_df
        # Avoid chained indexing warning by filtering stepwise
        diabetes_terms = clinical_terms_df[clinical_terms_df['category'] == 'PROBLEM']
        diabetes_terms = diabetes_terms[diabetes_terms['term'].str.contains('diabet', case=False)]['term'].str.lower().tolist()
        hypertension_terms = clinical_terms_df[clinical_terms_df['category'] == 'PROBLEM']
        hypertension_terms = hypertension_terms[hypertension_terms['term'].str.contains('hyperten|htn|blood pressure', case=False)]['term'].str.lower().tolist()
        alcoholism_terms = clinical_terms_df[clinical_terms_df['category'] == 'PROBLEM']
        alcoholism_terms = alcoholism_terms[alcoholism_terms['term'].str.contains('alcohol', case=False)]['term'].str.lower().tolist()
        medication_terms = clinical_terms_df[clinical_terms_df['category'] == 'MEDICATION']['term'].str.lower().tolist()
        procedure_terms = clinical_terms_df[clinical_terms_df['category'].isin(['PROCEDURE', 'TEST'])]['term'].str.lower().tolist()
        ref_templates = {}
        for ref in set(matches):
            ref_templates[ref] = [n for n in self.patient_notes_templates if ref.lower() in n.lower()]
        for term in diabetes_terms + hypertension_terms + alcoholism_terms + medication_terms + procedure_terms:
            ref_templates[term] = [n for n in self.patient_notes_templates if term in n.lower()]
        notes = set()
        for ref in matches:
            filtered = ref_templates.get(ref, [])
            if filtered:
                sampled = random.sample(filtered, min(2, len(filtered)))
                notes.update(sampled)
        for term in diabetes_terms:
            if str(row.get('Diabetes', 0)) == '1' or term in str(row.get('PatientNotes', '')).lower():
                filtered = ref_templates.get(term, [])
                if filtered:
                    notes.update(random.sample(filtered, min(1, len(filtered))))
        for term in hypertension_terms:
            if str(row.get('Hypertension', 0)) == '1' or term in str(row.get('PatientNotes', '')).lower():
                filtered = ref_templates.get(term, [])
                if filtered:
                    notes.update(random.sample(filtered, min(1, len(filtered))))
        for term in alcoholism_terms:
            if str(row.get('Alcoholism', 0)) == '1' or term in str(row.get('PatientNotes', '')).lower():
                filtered = ref_templates.get(term, [])
                if filtered:
                    notes.update(random.sample(filtered, min(1, len(filtered))))
        if not notes:
            fallback = [n for n in self.patient_notes_templates if 'no chronic conditions' in n.lower() or 'general health checkup' in n.lower()]
            notes.update(random.sample(fallback, min(2, len(fallback))))
        return " ".join(random.sample(list(notes), min(3, len(notes))))

    # --- Patient Sentiment Generation ---
    def _generate_patient_sentiment(self, row):
        matches = self._apply_rule_engine(row, 'patient_sentiment')
        ref_templates = {}
        for ref in set(matches):
            ref_templates[ref] = [s for s in self.patient_sentiment_templates if ref.lower() in s.lower()]
        sentiments = set()
        for ref in matches:
            filtered = ref_templates.get(ref, [])
            if filtered:
                sampled = random.sample(filtered, min(2, len(filtered)))
                sentiments.update(sampled)
        if not sentiments:
            fallback = [s for s in self.patient_sentiment_templates if 'optimistic' in s.lower() or 'hopeful' in s.lower() or 'positive' in s.lower()]
            sentiments.update(random.sample(fallback, min(2, len(fallback))))
        return " ".join(random.sample(list(sentiments), min(3, len(sentiments))))
