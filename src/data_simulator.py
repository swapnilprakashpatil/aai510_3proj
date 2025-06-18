# data_simulator.py

import pandas as pd
import random
import os
import csv
import numpy as np
from functools import lru_cache
from collections import defaultdict

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
        
        # Performance optimizations - precompute expensive operations
        self._precompute_templates()
        self._compile_rules()

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

    # --- Performance Optimizations ---
    def _precompute_templates(self):
        
        # Define emotion keyword mappings for better template matching
        self.emotion_keywords = {
            'confusion': ['confusion', 'confused', 'uncertain', 'unclear', 'puzzled', 'instructions', 'understand'],
            'anxiety': ['anxiety', 'anxious', 'worried', 'nervous', 'concerned', 'stress'],
            'stress': ['stress', 'stressed', 'overwhelmed', 'pressure', 'burden', 'demands'],
            'fear': ['fear', 'afraid', 'scared', 'frightened', 'terrified', 'panic'],
            'hopeful': ['hopeful', 'hope', 'optimistic', 'positive', 'confident', 'encouraged']
        }
        
        # Precompute lowercased templates to avoid repeated .lower() calls
        self.sentiment_templates_lower = [s.lower() for s in self.patient_sentiment_templates]
        
        # Precompute emotion template mappings
        self.emotion_template_cache = {}
        for emotion, keywords in self.emotion_keywords.items():
            self.emotion_template_cache[emotion] = [
                s for i, s in enumerate(self.patient_sentiment_templates)
                if any(keyword in self.sentiment_templates_lower[i] for keyword in keywords)
            ]
        
        # Precompute fallback emotion templates
        self.fallback_emotion_templates = []
        for emotion, keywords in [
            ('confusion', ['confusion', 'confused', 'uncertain']),
            ('anxiety', ['anxiety', 'anxious', 'worried']),
            ('stress', ['stress', 'stressed', 'overwhelmed']),
            ('fear', ['fear', 'afraid', 'scared']),
            ('hopeful', ['hopeful', 'hope', 'optimistic', 'positive'])
        ]:
            emotion_templates = [
                s for i, s in enumerate(self.patient_sentiment_templates)
                if any(keyword in self.sentiment_templates_lower[i] for keyword in keywords)
            ]
            self.fallback_emotion_templates.extend(emotion_templates)
    
    def _compile_rules(self):

        self.compiled_rules = defaultdict(list)
        
        for _, rule in self.rule_engine.iterrows():
            template_type = rule['template_type']
            condition = rule['condition']
            template_ref = rule['template_ref']
            
            # Skip empty conditions
            if pd.isna(condition) or condition.strip() == '':
                continue
                
            try:
                # Convert condition to a compiled code object for faster evaluation
                compiled_condition = compile(condition, '<string>', 'eval')
                self.compiled_rules[template_type].append({
                    'condition': compiled_condition,
                    'template_ref': template_ref,
                    'original_condition': condition  # Keep for debugging
                })
            except SyntaxError:
                # Skip invalid conditions
                continue

    # --- Optimized Rule Engine ---
    def _apply_rule_engine(self, row, template_type):

        matches = []
        
        # Prepare context once
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
        
        # Use precompiled rules for faster evaluation
        for rule in self.compiled_rules.get(template_type, []):
            try:
                if eval(rule['condition'], {}, context):
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
                    self._generate_patient_sentiment_optimized(row),
                    self._generate_no_show_reason(row)
                )
            
            # Use vectorized operations for better performance
            print(f"Processing {len(df)} rows in parallel with {n_jobs} jobs...")
            results = Parallel(n_jobs=n_jobs, prefer='threads', batch_size='auto')(
                delayed(process_row)(row) for _, row in df.iterrows()
            )
            
            notes, sentiments, reasons = zip(*results)
            df[notes_col] = notes
            df[sentiment_col] = sentiments
            df[reason_col] = reasons
        else:
            print(f"Processing {len(df)} rows sequentially...")
            df[notes_col] = df.apply(self._generate_patient_notes, axis=1)
            df[sentiment_col] = df.apply(self._generate_patient_sentiment_optimized, axis=1)
            df[reason_col] = df.apply(self._generate_no_show_reason, axis=1)
        
        df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
        print(f"Data simulation completed. Output saved to {output_csv}")
        return df

    # --- No Show Reason Generation ---
    def _generate_no_show_reason(self, row):
        template_type = 'no_show_reason' if str(row['No-show']).strip().lower() == 'yes' else 'positive_attendance_reason'
        matches = self._apply_rule_engine(row, template_type)
        
        ref_templates = {}
        for ref in set(matches):
            if ref == 'diabetes':
                ref_templates[ref] = [r for r in self.general_no_show_reasons if 'diabetes' in r.lower()]
            elif ref == 'hypertension':
                ref_templates[ref] = [r for r in self.general_no_show_reasons if 'hypertension' in r.lower() or 'pressure' in r.lower()]
            elif ref == 'alcohol':
                ref_templates[ref] = [r for r in self.general_no_show_reasons if 'alcohol' in r.lower()]
            elif ref == 'handicap':
                ref_templates[ref] = [r for r in self.general_no_show_reasons if 'mobility' in r.lower() or 'handicap' in r.lower()]
            elif ref == 'yes':
                ref_templates[ref] = [r for r in self.sms_based_reasons['reason'].tolist() if 'sms' in r.lower() or 'reminder' in r.lower()]
            elif ref == 'child':
                ref_templates[ref] = [r for r in self.age_based_reasons[self.age_based_reasons['age_group'] == 'child']['reason'].tolist()]
            elif ref == 'elderly':
                ref_templates[ref] = [r for r in self.age_based_reasons[self.age_based_reasons['age_group'] == 'elderly']['reason'].tolist()]
            elif ref == 'F':
                ref_templates[ref] = [r for r in self.gender_based_reasons[self.gender_based_reasons['gender'] == 'F']['reason'].tolist()]
            elif ref == 'M':
                ref_templates[ref] = [r for r in self.gender_based_reasons[self.gender_based_reasons['gender'] == 'M']['reason'].tolist()]
            else:
                if template_type == 'no_show_reason':
                    ref_templates[ref] = self.general_no_show_reasons
                else:
                    ref_templates[ref] = self.positive_attendance_reasons
        
        reasons = set()
        for ref in matches:
            filtered = ref_templates.get(ref, [])
            if filtered:
                sampled = random.sample(filtered, min(1, len(filtered)))
                reasons.update(sampled)
        
        if not reasons:
            fallback = self.general_no_show_reasons if template_type == 'no_show_reason' else self.positive_attendance_reasons
            reasons.update(random.sample(fallback, min(1, len(fallback))))
        
        return " ".join(random.sample(list(reasons), min(2, len(reasons))))

    # --- Patient Notes Generation ---
    def _generate_patient_notes(self, row):
        matches = self._apply_rule_engine(row, 'patient_notes')
        ref_templates = {}
        for ref in set(matches):
            ref_templates[ref] = [n for n in self.patient_notes_templates if ref.lower() in n.lower()]
        
        notes = set()
        for ref in matches:
            filtered = ref_templates.get(ref, [])
            if filtered:
                sampled = random.sample(filtered, min(2, len(filtered)))
                notes.update(sampled)
        
        if not notes:
            fallback = [n for n in self.patient_notes_templates if 'routine' in n.lower() or 'checkup' in n.lower()]
            notes.update(random.sample(fallback, min(2, len(fallback))))
        
        return " ".join(random.sample(list(notes), min(3, len(notes))))

    # --- Optimized Patient Sentiment Generation ---
    def _generate_patient_sentiment_optimized(self, row):

        matches = self._apply_rule_engine(row, 'patient_sentiment')
        
        sentiments = set()
        for ref in matches:
            if ref in self.emotion_template_cache:
                # Use precomputed templates
                filtered = self.emotion_template_cache[ref]
                if filtered:
                    sampled = random.sample(filtered, min(2, len(filtered)))
                    sentiments.update(sampled)
            else:
                # Fallback to original matching for non-emotion refs
                filtered = [s for s in self.patient_sentiment_templates if ref.lower() in s.lower()]
                if filtered:
                    sampled = random.sample(filtered, min(2, len(filtered)))
                    sentiments.update(sampled)
        
        if not sentiments:
            # Use precomputed fallback templates
            if self.fallback_emotion_templates:
                sentiments.update(random.sample(self.fallback_emotion_templates, min(2, len(self.fallback_emotion_templates))))
            else:
                # Ultimate fallback
                sentiments.update(random.sample(self.patient_sentiment_templates, min(2, len(self.patient_sentiment_templates))))
                
        return " ".join(random.sample(list(sentiments), min(3, len(sentiments))))

    # --- Legacy Method for Backward Compatibility ---
    def _generate_patient_sentiment(self, row):

        return self._generate_patient_sentiment_optimized(row)
