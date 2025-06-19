import medspacy
import pandas as pd
from medspacy.context import ConText
from medspacy.context.context_rule import ConTextRule
from medspacy.target_matcher import TargetRule
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
import joblib
import os
from medspacy.visualization import visualize_ent
import tempfile
import json

from src.config import RANDOM_STATE

class ClinicalTopicModel:
    def __init__(self, config):
        self.config = config
        self.nlp = self._configure_medspacy()
        self.vectorizer = None
        self.model = None
        self.best_params = None

    def _configure_medspacy(self):
        nlp = medspacy.load()
        
        # Load clinical terms and context rules from config path
        clinical_terms_path = self.config.CLINICAL_TERMS_RULES_PATH
        clinical_terms_df = pd.read_csv(clinical_terms_path)
        
        # Separate context rules from clinical terms
        context_rules = clinical_terms_df[clinical_terms_df['category'] == 'CONTEXT'].copy()
        clinical_terms = clinical_terms_df[clinical_terms_df['category'] != 'CONTEXT'].copy()
        
        # Prepare context rules for ConText
        context_rule_list = []
        for _, rule in context_rules.iterrows():
            if pd.notnull(rule['context_type']) and pd.notnull(rule['pattern']) and pd.notnull(rule['direction']):
                context_rule_list.append({
                    "category": rule['context_type'],
                    "literal": rule['term'],
                    "direction": rule['direction'],
                    "pattern": rule['pattern']
                })
        # Write context rules to a temp JSON file if any custom rules exist
        context_rules_path = None
        if context_rule_list:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w', encoding='utf-8')
            json.dump({"context_rules": context_rule_list}, tmp)
            tmp.close()
            context_rules_path = tmp.name
        # Add ConText to pipeline by name (spaCy v3+), only if not already present
        if "medspacy_context" not in nlp.pipe_names:
            nlp.add_pipe("medspacy_context", last=True)
        if context_rules_path:
            context = nlp.get_pipe("medspacy_context")
            with open(context_rules_path, 'r', encoding='utf-8') as f:
                rule_data = json.load(f)
                for r in rule_data["context_rules"]:
                    context.add(ConTextRule.from_dict(r))
        
        # Add clinical terms to target matcher
        matcher = nlp.get_pipe("medspacy_target_matcher")
        for _, row in clinical_terms.iterrows():
            matcher.add(TargetRule(row['term'], row['category']))
        
        return nlp

    def extract_clinical_concepts(self, text):
        doc = self.nlp(text)
        concepts = []
        for ent in doc.ents:
            concept = ent.text.lower().replace(' ', '_')
            
            # Check for negation and assertion context
            if hasattr(ent._, 'is_negated') and ent._.is_negated:
                concept = f"negated_{concept}"
            elif hasattr(ent._, 'is_uncertain') and ent._.is_uncertain:
                concept = f"uncertain_{concept}"
            elif hasattr(ent._, 'is_possible') and ent._.is_possible:
                concept = f"possible_{concept}"
                
            concepts.append(concept)
        
        return ' '.join(concepts)

    def preprocess_notes(self, df, keyword):
        df_cond = df[df['PatientNotes'].str.contains(keyword, case=False, na=False)].copy()
        df_cond['PatientNotes_clean'] = df_cond['PatientNotes'].apply(lambda x: self.extract_clinical_concepts(str(x)))
        df_cond = df_cond[df_cond['PatientNotes_clean'].str.strip().astype(bool)]
        return df_cond

    def vectorize_notes(self, notes):
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        X = self.vectorizer.fit_transform(notes)
        return X

    def train(self, notes, n_topics_range=(2, 6)):
        X = self.vectorize_notes(notes)
        # Use n_jobs=-1 for parallelism, reduce param grid for speed
        search_params = {'n_components': list(range(n_topics_range[0], n_topics_range[1]+1)), 'learning_decay': [0.7]}  # Only one decay value for speed
        lda = LatentDirichletAllocation(random_state=RANDOM_STATE, learning_method='batch', batch_size=256, n_jobs=-1)
        model = GridSearchCV(lda, param_grid=search_params, cv=2, n_jobs=-1, verbose=1)
        model.fit(X)
        self.model = model.best_estimator_
        self.best_params = model.best_params_
        return self.model, self.best_params

    def evaluate(self, notes):
        X = self.vectorizer.transform(notes)
        perplexity = self.model.perplexity(X)
        topic_assignments = self.model.transform(X).argmax(axis=1)
        sil_score = None
        if X.shape[0] > 1 and self.model.n_components > 1:
            sil_score = silhouette_score(X, topic_assignments)
        return perplexity, sil_score

    def export(self, path_prefix):
        os.makedirs(path_prefix, exist_ok=True)
        joblib.dump(self.model, os.path.join(path_prefix, "lda_model.joblib"))
        joblib.dump(self.vectorizer, os.path.join(path_prefix, "vectorizer.joblib"))

    def get_topics(self, n_top_words=10):
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for idx, topic in enumerate(self.model.components_):
            topics.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        return topics

    def plot_medspacy_ents(self, text):
        doc = self.nlp(text)
        visualize_ent(doc)
        
    def analyze_context_attributes(self, text):

        doc = self.nlp(text)
        entity_info = []
        
        for ent in doc.ents:
            info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            }
            
            # Check context attributes
            if hasattr(ent._, 'is_negated'):
                info['is_negated'] = ent._.is_negated
            if hasattr(ent._, 'is_uncertain'):
                info['is_uncertain'] = ent._.is_uncertain
            if hasattr(ent._, 'is_possible'):
                info['is_possible'] = ent._.is_possible
            if hasattr(ent._, 'is_historical'):
                info['is_historical'] = ent._.is_historical
            if hasattr(ent._, 'is_hypothetical'):
                info['is_hypothetical'] = ent._.is_hypothetical
            if hasattr(ent._, 'is_family'):
                info['is_family'] = ent._.is_family
                
            entity_info.append(info)
            
        return entity_info
