import medspacy
import pandas as pd
from medspacy.target_matcher import TargetRule
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
import joblib

class ClinicalTopicModel:
    def __init__(self, config):
        self.config = config
        self.nlp = self._configure_medspacy()
        self.vectorizer = None
        self.model = None
        self.best_params = None

    def _configure_medspacy(self):
        nlp = medspacy.load()
        # Load clinical terms from config path
        clinical_terms_path = self.config.CLINICAL_TERMS_RULES_PATH
        clinical_terms_df = pd.read_csv(clinical_terms_path)
        matcher = nlp.get_pipe("medspacy_target_matcher")
        for _, row in clinical_terms_df.iterrows():
            matcher.add(TargetRule(row['term'], row['category']))
        return nlp

    def extract_clinical_concepts(self, text):
        doc = self.nlp(text)
        return ' '.join([ent.text.lower().replace(' ', '_') for ent in doc.ents])

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
        lda = LatentDirichletAllocation(random_state=42, learning_method='batch', batch_size=256, n_jobs=-1)
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
        joblib.dump(self.model, f"{path_prefix}_lda_model.joblib")
        joblib.dump(self.vectorizer, f"{path_prefix}_vectorizer.joblib")

    def get_topics(self, n_top_words=10):
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for idx, topic in enumerate(self.model.components_):
            topics.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        return topics
