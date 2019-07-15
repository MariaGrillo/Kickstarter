import json
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

SELECTED_NAME = ['Los Angeles', 'New York', 'London', 'Chicago', 'Brooklyn', 'San Francisco', 'Portland', 'Seattle',
                 'Austin', 'Atlanta', 'Boston', 'Nashville', 'Philadelphia', 'San Diego', 'Dallas', 'Washington',
                 'Denver', 'Minneapolis', 'Toronto', 'Houston', 'Las Vegas', 'Phoenix', 'Orlando', 'Salt Lake City',
                 'Vancouver']

SELECTED_STATE = ['CA', 'NY', 'England', 'TX', 'FL', 'IL', 'WA', 'PA', 'GA', 'MA', 'OH', 'OR', 'MI', 'TN', 'NC', 'CO',
                  'AZ', 'ON', 'VA', 'MN', 'UT', 'NJ', 'MO', 'MD', 'WI']

SELECTED_COUNTRY = ['US', 'GB', 'CA']


class CountryTransformer(BaseEstimator, TransformerMixin):
    """Transform countries into larger groups to avoid having
    too many dummies."""

    countries = {
        'US': 'US',
        'CA': 'Canada',
        'GB': 'UK & Ireland',
        'AU': 'Oceania',
        'IE': 'UK & Ireland',
        'SE': 'Europe',
        'CH': "Europe",
        'IT': 'Europe',
        'FR': 'Europe',
        'NZ': 'Oceania',
        'DE': 'Europe',
        'NL': 'Europe',
        'NO': 'Europe',
        'MX': 'Other',
        'ES': 'Europe',
        'DK': 'Europe',
        'BE': 'Europe',
        'AT': 'Europe',
        'HK': 'Other',
        'SG': 'Other',
        'LU': 'Europe'
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame({"country": X.country.map(self.countries)})


class KickstarterModel:

    # Update parameters here after re-tuning the model
    params = {"min_samples_split": 70,
              "n_estimators": 50}

    def __init__(self):

        country_processor = Pipeline([("transfomer", CountryTransformer()),
                                      ("one_hot", OneHotEncoder(sparse=False, handle_unknown="ignore"))])

        self.model = RandomForestClassifier(**self.params)
        self.preproc = make_column_transformer(
            (OneHotEncoder(categories='auto'), ['location.type', 'category.id', 'subcategory.id']),
            (StandardScaler(), ['name', 'blurb', 'slug', 'log_goal_usd', 'deadline-created_at',
                                'deadline-launched_at', 'launched_at-created_at']),
            (country_processor, ["country"])
            )

    def json_helper(self, x):
        if type(x) == str:
            return json.loads(x)
        else:
            return {}

    def get_category(self, x):
        temp_json = self.json_helper(x)
        category = temp_json.get("parent_id")
        sub_category = temp_json.get("position")
        return category, sub_category

    def get_location(self, x):
        temp_json = self.json_helper(x)
        name = temp_json.get("name")
        state = temp_json.get("state")
        l_type = temp_json.get("type")
        return name, state, l_type

    def preprocess_common(self, df):
        """Method used by both preprocess_training_data
        and preprocess_unseen_data"""

        X = pd.DataFrame()

        X['name'] = df['name'].fillna('').apply(lambda x: len(x))
        X['blurb'] = df['blurb'].fillna('').apply(lambda x: len(x))
        X['slug'] = df['slug'].fillna('').apply(lambda x: len(x))

        df['adjusted_goal'] = df.goal * df.static_usd_rate
        X['log_goal_usd'] = np.log(df['adjusted_goal'])

        cat_tuple_list = df['category'].apply(self.get_category)
        X['category.id'] = [cat for(cat, sub) in cat_tuple_list]
        X['subcategory.id'] = [sub for (cat, sub) in cat_tuple_list]

        loc_tuple_list = df['location'].apply(self.get_location)
        X['location.name'] = [name for(name, state, l_type) in loc_tuple_list]
        X['location.state'] = [state for (name, state, l_type) in loc_tuple_list]
        X['location.type'] = [l_type for (name, state, l_type) in loc_tuple_list]

        X['location.type'] = X['location.type'].fillna('')

        X['country'] = df['country'].fillna('')

        X['deadline-created_at'] = df['deadline'] - df['created_at']
        X['deadline-launched_at'] = df['deadline'] - df['launched_at']
        X['launched_at-created_at'] = df['launched_at'] - df['created_at']

        for name in SELECTED_NAME:
            X['location_name_{}'.format(name)] = X['location.name'].apply(lambda x: 1 if x == name else 0)

        for state in SELECTED_STATE:
            X['location_state_{}'.format(state)] = X['location.state'].apply(lambda x: 1 if x == state else 0)

        X = X.drop(['location.name', 'location.state'], axis=1)

        return X

    def preprocess_training_data(self, df):

        X = self.preprocess_common(df)
        print(X.isna().any())
        X = self.preproc.fit_transform(X)

        state_transform_func = lambda x: 1 if x == 'successful' else 0
        y = df['state'].apply(state_transform_func)

        return X, y

    def fit(self, X, y):
        self.model.fit(X, y)

    def preprocess_unseen_data(self, df):

        X = self.preprocess_common(df)
        return self.preproc.transform(X)

    def predict(self, X):

        return self.model.predict(X)
