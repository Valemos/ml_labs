import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans


class TransformerWithoutFit(TransformerMixin, BaseEstimator):
    def fit(self, df, y=None, **params):
        return self


class PrimaryTransformer(TransformerWithoutFit):
    unnecessary_columns = ["SK_ID_CURR", "NAME_TYPE_SUITE", "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
                           "DAYS_ID_PUBLISH", "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START",
                           "REGION_RATING_CLIENT", "DAYS_LAST_PHONE_CHANGE", "OBS_30_CNT_SOCIAL_CIRCLE",
                           "DEF_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "FLAG_MOBIL", "FLAG_EMP_PHONE",
                           "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL", "FLAG_CONT_MOBILE"]
    feat_categorical = {"TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
                        "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
                        "OCCUPATION_TYPE", "REGION_RATING_CLIENT_W_CITY", "REG_REGION_NOT_LIVE_REGION",
                        "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY",
                        "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY", "ORGANIZATION_TYPE"}
    feat_high_correlating = ["CNT_CHILDREN", "AMT_GOODS_PRICE", "AMT_ANNUITY"]

    def transform(self, df, **transform_params):
        df.drop(columns=self.unnecessary_columns, inplace=True)
        df.drop(columns=self.feat_high_correlating, inplace=True)
        df.drop(columns=[i for i in df.columns if i.startswith("FLAG_DOCUMENT_")], inplace=True)

        for feature in self.feat_categorical:
            df[feature] = df[feature].astype("category")
        return df


class FillNA(TransformerWithoutFit):

    def transform(self, df, **params):
        df["CNT_FAM_MEMBERS"].fillna(0, inplace=True)
        return df


class DropExternalSourcesColumns(TransformerWithoutFit):

    def transform(self, df, **params):
        df.drop(columns=[col for col in df.columns if col.startswith("EXT_SOURCE_")], inplace=True)
        return df


class RequestsColumnTransformer(TransformerWithoutFit):
    requests_columns = ["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
                        "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"]
    new_requests_col = "AMT_REQ_CREDIT_BUREAU"

    def transform(self, df, **transform_params):
        df["AMT_REQ_CREDIT_BUREAU"] = np.sum(df[self.requests_columns], axis=1)
        df.drop(columns=self.requests_columns, inplace=True)
        return df


class BuildingColumnsTransformer(TransformerWithoutFit):
    building_columns = ["APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG",
                        "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG",
                        "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG",
                        "NONLIVINGAREA_AVG", "APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE",
                        "YEARS_BUILD_MODE", "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE", "FLOORSMAX_MODE",
                        "FLOORSMIN_MODE", "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE",
                        "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE", "APARTMENTS_MEDI", "BASEMENTAREA_MEDI",
                        "YEARS_BEGINEXPLUATATION_MEDI", "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI",
                        "ENTRANCES_MEDI", "FLOORSMAX_MEDI", "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI",
                        "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI", "FONDKAPREMONT_MODE",
                        "HOUSETYPE_MODE", "TOTALAREA_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"]

    def transform(self, df, **params):
        df["BUILDING_INFO_AVAILABLE"] = ~df[self.building_columns].isnull().any(axis=1)
        df.drop(columns=self.building_columns, inplace=True)
        return df


class CarOwnAgeTransformer(TransformerWithoutFit):
    flag_own_car = "FLAG_OWN_CAR"
    own_car_age = "OWN_CAR_AGE"

    def transform(self, df, **params):
        df[self.flag_own_car] = ~df[self.own_car_age].isnull()
        df[self.flag_own_car] = df[self.flag_own_car].astype("category")
        df.drop(columns=[self.own_car_age], inplace=True)
        return df


class DropOutliers(TransformerWithoutFit):

    def transform(self, df, **params):
        df = self.drop_upper_percentile(df, "AMT_INCOME_TOTAL")
        df = self.drop_upper_percentile(df, "AMT_CREDIT")
        df.drop(df[df["CODE_GENDER"] == "XNA"].index, inplace=True)
        return df

    @staticmethod
    def drop_upper_percentile(df: pd.DataFrame, feature):
        upper_border = df[feature].describe()["75%"]
        return df.drop(df[df[feature] > upper_border].index, axis="index")


class EncodeBinaryCategoricalValues(TransformerWithoutFit):

    def transform(self, df, **params):
        categorical_features = df.select_dtypes("category")

        for feature in categorical_features.columns:
            if len(df[feature].unique()) == 2:
                df[feature] = df[feature].cat.codes
        return df


class ClusterizeIncomeTotal(BaseEstimator, TransformerMixin):
    feature_income_total = "AMT_INCOME_TOTAL"

    def __init__(self):
        initial_values = np.array([[50000], [90000], [110000], [130000], [160000], [180000], [200000]])
        self.kmeans = KMeans(n_clusters=initial_values.shape[0],
                             init=initial_values, n_init=1)

    def fit(self, df, y=None, **params):
        self.kmeans.fit(np.asarray(df[self.feature_income_total]).reshape(-1, 1))
        return self

    def transform(self, df, **params):
        df[self.feature_income_total] = self.to_clusters(df)
        df[self.feature_income_total] = df[self.feature_income_total].astype("int8")
        return df

    def to_clusters(self, df):
        return self.kmeans.predict(np.asarray(df[self.feature_income_total]).reshape(-1, 1))


class EncodeOtherCategoricalValues(TransformerWithoutFit):

    def transform(self, df: pd.DataFrame, **params):
        categorical_features = df.select_dtypes("category")
        dummies = pd.get_dummies(categorical_features)
        return pd.concat([df.drop(columns=categorical_features.columns), dummies], axis=1)


data_prepare_pipeline = make_pipeline(
    PrimaryTransformer(),
    FillNA(),
    DropExternalSourcesColumns(),
    CarOwnAgeTransformer(),
    RequestsColumnTransformer(),
    BuildingColumnsTransformer(),
    DropOutliers(),
    ClusterizeIncomeTotal(),
    'passthrough'
)

data_train_pipeline = make_pipeline(
    data_prepare_pipeline,
    EncodeBinaryCategoricalValues(),
    EncodeOtherCategoricalValues()
)