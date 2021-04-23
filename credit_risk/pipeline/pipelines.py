from .transform_pipeline_parts import *


def make_prepare_pipeline():
    return make_pipeline(
            DropUnnecessaryColumns(),
            DropColumns(RequestsColumnTransformer.requests_columns),
            DropColumns(["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]),
            CategoryTypeTransformer(),
            ReplaceRedundantCategories(), 
            ReplaceUnknownCategories("CODE_GENDER", "M"),
            ReplaceUnknownCategories("REGION_RATING_CLIENT_W_CITY", 1),
            ReplaceUnknownCategories("NAME_INCOME_TYPE", 1),
            BuildingColumnsTransformer(),
            CarOwnAgeTransformer(),
            FillNA(["CNT_FAM_MEMBERS"], "constant", fill_value=0),
            'passthrough'
        )

def make_encode_pipeline():
    return make_pipeline(
            EncodeCategoricalValues(),
            'passthrough'
        )

def make_train_pipeline():
    return make_pipeline(
            make_prepare_pipeline(),
            make_encode_pipeline(),
            DropColumns(["SK_ID_CURR"]),
            DropRowsByBuilding()
        )

def get_predict_pipeline(train_pipeline):
    return make_pipeline(
            train_pipeline.named_steps["pipeline-1"],
            train_pipeline.named_steps["pipeline-2"]
        )
