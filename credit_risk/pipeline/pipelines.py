from .transform_pipeline_parts import *


data_prepare_pipeline = make_pipeline(
    DropUnnecessary(),
    DropColumns(RequestsColumnTransformer.requests_columns),
    TypeTransformer(),
    BuildingColumnsTransformer(),
    FillNA(BuildingColumnsTransformer.columns_trimmed, "mean"),
    'passthrough'
)

data_train_pipeline = make_pipeline(
    data_prepare_pipeline,
    EncodeBinaryCategoricalValues(),
    EncodeOtherCategoricalValues()
)
