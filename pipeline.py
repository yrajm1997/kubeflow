from kfp.dsl import component, pipeline, Input, Output, Artifact
import kfp

@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def prepare_data(data_path: str, final_df: Output[Artifact]):
    import pandas as pd
    from sklearn import datasets

    # Load dataset
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target

    df = df.dropna()
    df.to_csv(final_df.path, index=False)


@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def train_test_splitting(final_df: Input[Artifact], data_path: str, X_train: Output[Artifact], X_test: Output[Artifact], y_train: Output[Artifact], y_test: Output[Artifact]):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    final_data = pd.read_csv(final_df.path)

    target_column = 'species'
    X = final_data.loc[:, final_data.columns != target_column]
    y = final_data.loc[:, final_data.columns == target_column]

    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X, y, test_size=0.3, stratify=y, random_state=47)

    X_train_data.to_csv(X_train.path, index=False)
    X_test_data.to_csv(X_test.path, index=False)
    y_train_data.to_csv(y_train.path, index=False)
    y_test_data.to_csv(y_test.path, index=False)


@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def training_basic_classifier(X_train: Input[Artifact], y_train: Input[Artifact], save_model: Output[Artifact]):
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    import joblib

    X_train_data = pd.read_csv(X_train.path)
    y_train_data = pd.read_csv(y_train.path)

    classifier = LogisticRegression(max_iter=500)
    classifier.fit(X_train_data, y_train_data)
    
    joblib.dump(classifier, save_model.path)


@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def inference_with_model(model_path: Input[Artifact]):
    import pandas as pd
    import numpy as np
    # from sklearn.linear_model import LogisticRegression
    import joblib
    
    model = joblib.load(model_path.path)
    input_sample = np.array([[2.5, 3.5, 4.5, 5.5]])
    
    pred = model.predict(input_sample)
    print(pred)



@pipeline(
    name="Basic ML Pipeline",
    description="A simple pipeline for data preparation, training/test splitting, and model training."
)
def basic_ml_pipeline(data_path: str = '/data'):
    prepare_data_task = prepare_data(data_path=data_path)
    train_test_split_task = train_test_splitting(final_df=prepare_data_task.outputs['final_df'], data_path=data_path)
    training_task = training_basic_classifier(X_train=train_test_split_task.outputs['X_train'], y_train=train_test_split_task.outputs['y_train'])
    inference_task = inference_with_model(model_path=training_task.outputs['save_model'])


# Compile the pipeline to an IR YAML file
if __name__ == '__main__':
    compiler = kfp.compiler.Compiler()
    compiler.compile(basic_ml_pipeline, "basic_ml_pipeline_v5.yaml")

