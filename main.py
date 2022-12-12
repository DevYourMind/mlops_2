from flask import Flask
from flask_restx import Api, Resource
from werkzeug.datastructures import FileStorage
import models
import json
import glob
from pathlib import Path
import numpy as np
import os
from io import BytesIO
import psycopg2
from sqlalchemy import create_engine
import pandas as pd
import pickle


POSTGRES_HOST = "localhost"
POSTGRES_DB = "hwdb"
POSTGRES_USER = "hwdb_user"
POSTGRES_PASSWORD = "hwdb_passwd"
POSTGRES_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}"

app = Flask(__name__)
api = Api(app)

models_dir = Path('trained_models')
models_dir.mkdir(parents=True, exist_ok=True)

upload_parser = api.parser()
upload_parser.add_argument(
    'params',
    location='files',
    type=FileStorage,
    required=False
)
upload_parser.add_argument(
    'model',
    choices=['All', 'CatBoost', 'LightGBM'],
    location='args',
    required=True
)


@api.route(
    '/load_train',
    methods=['PUT', 'POST'],
    doc={'description': 'Загрузка параметров и обучение модели'})
@api.expect(upload_parser)
class FitModel(Resource):
    """
    Data generating and models' fitting.
    """

    def __init__(self, api=None, *args, **kwargs):
        super().__init__(api, *args, **kwargs)

    @api.doc(params={
        'model': 'Выберите модель',
        'params': 'Выберите .json файл с гиперпараметрами модели (доступно, если не выбрана опция "All")',
    })
    @api.response(200, 'Success')
    @api.response(400, 'Validation Error')
    @api.response(500, 'No such file or directory')
    def put(self):
        args = upload_parser.parse_args()
        model_name = args['model']
        try:
            pg_client = create_engine(POSTGRES_URL)
            self.df = pd.read_sql_query('SELECT * FROM public.data', pg_client)
            pg_client.dispose()
            X = self.df.drop('target', axis=1)
            y = self.df['target']
        except Exception:
            raise TypeError("No Data")

        if model_name == 'All':
            cb = models.train_dump_model(
                model_name='CatBoost', models_dir=models_dir, X=X, y=y)
            lg = models.train_dump_model(
                model_name='LightGBM', models_dir=models_dir, X=X, y=y)

            pg_client = create_engine(POSTGRES_URL)
            buffer = BytesIO()
            pickle.dump(cb, buffer)
            buffer.seek(0)
            pg_client.execute(
                f"""
                    INSERT INTO public.models
                    VALUES ('CatBoost', %s)
                """,
                (psycopg2.Binary(buffer.read()))
            )
            pg_client.dispose()

            pg_client = create_engine(POSTGRES_URL)
            buffer = BytesIO()
            pickle.dump(lg, buffer)
            buffer.seek(0)
            pg_client.execute(
                f"""
                    INSERT INTO public.models
                    VALUES ('LightGBM', %s)
                """,
                (psycopg2.Binary(buffer.read()))
            )
            pg_client.dispose()
            return 'All models have been fitted'
        else:
            params = args['params']
            if params is not None:
                params = json.load(params)
            try:
                model = models.train_dump_model(
                    model_name=model_name, models_dir=models_dir, X=X, y=y, params=params)
                pg_client = create_engine(POSTGRES_URL)
                buffer = BytesIO()
                pickle.dump(model, buffer)
                buffer.seek(0)
                pg_client.execute(
                    f"""
                        INSERT INTO public.models
                        VALUES ('{model_name}', %s)
                    """,
                    (psycopg2.Binary(buffer.read()))
                )
                pg_client.dispose()
            except TypeError:
                return 'Incorrect parametres'
            return f"Model {model_name} has been fitted"


@api.route('/show_models')
class ShowModels(Resource):
    """
    Show list of models.
    """
    @api.doc(
        params={
            'model_type': 'Выберите семейство обученных моделей'
        }
    )
    def get(self):
        pg_client = create_engine(POSTGRES_URL)
        models = pd.read_sql_query(
            'SELECT "model_name" FROM public.models;', pg_client).to_dict(orient="index")
        if len(models) == 0:
            return 'No models trained'
        else:
            return models


model_delete_parser = api.parser()
model_delete_parser.add_argument(
    'model_type',
    required=True,
    location='args',
    choices=models.get_trained_models(Path('trained_models')) + ["All"]
)


@api.route('/delete_models', methods=['GET'],
           doc={'description': 'Удаление моделей'})
class DeleteModels(Resource):
    """
    Delete chosen models.
    """
    @api.doc(
        params={
            'model_type': 'Выберите модель для удаления'
        }
    )
    @api.response(500, 'Model has been removed')
    def get(self):
        args = model_delete_parser.parse_args()
        model_to_delete = args['model_type']
        pg_client = create_engine(POSTGRES_URL)
        if model_to_delete == 'All':
            pg_client.execute(
                'TRUNCATE public.models; DELETE FROM public.models;'
            )
            pg_client.dispose()
            return 'All models have been deleted'
        pg_client.execute(
            f"""
                DELETE FROM public.models WHERE "model_name" = '{model_to_delete}';
            """
        )
        return f'Model {model_to_delete} has been removed'


predict_parser = api.parser()
predict_parser.add_argument(
    'model_type',
    required=True,
    location='args',
    choices=models.get_trained_models(models_dir)
)


@api.route('/predict', methods=['GET'],
           doc={'description': 'Сделать предсказание на сгенерированном сэмпле'})
@api.expect(predict_parser)
class Predict(Resource):
    """
    Make a prediction for sample data.
    """
    @api.doc(
        params={
            'model_type': 'Выберите модель для прогноза'
        }
    )
    def get(self):
        args = predict_parser.parse_args()
        model_to_predict = args['model_type']
        pg_client = create_engine(POSTGRES_URL)
        self.df = pd.read_sql_query('SELECT * FROM public.data', pg_client)
        pg_client.dispose()
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        X_sample = X.iloc[-1]
        y_sample = y.iloc[-1]
        model_raw = pg_client.execute(
            f"""SELECT "model_weights" FROM public.models WHERE "model_name" = '{model_to_predict}';""").fetchone()[0]
        model = pickle.loads(model_raw)
        prediction = model.predict(X_sample.reshape(1, -1))
        return f'sample: {y_sample}, prediction: {prediction}'


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
